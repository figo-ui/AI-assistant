import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "open_datasets"
    / "extracted"
    / "synthea-10k-covid19-csv"
    / "10k_synthea_covid19_csv"
)
DEFAULT_CASES_OUT = Path(__file__).resolve().parents[2] / "data" / "synthea_case_submissions.csv"
DEFAULT_TRAINING_OUT = Path(__file__).resolve().parents[2] / "data" / "synthea_symptom_condition.csv"
DEFAULT_SUMMARY_OUT = Path(__file__).resolve().parents[2] / "data" / "synthea_import_summary.json"

VITAL_KEYWORDS = {
    "blood pressure",
    "body weight",
    "body height",
    "body mass index",
    "heart rate",
    "respiratory rate",
    "body temperature",
    "oxygen saturation",
}

LAB_KEYWORDS = {
    "glucose",
    "creatinine",
    "urea nitrogen",
    "potassium",
    "sodium",
    "hemoglobin a1c",
    "cholesterol",
    "triglyceride",
    "platelets",
    "hemoglobin",
    "leukocytes",
}

ADMIN_DESCRIPTION_PATTERNS = (
    "encounter for symptom",
    "well child visit",
    "general examination of patient",
    "death certification",
    "record artifact",
)

SYMPTOM_HINTS = (
    "pain",
    "fever",
    "cough",
    "fatigue",
    "weakness",
    "shortness of breath",
    "breathlessness",
    "wheezing",
    "headache",
    "sore throat",
    "runny nose",
    "congestion",
    "vomiting",
    "nausea",
    "diarrhea",
    "rash",
    "itch",
    "swelling",
    "dizziness",
    "loss of smell",
    "loss of taste",
    "chills",
    "malaise",
)

DIAGNOSIS_FILTER_HINTS = (
    "suspected ",
    "disorder",
    "disease",
    "syndrome",
    "infection",
    "cancer",
    "fracture",
    "sprain",
    "arrest",
)

WHITESPACE_RE = re.compile(r"\s+")


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_text(value: str) -> str:
    text = _safe_text(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _dedupe_non_empty(values: Iterable[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = _normalize_text(text)
    return any(keyword in low for keyword in keywords)


def _is_admin_or_death_row(description: str, reason: str) -> bool:
    combined = f"{description} {reason}".lower()
    return _contains_any(combined, ADMIN_DESCRIPTION_PATTERNS)


def _is_symptom_description(description: str) -> bool:
    low = _normalize_text(description)
    if not low:
        return False
    if "(finding)" in low:
        return True
    if _contains_any(low, SYMPTOM_HINTS):
        return True
    return False


def _is_probable_diagnosis(description: str) -> bool:
    low = _normalize_text(description)
    if not low:
        return False
    if _is_symptom_description(low):
        return False
    return _contains_any(low, DIAGNOSIS_FILTER_HINTS) or not low.endswith("(finding)")


def _load_observations_condensed(path: Path) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION", "VALUE", "UNITS", "TYPE"],
        chunksize=250_000,
    ):
        chunk = chunk.fillna("")
        chunk["DESCRIPTION"] = chunk["DESCRIPTION"].astype(str)
        keep = chunk["DESCRIPTION"].map(lambda text: _contains_any(text, VITAL_KEYWORDS | LAB_KEYWORDS))
        filtered = chunk.loc[keep].copy()
        if filtered.empty:
            continue
        filtered["DATE"] = _as_datetime(filtered["DATE"])
        filtered = filtered.dropna(subset=["DATE"])
        chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION", "VALUE", "UNITS", "TYPE"])
    return pd.concat(chunks, ignore_index=True)


def _build_history_payload(observations: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    if observations.empty:
        return {"vitals_history": [], "labs_history": []}

    obs = observations.sort_values("DATE")
    vitals: List[Dict[str, str]] = []
    labs: List[Dict[str, str]] = []
    for _, row in obs.iterrows():
        description = _safe_text(row["DESCRIPTION"])
        item = {
            "date": row["DATE"].isoformat(),
            "name": description,
            "value": _safe_text(row["VALUE"]),
            "units": _safe_text(row["UNITS"]),
        }
        if _contains_any(description, VITAL_KEYWORDS):
            vitals.append(item)
        if _contains_any(description, LAB_KEYWORDS):
            labs.append(item)

    return {"vitals_history": vitals[-12:], "labs_history": labs[-12:]}


def _observation_signal_rows(observations: pd.DataFrame) -> List[str]:
    if observations.empty:
        return []
    signals: List[str] = []
    for _, row in observations.iterrows():
        description = _normalize_text(row["DESCRIPTION"])
        raw_value = _safe_text(row["VALUE"])
        try:
            numeric_value = float(raw_value)
        except Exception:
            numeric_value = None

        if "body temperature" in description and numeric_value is not None and numeric_value >= 37.8:
            signals.append("fever")
        elif "oxygen saturation" in description and numeric_value is not None and numeric_value <= 92:
            signals.append("low oxygen saturation")
        elif "respiratory rate" in description and numeric_value is not None and numeric_value >= 22:
            signals.append("fast breathing")
        elif "heart rate" in description and numeric_value is not None and numeric_value >= 100:
            signals.append("rapid heartbeat")
        elif "pain severity" in description and numeric_value is not None:
            signals.append(f"pain score {int(round(numeric_value))}/10")
    return _dedupe_non_empty(signals, limit=6)


def _choose_label(reason: str, encounter_conditions: List[str]) -> str:
    if reason and _is_probable_diagnosis(reason) and not _is_admin_or_death_row("", reason):
        return reason

    for description in encounter_conditions:
        if _is_probable_diagnosis(description) and not _is_admin_or_death_row("", description):
            return description
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Synthea into leakage-reduced symptom narratives.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--out-cases", default=str(DEFAULT_CASES_OUT))
    parser.add_argument("--out-training", default=str(DEFAULT_TRAINING_OUT))
    parser.add_argument("--out-summary", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Synthea input directory not found: {input_dir}")

    patients = pd.read_csv(
        input_dir / "patients.csv",
        usecols=["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY", "STATE"],
    ).rename(columns={"Id": "PATIENT"})
    patients["BIRTHDATE"] = _as_datetime(patients["BIRTHDATE"])

    encounters = pd.read_csv(
        input_dir / "encounters.csv",
        usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"],
    ).rename(columns={"Id": "ENCOUNTER"})
    encounters["START"] = _as_datetime(encounters["START"])
    encounters = encounters.dropna(subset=["START"]).sort_values("START")

    conditions = pd.read_csv(
        input_dir / "conditions.csv",
        usecols=["PATIENT", "ENCOUNTER", "DESCRIPTION"],
    ).fillna("")

    observations = _load_observations_condensed(input_dir / "observations.csv")

    patients_by_id = patients.set_index("PATIENT", drop=False)
    conditions_by_encounter = {key: frame for key, frame in conditions.groupby("ENCOUNTER")}
    observations_by_encounter = {key: frame for key, frame in observations.groupby("ENCOUNTER")} if not observations.empty else {}

    rows: List[Dict[str, object]] = []
    skipped_admin_or_death = 0
    skipped_without_label = 0
    skipped_without_symptoms = 0

    for _, encounter in encounters.iterrows():
        if args.max_cases and len(rows) >= args.max_cases:
            break

        encounter_description = _safe_text(encounter.get("DESCRIPTION"))
        reason = _safe_text(encounter.get("REASONDESCRIPTION"))
        if _is_admin_or_death_row(encounter_description, reason):
            skipped_admin_or_death += 1
            continue

        encounter_conditions_frame = conditions_by_encounter.get(encounter["ENCOUNTER"], pd.DataFrame())
        encounter_conditions = (
            _dedupe_non_empty(encounter_conditions_frame["DESCRIPTION"].tolist(), limit=20)
            if not encounter_conditions_frame.empty
            else []
        )
        label = _choose_label(reason=reason, encounter_conditions=encounter_conditions)
        if not label:
            skipped_without_label += 1
            continue

        symptom_conditions = [
            description
            for description in encounter_conditions
            if _is_symptom_description(description) and _normalize_text(description) != _normalize_text(label)
        ]
        encounter_observations = observations_by_encounter.get(encounter["ENCOUNTER"], pd.DataFrame())
        observation_signals = _observation_signal_rows(encounter_observations)
        symptom_clauses = _dedupe_non_empty(symptom_conditions + observation_signals, limit=8)
        if len(symptom_clauses) < 1:
            skipped_without_symptoms += 1
            continue

        patient_record = patients_by_id.loc[encounter["PATIENT"]] if encounter["PATIENT"] in patients_by_id.index else None
        age = None
        gender = ""
        race = ""
        ethnicity = ""
        state = ""
        if patient_record is not None:
            birthdate = patient_record["BIRTHDATE"]
            if pd.notna(birthdate):
                age = int((encounter["START"] - birthdate).days / 365.25)
            gender = _safe_text(patient_record["GENDER"])
            race = _safe_text(patient_record["RACE"])
            ethnicity = _safe_text(patient_record["ETHNICITY"])
            state = _safe_text(patient_record["STATE"])

        symptom_text = "Symptoms reported: " + ", ".join(symptom_clauses[:6])
        history_payload = _build_history_payload(
            encounter_observations if isinstance(encounter_observations, pd.DataFrame) else pd.DataFrame()
        )

        metadata = {
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": ethnicity,
            "state": state,
            "encounter_class": _safe_text(encounter.get("ENCOUNTERCLASS")),
            "vitals_history": history_payload["vitals_history"],
            "labs_history": history_payload["labs_history"],
        }

        rows.append(
            {
                "case_external_id": f"{encounter['PATIENT']}:{encounter['ENCOUNTER']}",
                "patient_id": encounter["PATIENT"],
                "encounter_id": encounter["ENCOUNTER"],
                "encounter_start": encounter["START"].isoformat(),
                "symptom_text": symptom_text,
                "condition": label,
                "consent_given": True,
                "metadata": json.dumps(metadata, ensure_ascii=True),
            }
        )

    if not rows:
        raise ValueError("No clean Synthea cases were generated. Adjust import filters or source data.")

    cases_df = pd.DataFrame(rows)
    training_df = cases_df[["symptom_text", "condition"]].drop_duplicates().reset_index(drop=True)

    out_cases = Path(args.out_cases)
    out_training = Path(args.out_training)
    out_summary = Path(args.out_summary)
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    out_training.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    cases_df.to_csv(out_cases, index=False)
    training_df.to_csv(out_training, index=False)

    summary = {
        "input_dir": str(input_dir),
        "cases_generated": int(len(cases_df)),
        "training_rows": int(len(training_df)),
        "unique_conditions": int(training_df["condition"].nunique()),
        "skipped_admin_or_death": int(skipped_admin_or_death),
        "skipped_without_label": int(skipped_without_label),
        "skipped_without_symptoms": int(skipped_without_symptoms),
        "narrative_style": "symptom_only",
        "outputs": {
            "cases_csv": str(out_cases),
            "training_csv": str(out_training),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
