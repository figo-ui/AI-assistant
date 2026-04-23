import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)
REASON_CLAUSE_RE = re.compile(r"\breason:\s*[^|]+")
COMORBIDITY_CLAUSE_RE = re.compile(r"\bcomorbidities:\s*[^|]+")
MEDICATION_CLAUSE_RE = re.compile(r"\bcurrent medications?:\s*[^|]+")
ALLERGY_CLAUSE_RE = re.compile(r"\bknown allergies?:\s*[^|]+")
ADMIN_CLAUSE_RE = re.compile(
    r"\b("
    r"encounter for symptom(?: \(procedure\))?|"
    r"general examination of patient(?: \(procedure\))?|"
    r"patient encounter procedure|"
    r"well child visit(?: \(procedure\))?|"
    r"death certification|"
    r"symptoms reported:|"
    r"hypertension follow up encounter"
    r")\b"
)
TEXT_ALIASES = ["symptom_text", "text", "symptoms", "input_text", "description"]
LABEL_ALIASES = ["label", "condition", "disease", "Disease", "target", "prognosis"]
MAP_LABEL_ALIASES = ["label", "label_id", "id", "code", "target"]
MAP_NAME_ALIASES = ["condition", "disease", "disease_name", "label_name", "name"]


def clean_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("|", " ")
    text = REASON_CLAUSE_RE.sub(" ", text)
    text = COMORBIDITY_CLAUSE_RE.sub(" ", text)
    text = MEDICATION_CLAUSE_RE.sub(" ", text)
    text = ALLERGY_CLAUSE_RE.sub(" ", text)
    text = ADMIN_CLAUSE_RE.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def strip_target_leakage(symptom_text: str, condition: str) -> str:
    text = clean_text(symptom_text)
    target = clean_text(condition)
    target_compact = re.sub(r"\s*\(.*?\)\s*", " ", str(condition).strip().lower())
    target_compact = clean_text(target_compact)
    for term in {target, target_compact}:
        if term and len(term) >= 4:
            text = re.sub(rf"\b{re.escape(term)}\b", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def detect_text_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    text_col = next((col for col in TEXT_ALIASES if col in df.columns), None)
    label_col = next((col for col in LABEL_ALIASES if col in df.columns), None)
    if not text_col or not label_col:
        raise ValueError(
            "Could not detect text/label columns. "
            "Expected one of text columns "
            f"{TEXT_ALIASES} and one of label columns {LABEL_ALIASES}."
        )
    return text_col, label_col


def load_label_mapping(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {normalize_label(k): str(v).strip() for k, v in data.items() if str(v).strip()}
        if isinstance(data, list):
            result: Dict[str, str] = {}
            for item in data:
                if isinstance(item, dict):
                    k = normalize_label(item.get("label") or item.get("id") or item.get("code"))
                    v = str(item.get("condition") or item.get("disease") or item.get("name") or "").strip()
                    if k and v:
                        result[k] = v
            return result
        raise ValueError("Unsupported JSON mapping format.")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        label_col = next((col for col in MAP_LABEL_ALIASES if col in df.columns), None)
        name_col = next((col for col in MAP_NAME_ALIASES if col in df.columns), None)
        if not label_col or not name_col:
            if len(df.columns) >= 2:
                label_col, name_col = df.columns[0], df.columns[1]
            else:
                raise ValueError("Mapping CSV must contain at least 2 columns (label, disease_name).")

        out: Dict[str, str] = {}
        for _, row in df.iterrows():
            k = normalize_label(row[label_col])
            v = str(row[name_col]).strip()
            if k and v:
                out[k] = v
        return out

    raise ValueError("Mapping file must be .csv or .json")


def is_numeric_label(label: str) -> bool:
    return bool(re.fullmatch(r"\d+", label))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess healthcare text dataset and optionally retrain text model."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV dataset.")
    parser.add_argument(
        "--output-csv",
        help="Path to save processed CSV (default: <input_stem>_processed.csv).",
    )
    parser.add_argument(
        "--mapping-file",
        help="Optional CSV/JSON mapping for numeric labels to disease names.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples per class to keep (ultra-rare classes are removed).",
    )
    parser.add_argument(
        "--label-prefix",
        default="Condition ",
        help="Prefix used when no mapping exists for numeric labels.",
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run retraining after preprocessing (default: true).",
    )
    parser.add_argument(
        "--model-out-dir",
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size for retraining.")
    parser.add_argument(
        "--model-type",
        choices=["lr", "xgb"],
        default="lr",
        help="Classifier backend for text model training (default: lr).",
    )
    parser.add_argument(
        "--rebalance-mode",
        choices=["none", "under_over", "smote_svd"],
        default="under_over",
        help="Rebalancing mode for text model training.",
    )
    parser.add_argument(
        "--majority-labels",
        default="COVID-19,Suspected COVID-19",
        help="Comma-separated majority class labels to cap during undersampling.",
    )
    parser.add_argument(
        "--majority-cap-fraction",
        type=float,
        default=0.20,
        help="Maximum fraction per majority label after undersampling.",
    )
    parser.add_argument(
        "--minority-target",
        type=int,
        default=200,
        help="Target minimum count for minority classes after oversampling.",
    )
    parser.add_argument(
        "--smote-svd-components",
        type=int,
        default=384,
        help="SVD components when using smote_svd rebalance mode.",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply calibrated probabilities (LR only).",
    )
    parser.add_argument(
        "--export-condition-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export condition name map JSON into model output directory (default: true).",
    )
    parser.add_argument(
        "--exclude-generic-conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop generic condition labels like 'Condition 275' after mapping (default: true).",
    )
    parser.add_argument(
        "--train-dialogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train dialogue intent model using MedQuAD dataset (default: false).",
    )
    parser.add_argument(
        "--dialogue-dataset",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dialogue" / "medquad_dialogue_pairs.csv"),
        help="Dialogue dataset CSV path.",
    )
    parser.add_argument(
        "--dialogue-max-samples",
        type=int,
        default=30000,
        help="Max samples when building MedQuAD dialogue CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_processed.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mapping = load_label_mapping(Path(args.mapping_file)) if args.mapping_file else {}

    df = pd.read_csv(input_path)
    text_col, label_col = detect_text_label_columns(df)
    work = df[[text_col, label_col]].copy()
    work.columns = ["symptom_text", "label_id"]

    work["symptom_text"] = work["symptom_text"].map(clean_text)
    work["label_id"] = work["label_id"].map(normalize_label)
    work = work[(work["symptom_text"] != "") & (work["label_id"] != "")].copy()

    before_rows = len(work)
    work = work.drop_duplicates(subset=["symptom_text", "label_id"]).reset_index(drop=True)
    dedup_rows = before_rows - len(work)

    counts = work["label_id"].value_counts()
    keep_labels = counts[counts >= args.min_samples].index
    removed_labels = int((counts < args.min_samples).sum())
    work = work[work["label_id"].isin(keep_labels)].reset_index(drop=True)

    if len(work) == 0:
        raise ValueError(
            "No data left after filtering rare classes. "
            "Reduce --min-samples or check dataset integrity."
        )

    def map_label_to_condition(label: str) -> str:
        if label in mapping:
            return mapping[label]
        if is_numeric_label(label):
            return f"{args.label_prefix}{label}".strip()
        return label

    work["condition"] = work["label_id"].map(map_label_to_condition)
    work["symptom_text"] = [
        strip_target_leakage(symptom_text=text, condition=condition)
        for text, condition in zip(work["symptom_text"], work["condition"])
    ]
    work = work[work["symptom_text"] != ""].reset_index(drop=True)

    # Keep columns trainer can consume directly.
    processed = work[["symptom_text", "condition", "label_id"]].copy()
    generic_rows_removed = 0
    generic_classes_removed = 0
    if args.exclude_generic_conditions:
        generic_mask = processed["condition"].astype(str).str.strip().str.match(GENERIC_CONDITION_RE)
        generic_rows_removed = int(generic_mask.sum())
        generic_classes_removed = int(processed.loc[generic_mask, "condition"].nunique())
        processed = processed.loc[~generic_mask].reset_index(drop=True)

        if len(processed) == 0:
            raise ValueError("All rows were removed by --exclude-generic-conditions.")

    processed.to_csv(output_csv, index=False)

    label_map = (
        processed[["label_id", "condition"]]
        .drop_duplicates()
        .sort_values(by="label_id")
        .reset_index(drop=True)
    )
    map_csv = output_csv.with_name(f"{output_csv.stem}_label_map.csv")
    map_json = output_csv.with_name(f"{output_csv.stem}_label_map.json")
    label_map.to_csv(map_csv, index=False)
    map_json.write_text(
        json.dumps(dict(zip(label_map["label_id"], label_map["condition"])), indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_rows": int(before_rows),
        "rows_after_dedup": int(before_rows - dedup_rows),
        "deduplicates_removed": int(dedup_rows),
        "rows_after_filter": int(len(processed)),
        "kept_classes": int(processed["label_id"].nunique()),
        "removed_ultra_rare_classes": int(removed_labels),
        "min_samples_threshold": int(args.min_samples),
        "mapping_entries_used": int(len(mapping)),
        "exclude_generic_conditions": bool(args.exclude_generic_conditions),
        "generic_rows_removed": int(generic_rows_removed),
        "generic_classes_removed": int(generic_classes_removed),
    }
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.export_condition_map:
        model_out_dir = Path(args.model_out_dir)
        model_out_dir.mkdir(parents=True, exist_ok=True)
        condition_map_path = model_out_dir / "condition_name_map.json"
        condition_map_path.write_text(
            json.dumps(dict(zip(label_map["label_id"], label_map["condition"])), indent=2),
            encoding="utf-8",
        )
        print(f"Condition map for runtime: {condition_map_path}")

    print("Preprocessing complete.")
    print(f"Processed CSV: {output_csv}")
    print(f"Label map CSV: {map_csv}")
    print(f"Label map JSON: {map_json}")
    print(f"Summary JSON: {summary_path}")
    print(json.dumps(summary, indent=2))

    if args.train:
        train_script = Path(__file__).with_name("train_text_model.py")
        cmd = [
            sys.executable,
            str(train_script),
            "--dataset",
            str(output_csv),
            "--out-dir",
            str(Path(args.model_out_dir)),
            "--test-size",
            str(args.test_size),
            "--model-type",
            str(args.model_type),
            "--rebalance-mode",
            str(args.rebalance_mode),
            "--majority-cap-fraction",
            str(args.majority_cap_fraction),
            "--minority-target",
            str(args.minority_target),
            "--smote-svd-components",
            str(args.smote_svd_components),
        ]
        if args.calibrate:
            cmd.append("--calibrate")
        majority_labels = [v.strip() for v in str(args.majority_labels).split(",") if v.strip()]
        for lbl in majority_labels:
            cmd.extend(["--majority-label", lbl])
        print("Running retraining command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.train_dialogue:
        dialogue_dataset_path = Path(args.dialogue_dataset)
        if not dialogue_dataset_path.exists():
            build_dialogue_script = Path(__file__).with_name("build_medquad_dialogue_dataset.py")
            build_cmd = [
                sys.executable,
                str(build_dialogue_script),
                "--output-csv",
                str(dialogue_dataset_path),
                "--max-samples",
                str(args.dialogue_max_samples),
            ]
            print("Building dialogue dataset:")
            print(" ".join(build_cmd))
            subprocess.run(build_cmd, check=True)

        dialogue_train_script = Path(__file__).with_name("train_dialogue_model.py")
        dialogue_cmd = [
            sys.executable,
            str(dialogue_train_script),
            "--dataset",
            str(dialogue_dataset_path),
            "--out-dir",
            str(Path(args.model_out_dir)),
            "--test-size",
            str(args.test_size),
        ]
        print("Running dialogue model training:")
        print(" ".join(dialogue_cmd))
        subprocess.run(dialogue_cmd, check=True)


if __name__ == "__main__":
    main()
