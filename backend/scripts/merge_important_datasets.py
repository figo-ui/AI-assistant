import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)
LEAKAGE_CLAUSE_RE = re.compile(
    r"\b(reason:|comorbidities:|current medications?:|known allergies?:)\s*[^|]+",
    re.IGNORECASE,
)
TEXT_ALIASES = ["symptom_text", "text", "symptoms", "input_text", "description"]
LABEL_ALIASES = ["condition", "label", "disease", "Disease", "target", "prognosis"]


def clean_text(value: str) -> str:
    text = str(value).strip().lower()
    text = LEAKAGE_CLAUSE_RE.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def detect_text_label_columns(df: pd.DataFrame) -> tuple[str, str]:
    text_col = next((col for col in TEXT_ALIASES if col in df.columns), None)
    label_col = next((col for col in LABEL_ALIASES if col in df.columns), None)
    if not text_col or not label_col:
        raise ValueError(
            "Could not detect text/label columns. Expected one of "
            f"text={TEXT_ALIASES} and label={LABEL_ALIASES}."
        )
    return text_col, label_col


def load_dataset(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    text_col, label_col = detect_text_label_columns(df)
    out = df[[text_col, label_col]].copy()
    out.columns = ["symptom_text", "condition"]
    out["symptom_text"] = out["symptom_text"].map(clean_text)
    out["condition"] = out["condition"].astype(str).str.strip()
    out["source_dataset"] = source_name
    out = out[(out["symptom_text"] != "") & (out["condition"] != "")]
    out = out[~out["condition"].astype(str).str.strip().str.match(GENERIC_CONDITION_RE)]
    return out


def summarize_sources(frames: List[pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for frame in frames:
        source = str(frame["source_dataset"].iloc[0]) if len(frame) else "unknown"
        summary[source] = {
            "rows": int(len(frame)),
            "unique_conditions": int(frame["condition"].nunique()),
        }
    return summary


def main() -> None:
    base_data = Path(__file__).resolve().parents[2] / "data"
    default_output = base_data / "integrated_important_symptom_condition.csv"
    default_summary = base_data / "integrated_important_symptom_condition_summary.json"

    parser = argparse.ArgumentParser(description="Merge important healthcare text datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("PATH", "SOURCE_NAME"),
        help="Input dataset path and logical source name. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output CSV path for merged dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(default_summary),
        help="Output JSON path for merge summary.",
    )
    args = parser.parse_args()

    datasets = args.dataset or [
        [str(base_data / "expanded_symptom_condition.csv"), "expanded_symptom_condition"],
        [str(base_data / "synthea_symptom_condition.csv"), "synthea_symptom_condition"],
    ]

    frames: List[pd.DataFrame] = []
    for path_str, source_name in datasets:
        frames.append(load_dataset(Path(path_str), source_name=source_name))

    merged = pd.concat(frames, ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    dedup_removed = before_dedup - len(merged)

    counts = merged["condition"].value_counts()
    merged["condition_count"] = merged["condition"].map(counts).astype(int)

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_path, index=False)

    summary = {
        "input_sources": summarize_sources(frames),
        "merged_rows_before_dedup": int(before_dedup),
        "merged_rows_after_dedup": int(len(merged)),
        "deduplicates_removed": int(dedup_removed),
        "unique_conditions": int(merged["condition"].nunique()),
        "rows_with_condition_count_lt_2": int((merged["condition_count"] < 2).sum()),
        "output_csv": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
