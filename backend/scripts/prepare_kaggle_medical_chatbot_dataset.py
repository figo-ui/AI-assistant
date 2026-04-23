import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


TAG_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\s\+\/]*")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = str(value or "").strip()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def normalize_label(value) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "+1", "positive", "pos"}:
        return "1"
    if text in {"-1", "-1.0", "negative", "neg"}:
        return "-1"
    return text


def parse_tags(raw: str) -> List[str]:
    text = clean_text(raw).strip("[]")
    if not text:
        return []
    tags = [clean_text(t).strip("'\"").lower() for t in TAG_TOKEN_RE.findall(text)]
    return [t for t in tags if t]


def build_preference_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    pairs: List[Tuple[str, str, str, str]] = []
    grouped = df.groupby("user_text", sort=False)
    for user_text, group in grouped:
        pos = group[group["label"] == "1"]["assistant_text"].drop_duplicates().tolist()
        neg = group[group["label"] == "-1"]["assistant_text"].drop_duplicates().tolist()
        tags = " | ".join(group["tags"].dropna().astype(str).drop_duplicates().tolist())
        if not pos or not neg:
            continue
        for chosen in pos:
            for rejected in neg:
                pairs.append((user_text, chosen, rejected, tags))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Kaggle medical-chatbot dataset into (1) positive QA rows and "
            "(2) preference pairs for optional, non-production experiments."
        )
    )
    parser.add_argument(
        "--train-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "kaggle"
            / "raw"
            / "medical-chatbot-dataset"
            / "train_data_chatbot.csv"
        ),
        help="Path to train_data_chatbot.csv",
    )
    parser.add_argument(
        "--validation-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "kaggle"
            / "raw"
            / "medical-chatbot-dataset"
            / "validation_data_chatbot.csv"
        ),
        help="Path to validation_data_chatbot.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "dialogue"
            / "experimental_unknown_license"
        ),
        help="Output directory for processed files.",
    )
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    validation_csv = Path(args.validation_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not validation_csv.exists():
        raise FileNotFoundError(
            "Missing Kaggle chatbot CSV files. Expected train and validation CSV paths."
        )

    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(validation_csv, low_memory=False)
    raw_df = pd.concat([train_df, val_df], ignore_index=True)

    required = ["short_question", "short_answer", "tags", "label"]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Input CSVs missing required columns: {missing}")

    df = raw_df[required].copy()
    df = df.rename(
        columns={
            "short_question": "user_text",
            "short_answer": "assistant_text",
        }
    )
    df["user_text"] = df["user_text"].map(clean_text)
    df["assistant_text"] = df["assistant_text"].map(clean_text)
    df["tags"] = df["tags"].astype(str).map(lambda v: ", ".join(parse_tags(v)))
    df["label"] = df["label"].map(normalize_label)
    df = df[(df["user_text"] != "") & (df["assistant_text"] != "")].copy()
    df = df[df["label"].isin(["1", "-1"])].reset_index(drop=True)

    positive_qa = (
        df[df["label"] == "1"][["user_text", "assistant_text", "tags"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    positive_qa["source"] = "kaggle_medical_chatbot_unknown_license"

    pairs = build_preference_pairs(df)
    preference_df = pd.DataFrame(
        pairs,
        columns=["user_text", "chosen_answer", "rejected_answer", "tags"],
    )
    if len(preference_df) > 0:
        preference_df = preference_df.drop_duplicates().reset_index(drop=True)
    preference_df["source"] = "kaggle_medical_chatbot_unknown_license"

    summary = {
        "license_status": "unknown",
        "production_use_recommendation": "do_not_use_until_license_is_clarified",
        "input_rows": int(len(raw_df)),
        "rows_after_cleaning": int(len(df)),
        "label_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        },
        "unique_questions": int(df["user_text"].nunique()),
        "positive_qa_rows": int(len(positive_qa)),
        "preference_pairs_rows": int(len(preference_df)),
        "questions_with_both_pos_neg_labels": int(
            (df.groupby("user_text")["label"].nunique() >= 2).sum()
        ),
        "notes": [
            "This dataset appears to contain paired positive and negative answers per question.",
            "Use positive rows for realism experiments and preference pairs for response-quality ranking experiments.",
            "Do not ship to client production workflows without explicit license confirmation from the publisher.",
        ],
    }

    positive_path = out_dir / "kaggle_unknown_positive_qa.csv"
    preference_path = out_dir / "kaggle_unknown_preference_pairs.csv"
    summary_path = out_dir / "kaggle_unknown_dataset_summary.json"

    positive_qa.to_csv(positive_path, index=False)
    preference_df.to_csv(preference_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved positive QA rows:", positive_path)
    print("Saved preference pairs:", preference_path)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
