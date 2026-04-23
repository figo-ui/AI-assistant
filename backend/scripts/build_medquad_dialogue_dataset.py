import argparse
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET

import pandas as pd


MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = WHITESPACE_RE.sub(" ", str(value or "").strip())
    return text


def download_if_needed(url: str, dest: Path, refresh: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return
    if refresh and dest.exists():
        dest.unlink()
    print(f"Downloading MedQuAD archive -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract_zip(zip_path: Path, extract_root: Path, refresh: bool = False) -> Path:
    if refresh and extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    candidates = [p for p in extract_root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError("No extracted MedQuAD directory found.")

    dataset_root = next((p for p in candidates if p.name.lower().startswith("medquad-")), candidates[0])
    return dataset_root


def parse_medquad_xml(root: Path, min_answer_chars: int) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    xml_paths = sorted(root.rglob("*.xml"))

    for xml_path in xml_paths:
        try:
            tree = ET.parse(xml_path)
            doc = tree.getroot()
        except ET.ParseError:
            continue

        source = clean_text(doc.attrib.get("source", "unknown")) or "unknown"
        focus_node = doc.find("Focus")
        focus = clean_text("".join(focus_node.itertext())) if focus_node is not None else ""

        qa_pairs = doc.find("QAPairs")
        if qa_pairs is None:
            continue

        for pair in qa_pairs.findall("QAPair"):
            qnode = pair.find("Question")
            anode = pair.find("Answer")
            if qnode is None or anode is None:
                continue

            question = clean_text("".join(qnode.itertext()))
            answer = clean_text("".join(anode.itertext()))
            qtype = clean_text(qnode.attrib.get("qtype", "general")).lower() or "general"

            if not question or not answer:
                continue
            if len(answer) < min_answer_chars:
                continue

            rows.append(
                {
                    "user_text": question,
                    "assistant_text": answer,
                    "intent": qtype,
                    "focus": focus,
                    "source": source,
                }
            )

    if not rows:
        raise ValueError("No usable dialogue rows found in MedQuAD parse.")

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)
    return frame


def main() -> None:
    base_data = Path(__file__).resolve().parents[2] / "data"
    default_output_csv = base_data / "dialogue" / "medquad_dialogue_pairs.csv"
    default_summary_json = base_data / "dialogue" / "medquad_dialogue_summary.json"
    default_zip_path = base_data / "open_datasets" / "raw" / "medquad-master.zip"
    default_extract_dir = base_data / "open_datasets" / "extracted" / "medquad-master"

    parser = argparse.ArgumentParser(description="Download and build a MedQuAD dialogue dataset CSV.")
    parser.add_argument("--output-csv", default=str(default_output_csv), help="Output CSV path.")
    parser.add_argument("--summary-output", default=str(default_summary_json), help="Summary JSON path.")
    parser.add_argument("--archive-path", default=str(default_zip_path), help="Downloaded archive path.")
    parser.add_argument("--extract-dir", default=str(default_extract_dir), help="Extraction directory.")
    parser.add_argument("--max-samples", type=int, default=30000, help="Optional cap on final rows.")
    parser.add_argument("--min-answer-chars", type=int, default=80, help="Minimum answer length to keep.")
    parser.add_argument(
        "--refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Redownload and re-extract archive.",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    summary_output = Path(args.summary_output)
    archive_path = Path(args.archive_path)
    extract_dir = Path(args.extract_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    download_if_needed(MEDQUAD_ZIP_URL, archive_path, refresh=args.refresh)
    dataset_root = extract_zip(archive_path, extract_dir, refresh=args.refresh)
    frame = parse_medquad_xml(dataset_root, min_answer_chars=args.min_answer_chars)

    raw_count = len(frame)
    if args.max_samples and raw_count > args.max_samples:
        frame = frame.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    frame.to_csv(output_csv, index=False)

    summary = {
        "source_dataset": "MedQuAD",
        "source_url": MEDQUAD_ZIP_URL,
        "license": "CC BY 4.0",
        "raw_rows_after_cleaning": int(raw_count),
        "final_rows": int(len(frame)),
        "unique_intents": int(frame["intent"].nunique()),
        "intent_distribution_top10": frame["intent"].value_counts().head(10).to_dict(),
        "source_distribution": frame["source"].value_counts().to_dict(),
        "output_csv": str(output_csv),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dialogue dataset: {output_csv}")
    print(f"Saved summary: {summary_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
