import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are a healthcare triage assistant. "
    "Return exactly one JSON object with predictions, risk_level, red_flags, recommendation, and reasoning. "
    "Do not add prose before or after the JSON. "
    "Do not claim a final diagnosis."
)


def _safe_str(value) -> str:
    return str(value or "").strip()


def build_assistant_response(row: pd.Series) -> str:
    condition = _safe_str(row.get("condition"))
    urgency = _safe_str(row.get("urgency")) or "Medium"
    reasoning = _safe_str(row.get("reasoning")) or "Pattern matched from symptom cluster and clinical safety rules."
    differential = row.get("differential")
    if isinstance(differential, str) and differential.strip():
        try:
            diff_items = json.loads(differential)
        except Exception:
            diff_items = [item.strip() for item in differential.split("|") if item.strip()]
    else:
        diff_items = []
    if not diff_items:
        diff_items = [
            condition,
            "Alternative differential based on symptom overlap",
            "Alternative differential based on risk profile",
        ]
    while len(diff_items) < 3:
        diff_items.append("Additional differential candidate")

    recommendation = _safe_str(row.get("recommendation"))
    if not recommendation:
        if urgency == "High":
            recommendation = "Seek emergency in-person care immediately."
        elif urgency == "Medium":
            recommendation = "Arrange same-day or next-day clinical evaluation."
        else:
            recommendation = "Monitor symptoms and arrange routine care if symptoms persist or worsen."

    red_flags = _safe_str(row.get("red_flags")) or "Chest pain, trouble breathing, confusion, new one-sided weakness, or slurred speech."

    red_flag_items = [item.strip() for item in red_flags.split("|") if item.strip()]
    if not red_flag_items:
        red_flag_items = [item.strip() for item in red_flags.split(",") if item.strip()]
    if not red_flag_items:
        red_flag_items = ["seek urgent clinical review if symptoms worsen"]

    payload = {
        "predictions": [
            {"condition": diff_items[0], "probability": 0.62},
            {"condition": diff_items[1], "probability": 0.23},
            {"condition": diff_items[2], "probability": 0.15},
        ],
        "risk_level": urgency.title() if urgency.title() in {"Low", "Medium", "High"} else "Medium",
        "red_flags": red_flag_items[:5],
        "recommendation": recommendation,
        "reasoning": reasoning,
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare supervised JSONL for triage LLM fine-tuning.")
    parser.add_argument("--input-csv", required=True, help="CSV with symptom_text, condition and optional urgency/reasoning.")
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(__file__).resolve().parents[1] / "data" / "triage_llm_dataset.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path).fillna("")
    if "symptom_text" not in df.columns or "condition" not in df.columns:
        raise ValueError("Input CSV must include symptom_text and condition columns.")

    with out_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            user_text = _safe_str(row.get("symptom_text"))
            if not user_text:
                continue
            assistant_text = build_assistant_response(pd.Series(row))
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Saved LLM training dataset: {out_path}")


if __name__ == "__main__":
    main()
