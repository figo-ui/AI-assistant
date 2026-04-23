import argparse
import ast
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [text]
    return [value]


def _parse_ddx_evidence_token(token: str) -> str:
    parts = str(token).split("_@_")
    return parts[0].strip()


def build_ddxplus_symptom_condition(
    *,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    ds = load_dataset("aai530-group6/ddxplus", split="train", streaming=True)
    if max_rows > 0:
        ds = ds.shuffle(seed=seed, buffer_size=min(max_rows * 2, 10000))
        ds = itertools.islice(ds, max_rows)

    evidence_json_path = hf_hub_download(
        repo_id="aai530-group6/ddxplus",
        filename="release_evidences.json",
        repo_type="dataset",
    )
    evidence_lookup: Dict[str, Dict[str, Any]] = json.loads(Path(evidence_json_path).read_text(encoding="utf-8"))

    rows: List[Dict[str, str]] = []
    for item in ds:
        pathology = str(item.get("PATHOLOGY", "")).strip()
        if not pathology:
            continue

        sex = str(item.get("SEX", "")).strip()
        age = item.get("AGE", "")
        initial_token = str(item.get("INITIAL_EVIDENCE", "")).strip()
        evidence_tokens = _ensure_list(item.get("EVIDENCES"))

        token_set = []
        if initial_token:
            token_set.append(initial_token)
        token_set.extend(str(v).strip() for v in evidence_tokens if str(v).strip())

        phr = []
        seen = set()
        for raw_tok in token_set:
            base = _parse_ddx_evidence_token(raw_tok)
            if not base or base in seen:
                continue
            seen.add(base)
            meta = evidence_lookup.get(base, {})
            q_en = str(meta.get("question_en", "")).strip()
            if q_en:
                phr.append(q_en)
            if len(phr) >= 14:
                break

        if not phr:
            continue

        symptom_text = clean_text(
            f"patient age {age}, sex {sex}. "
            + " ".join(phr)
        )
        if not symptom_text:
            continue

        rows.append(
            {
                "symptom_text": symptom_text,
                "condition": pathology,
                "source_dataset": "ddxplus",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    return out


def build_medqa_symptom_condition(*, max_rows: int, seed: int) -> pd.DataFrame:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="train", streaming=True)
    if max_rows > 0:
        ds = ds.shuffle(seed=seed, buffer_size=min(max_rows * 2, 5000))
        ds = itertools.islice(ds, max_rows)

    diagnosis_cues = (
        "most likely diagnosis",
        "most likely condition",
        "most likely cause",
        "diagnosis is",
        "likely diagnosis",
        "underlying diagnosis",
    )
    medication_like_cues = (" mg", "tablet", "capsule", "intravenous", "iv ", "dose", "q12h")

    rows: List[Dict[str, str]] = []
    for item in ds:
        sent1 = str(item.get("sent1", "")).strip()
        sent2 = str(item.get("sent2", "")).strip()
        question = clean_text(f"{sent1} {sent2}".strip())
        if not question:
            continue
        if not any(cue in question for cue in diagnosis_cues):
            continue

        options = [
            str(item.get("ending0", "")).strip(),
            str(item.get("ending1", "")).strip(),
            str(item.get("ending2", "")).strip(),
            str(item.get("ending3", "")).strip(),
        ]
        try:
            label_idx = int(item.get("label", -1))
        except Exception:
            continue
        if label_idx < 0 or label_idx >= len(options):
            continue

        condition = options[label_idx]
        if not condition:
            continue
        cond_l = condition.lower()
        if any(cue in cond_l for cue in medication_like_cues):
            continue

        rows.append(
            {
                "symptom_text": question,
                "condition": condition,
                "source_dataset": "medqa_usmle",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    return out


def build_medmcqa_dialogue(*, max_rows: int, seed: int) -> pd.DataFrame:
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
    if max_rows > 0:
        ds = ds.shuffle(seed=seed, buffer_size=min(max_rows * 2, 10000))
        ds = itertools.islice(ds, max_rows)

    rows: List[Dict[str, str]] = []
    for item in ds:
        question = clean_text(item.get("question", ""))
        if not question:
            continue
        options = [
            str(item.get("opa", "")).strip(),
            str(item.get("opb", "")).strip(),
            str(item.get("opc", "")).strip(),
            str(item.get("opd", "")).strip(),
        ]
        try:
            correct = int(item.get("cop", -1))
        except Exception:
            continue
        # medmcqa cop is often 1..4
        idx = correct - 1 if correct in [1, 2, 3, 4] else correct
        if idx < 0 or idx >= len(options):
            continue
        answer = clean_text(options[idx])
        if not answer:
            continue
        rows.append(
            {
                "user_text": question,
                "assistant_text": answer,
                "intent": "clinical_qa",
                "source": "medmcqa",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)
    return out


def build_pubmedqa_dialogue(*, max_rows: int, seed: int) -> pd.DataFrame:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True)
    if max_rows > 0:
        ds = ds.shuffle(seed=seed, buffer_size=min(max_rows * 2, 5000))
        ds = itertools.islice(ds, max_rows)

    rows: List[Dict[str, str]] = []
    for item in ds:
        question = clean_text(item.get("question", ""))
        long_answer = str(item.get("long_answer", "")).strip()
        final_decision = str(item.get("final_decision", "")).strip().lower()
        if not question or not long_answer:
            continue

        assistant = clean_text(f"{long_answer} final decision: {final_decision}")
        if not assistant:
            continue
        rows.append(
            {
                "user_text": question,
                "assistant_text": assistant,
                "intent": "evidence_qa",
                "source": "pubmedqa",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)
    return out


def _cap_per_class(df: pd.DataFrame, class_col: str, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0:
        return df
    return (
        df.groupby(class_col, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), cap), random_state=seed))
        .reset_index(drop=True)
    )


def _safe_read_csv(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(usecols))
    return pd.read_csv(path, usecols=list(usecols))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build expanded open/free medical datasets for text classification and dialogue training."
        )
    )
    parser.add_argument(
        "--base-text-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "kaggle"
            / "processed"
            / "integrated_important_plus_kaggle_processed_min5.csv"
        ),
        help="Existing processed text training CSV with symptom_text,condition columns.",
    )
    parser.add_argument(
        "--base-dialogue-csv",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dialogue" / "medquad_dialogue_pairs.csv"),
        help="Existing dialogue dataset CSV to merge with new QA dialogue rows.",
    )
    parser.add_argument(
        "--out-text-csv",
        default=str(Path(__file__).resolve().parents[2] / "data" / "expanded_symptom_condition.csv"),
        help="Output merged text dataset CSV.",
    )
    parser.add_argument(
        "--out-dialogue-csv",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dialogue" / "expanded_medical_dialogue.csv"),
        help="Output merged dialogue dataset CSV.",
    )
    parser.add_argument("--ddxplus-max-rows", type=int, default=150000, help="Max DDXPlus rows to include.")
    parser.add_argument("--medqa-max-rows", type=int, default=12000, help="Max MedQA rows to include.")
    parser.add_argument("--medmcqa-max-rows", type=int, default=100000, help="Max MedMCQA rows to include.")
    parser.add_argument("--pubmedqa-max-rows", type=int, default=50000, help="Max PubMedQA rows to include.")
    parser.add_argument("--per-condition-cap", type=int, default=1200, help="Cap rows per condition after merge.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_text_csv = Path(args.out_text_csv)
    out_dialogue_csv = Path(args.out_dialogue_csv)
    out_text_csv.parent.mkdir(parents=True, exist_ok=True)
    out_dialogue_csv.parent.mkdir(parents=True, exist_ok=True)

    base_text = _safe_read_csv(Path(args.base_text_csv), ["symptom_text", "condition"])
    base_text["source_dataset"] = "base_existing"

    ddxplus_df = build_ddxplus_symptom_condition(max_rows=int(args.ddxplus_max_rows), seed=int(args.seed))
    medqa_df = build_medqa_symptom_condition(max_rows=int(args.medqa_max_rows), seed=int(args.seed))

    text_merged = pd.concat([base_text, ddxplus_df, medqa_df], ignore_index=True)
    text_merged["symptom_text"] = text_merged["symptom_text"].map(clean_text)
    text_merged["condition"] = text_merged["condition"].astype(str).str.strip()
    text_merged = text_merged[(text_merged["symptom_text"] != "") & (text_merged["condition"] != "")]
    text_merged = text_merged.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    text_merged = _cap_per_class(
        text_merged,
        class_col="condition",
        cap=int(args.per_condition_cap),
        seed=int(args.seed),
    )
    text_merged.to_csv(out_text_csv, index=False)

    base_dialogue = _safe_read_csv(Path(args.base_dialogue_csv), ["user_text", "assistant_text", "intent"])
    if "source" not in base_dialogue.columns:
        base_dialogue["source"] = "base_existing"
    base_dialogue["source"] = base_dialogue["source"].fillna("base_existing")

    medmcqa_dialogue = build_medmcqa_dialogue(max_rows=int(args.medmcqa_max_rows), seed=int(args.seed))
    pubmedqa_dialogue = build_pubmedqa_dialogue(max_rows=int(args.pubmedqa_max_rows), seed=int(args.seed))

    dialogue_merged = pd.concat([base_dialogue, medmcqa_dialogue, pubmedqa_dialogue], ignore_index=True)
    dialogue_merged["user_text"] = dialogue_merged["user_text"].map(clean_text)
    dialogue_merged["assistant_text"] = dialogue_merged["assistant_text"].map(clean_text)
    dialogue_merged["intent"] = dialogue_merged["intent"].astype(str).str.strip().str.lower().replace("", "general")
    dialogue_merged["source"] = dialogue_merged["source"].astype(str).str.strip()
    dialogue_merged = dialogue_merged[
        (dialogue_merged["user_text"] != "") & (dialogue_merged["assistant_text"] != "")
    ]
    dialogue_merged = dialogue_merged.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)
    dialogue_merged.to_csv(out_dialogue_csv, index=False)

    summary = {
        "text_dataset": {
            "base_rows": int(len(base_text)),
            "ddxplus_rows": int(len(ddxplus_df)),
            "medqa_rows": int(len(medqa_df)),
            "merged_rows": int(len(text_merged)),
            "merged_classes": int(text_merged["condition"].nunique()),
            "output_csv": str(out_text_csv),
        },
        "dialogue_dataset": {
            "base_rows": int(len(base_dialogue)),
            "medmcqa_rows": int(len(medmcqa_dialogue)),
            "pubmedqa_rows": int(len(pubmedqa_dialogue)),
            "merged_rows": int(len(dialogue_merged)),
            "merged_intents": int(dialogue_merged["intent"].nunique()),
            "output_csv": str(out_dialogue_csv),
        },
        "licenses_note": {
            "ddxplus": "cc-by-4.0",
            "medmcqa": "apache-2.0",
            "pubmedqa": "mit",
            "medqa_usmle_hf": "cc-by-sa-4.0",
        },
    }
    summary_path = out_text_csv.with_name(f"{out_text_csv.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
