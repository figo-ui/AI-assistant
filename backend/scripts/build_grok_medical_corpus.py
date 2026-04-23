import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return WS_RE.sub(" ", text)


def _stream(ds_name: str, *args, split: str | None = None) -> Iterator[dict]:
    kwargs = {"streaming": True}
    if split is not None:
        kwargs["split"] = split
    return load_dataset(ds_name, *args, **kwargs)


def _take(records: Iterable[dict], limit: int) -> Iterator[dict]:
    return itertools.islice(records, limit) if limit > 0 else iter(records)


def _safe_load_stream(dataset_name: str, *configs, split: str | None = None) -> Iterator[dict]:
    last_exc = None
    for config in configs or (None,):
        try:
            if config is None:
                return _stream(dataset_name, split=split)
            return _stream(dataset_name, config, split=split)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise last_exc


def build_ddxplus(limit: int) -> pd.DataFrame:
    ds = _take(_safe_load_stream("aai530-group6/ddxplus", split="train"), limit)
    evid_path = hf_hub_download("aai530-group6/ddxplus", "release_evidences.json", repo_type="dataset")
    evid_map = json.loads(Path(evid_path).read_text(encoding="utf-8"))
    rows = []
    for item in ds:
        pathology = str(item.get("PATHOLOGY", "")).strip()
        if not pathology:
            continue
        tokens = [str(item.get("INITIAL_EVIDENCE", "")).strip()] + [str(v).strip() for v in item.get("EVIDENCES", []) or []]
        phrases = []
        seen = set()
        for token in tokens:
            base = token.split("_@_")[0]
            if not base or base in seen:
                continue
            seen.add(base)
            q = str(evid_map.get(base, {}).get("question_en", "")).strip()
            if q:
                phrases.append(q)
            if len(phrases) >= 14:
                break
        if not phrases:
            continue
        rows.append(
            {
                "symptom_text": norm(" ".join(phrases)),
                "condition": pathology,
                "urgency": "Medium",
                "reasoning": "Differential generated from symptom evidence profile.",
                "source_dataset": "ddxplus",
            }
        )
    return pd.DataFrame(rows)


def build_medqa(limit: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        try:
            ds = _take(_safe_load_stream("GBaker/MedQA-USMLE-4-options-hf", split=split), limit)
        except Exception:
            continue
        rows = []
        for item in ds:
            stem = norm(f"{item.get('sent1', '')} {item.get('sent2', '')}")
            if not stem or "diagnosis" not in stem and "most likely" not in stem:
                continue
            options = [str(item.get(f"ending{i}", "")).strip() for i in range(4)]
            try:
                idx = int(item.get("label", -1))
            except Exception:
                continue
            if 0 <= idx < len(options) and options[idx]:
                rows.append(
                    {
                        "symptom_text": stem,
                        "condition": options[idx],
                        "urgency": "Medium",
                        "reasoning": "USMLE diagnosis vignette mapped to most likely condition.",
                        "source_dataset": f"medqa_{split}",
                    }
                )
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_pubmedqa_dialogue(limit: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        try:
            ds = _take(_safe_load_stream("qiaojin/PubMedQA", "pqa_labeled", split=split), limit)
        except Exception:
            continue
        rows = []
        for item in ds:
            q = norm(item.get("question", ""))
            ans = str(item.get("long_answer", "")).strip()
            final_decision = str(item.get("final_decision", "")).strip()
            if q and ans:
                rows.append(
                    {
                        "user_text": q,
                        "assistant_text": norm(f"{ans} final decision: {final_decision}"),
                        "source_dataset": f"pubmedqa_{split}",
                        "intent": "evidence_qa",
                    }
                )
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_medmcqa_dialogue(limit: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        try:
            ds = _take(_safe_load_stream("openlifescienceai/medmcqa", split=split), limit)
        except Exception:
            continue
        rows = []
        for item in ds:
            q = norm(item.get("question", ""))
            opts = [str(item.get("opa", "")).strip(), str(item.get("opb", "")).strip(), str(item.get("opc", "")).strip(), str(item.get("opd", "")).strip()]
            try:
                idx = int(item.get("cop", -1))
            except Exception:
                continue
            idx = idx - 1 if idx in [1, 2, 3, 4] else idx
            if q and 0 <= idx < 4 and opts[idx]:
                rows.append(
                    {
                        "user_text": q,
                        "assistant_text": norm(opts[idx]),
                        "source_dataset": f"medmcqa_{split}",
                        "intent": "clinical_qa",
                    }
                )
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_bioasq_dialogue(limit: int) -> pd.DataFrame:
    dataset_options = [
        ("bigbio/bioasq", "bioasq_task_b"),
        ("bigbio/bioasq", "bioasq_all"),
    ]
    rows = []
    for dataset_name, config in dataset_options:
        try:
            ds = _take(_safe_load_stream(dataset_name, config, split="train"), limit)
        except Exception:
            continue
        for item in ds:
            question = norm(item.get("question", ""))
            answers = item.get("ideal_answer") or item.get("ideal_answers") or item.get("exact_answer")
            if isinstance(answers, list):
                answer = " ".join(str(v).strip() for v in answers if str(v).strip())
            else:
                answer = str(answers or "").strip()
            if question and answer:
                rows.append(
                    {
                        "user_text": question,
                        "assistant_text": norm(answer),
                        "source_dataset": f"{dataset_name}:{config}",
                        "intent": "biomedical_qa",
                    }
                )
        if rows:
            break
    return pd.DataFrame(rows)


def build_meddialog_dialogue(limit: int) -> pd.DataFrame:
    dataset_options = [
        ("bigbio/meddialog", "meddialog_en"),
        ("meddialog", "english"),
    ]
    rows = []
    for dataset_name, config in dataset_options:
        try:
            ds = _take(_safe_load_stream(dataset_name, config, split="train"), limit)
        except Exception:
            continue
        for item in ds:
            patient = norm(item.get("Patient", "") or item.get("patient", "") or item.get("utterance", ""))
            doctor = norm(item.get("Doctor", "") or item.get("doctor", "") or item.get("response", ""))
            if patient and doctor:
                rows.append(
                    {
                        "user_text": patient,
                        "assistant_text": doctor,
                        "source_dataset": f"{dataset_name}:{config}",
                        "intent": "doctor_patient_dialogue",
                    }
                )
        if rows:
            break
    return pd.DataFrame(rows)


def build_medical_o1(limit: int) -> pd.DataFrame:
    dataset_options = [
        ("FreedomIntelligence/medical-o1-reasoning-SFT", None),
    ]
    rows = []
    for dataset_name, config in dataset_options:
        try:
            ds = _take(_safe_load_stream(dataset_name, split="train"), limit)
        except Exception:
            continue
        for item in ds:
            question = norm(item.get("question", "") or item.get("input", "") or item.get("user", ""))
            answer = norm(item.get("response", "") or item.get("output", "") or item.get("assistant", ""))
            if question and answer:
                rows.append(
                    {
                        "user_text": question,
                        "assistant_text": answer,
                        "source_dataset": dataset_name,
                        "intent": "medical_reasoning",
                    }
                )
        if rows:
            break
    return pd.DataFrame(rows)


def cap_per_class(df: pd.DataFrame, column: str, cap: int) -> pd.DataFrame:
    if df.empty or cap <= 0:
        return df
    return df.groupby(column, group_keys=False).apply(lambda g: g.sample(n=min(len(g), cap), random_state=42)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build large multi-source medical corpus for classical and LLM triage training.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "grok_medical_corpus"))
    parser.add_argument("--ddxplus-limit", type=int, default=300000)
    parser.add_argument("--medqa-limit", type=int, default=20000)
    parser.add_argument("--medmcqa-limit", type=int, default=200000)
    parser.add_argument("--pubmedqa-limit", type=int, default=300000)
    parser.add_argument("--bioasq-limit", type=int, default=50000)
    parser.add_argument("--meddialog-limit", type=int, default=100000)
    parser.add_argument("--medical-o1-limit", type=int, default=100000)
    parser.add_argument("--per-condition-cap", type=int, default=1500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddxplus = build_ddxplus(int(args.ddxplus_limit))
    medqa = build_medqa(int(args.medqa_limit))
    text_df = pd.concat([ddxplus, medqa], ignore_index=True)
    text_df = text_df.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    text_df = cap_per_class(text_df, "condition", int(args.per_condition_cap))
    text_path = out_dir / "triage_supervised.csv"
    text_df.to_csv(text_path, index=False)

    dialogue_df = pd.concat(
        [
            build_medmcqa_dialogue(int(args.medmcqa_limit)),
            build_pubmedqa_dialogue(int(args.pubmedqa_limit)),
            build_bioasq_dialogue(int(args.bioasq_limit)),
            build_meddialog_dialogue(int(args.meddialog_limit)),
            build_medical_o1(int(args.medical_o1_limit)),
        ],
        ignore_index=True,
    )
    if not dialogue_df.empty:
        dialogue_df = dialogue_df.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)
    dialogue_path = out_dir / "triage_dialogue_reasoning.csv"
    dialogue_df.to_csv(dialogue_path, index=False)

    summary = {
        "triage_supervised_rows": int(len(text_df)),
        "triage_supervised_classes": int(text_df["condition"].nunique()) if not text_df.empty else 0,
        "dialogue_rows": int(len(dialogue_df)),
        "dialogue_intents": int(dialogue_df["intent"].nunique()) if not dialogue_df.empty else 0,
        "outputs": {
            "triage_supervised": str(text_path),
            "dialogue_reasoning": str(dialogue_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
