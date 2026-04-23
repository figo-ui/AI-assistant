"""
retrain_all_models.py
=====================
One-shot retraining script for all three models using the best available datasets.

Usage (from project root):
    python backend/scripts/retrain_all_models.py

Options:
    --text-only       Retrain text model only
    --dialogue-only   Retrain dialogue model only
    --skip-text       Skip text model
    --skip-dialogue   Skip dialogue model
    --min-samples N   Minimum samples per class (default: 5)
    --model-dir PATH  Output directory for model artifacts (default: backend/models)
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
MODELS_DIR = ROOT / "backend" / "models"

# Priority-ordered dataset paths for text triage
TEXT_DATASET_CANDIDATES = [
    DATA / "unified" / "ULTIMATE_TRIAGE_KNOWLEDGE.csv",
    DATA / "processed" / "expanded_symptom_condition_processed_min5_rebalanced.csv",
    DATA / "raw" / "kaggle" / "processed" / "integrated_important_plus_kaggle_processed_min5.csv",
    DATA / "processed" / "integrated_important_symptom_condition_processed_min5.csv",
]

# Priority-ordered dataset paths for dialogue
DIALOGUE_DATASET_CANDIDATES = [
    DATA / "unified" / "ULTIMATE_CONVERSATIONAL_QA.csv",
    DATA / "raw" / "dialogue" / "medquad_clinical_qa.csv",
    DATA / "raw" / "dialogue_legacy" / "expanded_medical_dialogue.csv",
]

GENERIC_RE_PATTERNS = ["condition \\d+", "class_\\d+"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_best_dataset(candidates: list[Path], label: str) -> pd.DataFrame:
    for path in candidates:
        if path.exists():
            log.info("[%s] Loading dataset: %s", label, path)
            df = pd.read_csv(path)
            log.info("[%s] Loaded %d rows, columns: %s", label, len(df), list(df.columns))
            return df
    raise FileNotFoundError(f"[{label}] No dataset found. Checked:\n" + "\n".join(str(p) for p in candidates))


def _detect_columns(df: pd.DataFrame, label: str) -> tuple[str, str]:
    """Auto-detect text and label columns."""
    text_candidates = ["symptom_text", "symptoms", "text", "input", "question", "utterance", "sentence"]
    label_candidates = ["condition", "label", "diagnosis", "intent", "category", "output", "answer"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if not text_col or not label_col:
        raise ValueError(
            f"[{label}] Cannot detect text/label columns. Found: {list(df.columns)}\n"
            f"Expected text col one of: {text_candidates}\n"
            f"Expected label col one of: {label_candidates}"
        )
    log.info("[%s] Using text_col='%s', label_col='%s'", label, text_col, label_col)
    return text_col, label_col


def _clean_df(df: pd.DataFrame, text_col: str, label_col: str, min_samples: int, label: str) -> pd.DataFrame:
    import re
    df = df[[text_col, label_col]].copy()
    df.dropna(inplace=True)
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Remove generic/placeholder labels
    for pattern in GENERIC_RE_PATTERNS:
        mask = df[label_col].str.match(pattern, case=False, na=False)
        removed = mask.sum()
        if removed:
            log.info("[%s] Removed %d generic label rows matching '%s'", label, removed, pattern)
        df = df[~mask]

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(subset=[text_col, label_col], inplace=True)
    log.info("[%s] Deduplication: %d → %d rows", label, before, len(df))

    # Filter rare classes
    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= min_samples].index
    before = len(df)
    df = df[df[label_col].isin(valid_classes)]
    log.info(
        "[%s] Class filter (min=%d): %d → %d rows, %d classes",
        label, min_samples, before, len(df), df[label_col].nunique()
    )
    return df.reset_index(drop=True)


def _rebalance(X: np.ndarray, y: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Under-sample majority, over-sample minority."""
    counts = pd.Series(y).value_counts()
    majority_cap = int(counts.quantile(0.85))
    minority_target = max(50, int(counts.median()))

    log.info("[%s] Rebalancing: majority_cap=%d, minority_target=%d", label, majority_cap, minority_target)

    sampling_strategy_under = {
        cls: min(count, majority_cap)
        for cls, count in counts.items()
        if count > majority_cap
    }
    if sampling_strategy_under:
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
        X, y = rus.fit_resample(X, y)

    counts_after = pd.Series(y).value_counts()
    sampling_strategy_over = {
        cls: minority_target
        for cls, count in counts_after.items()
        if count < minority_target
    }
    if sampling_strategy_over:
        ros = RandomOverSampler(sampling_strategy=sampling_strategy_over, random_state=42)
        X, y = ros.fit_resample(X, y)

    log.info("[%s] After rebalance: %d samples, %d classes", label, len(y), len(set(y)))
    return X, y


def _evaluate(clf, X_train, X_val, X_test, y_train, y_val, y_test, name: str, label: str) -> dict:
    train_f1 = f1_score(y_train, clf.predict(X_train), average="macro", zero_division=0)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro", zero_division=0)
    test_f1 = f1_score(y_test, clf.predict(X_test), average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    log.info("[%s] %s → train_f1=%.4f  val_f1=%.4f  test_f1=%.4f  val_acc=%.4f",
             label, name, train_f1, val_f1, test_f1, val_acc)
    return {"name": name, "train_f1": train_f1, "val_f1": val_f1, "test_f1": test_f1, "val_acc": val_acc}


# ─── Text Model ───────────────────────────────────────────────────────────────

def train_text_model(min_samples: int, model_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("STEP 1: TEXT TRIAGE MODEL")
    log.info("=" * 60)
    t0 = time.monotonic()

    df = _load_best_dataset(TEXT_DATASET_CANDIDATES, "TEXT")
    text_col, label_col = _detect_columns(df, "TEXT")
    df = _clean_df(df, text_col, label_col, min_samples, "TEXT")

    X_raw = df[text_col].values
    y_raw = df[label_col].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Split
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    log.info("TEXT split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    # TF-IDF (word + char n-grams)
    log.info("TEXT: Fitting TF-IDF vectorizer (word + char)...")
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vec.fit_transform(X_train_raw)
    X_val_tfidf = vec.transform(X_val_raw)
    X_test_tfidf = vec.transform(X_test_raw)

    # Rebalance
    X_train_bal, y_train_bal = _rebalance(X_train_tfidf, y_train, "TEXT")

    # Train candidates
    results = []
    best_clf = None
    best_val_f1 = -1.0

    candidates = [
        ("LR", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("LinearSVC+Cal", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42), cv=3)),
        ("SGD", SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]

    for name, clf in candidates:
        log.info("TEXT: Training %s...", name)
        clf.fit(X_train_bal, y_train_bal)
        r = _evaluate(clf, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, name, "TEXT")
        results.append(r)
        if r["val_f1"] > best_val_f1:
            best_val_f1 = r["val_f1"]
            best_clf = clf

    best_result = max(results, key=lambda x: x["val_f1"])
    log.info("TEXT: Best model: %s (val_f1=%.4f)", best_result["name"], best_result["val_f1"])

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, model_dir / "text_classifier.joblib")
    joblib.dump(vec, model_dir / "tfidf_vectorizer.joblib")

    labels_map = {int(i): str(label) for i, label in enumerate(le.classes_)}
    (model_dir / "text_labels.json").write_text(json.dumps(labels_map, indent=2))

    metrics = {
        "model_name": best_result["name"],
        "train_macro_f1": round(best_result["train_f1"], 4),
        "val_macro_f1": round(best_result["val_f1"], 4),
        "test_macro_f1": round(best_result["test_f1"], 4),
        "val_accuracy": round(best_result["val_acc"], 4),
        "classes": int(len(le.classes_)),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "vectorizer_mode": "word_ngram",
        "all_models": [
            {"name": r["name"], "val_f1": round(r["val_f1"], 4), "test_f1": round(r["test_f1"], 4), "train_f1": round(r["train_f1"], 4)}
            for r in results
        ],
        "training_time_seconds": round(time.monotonic() - t0, 1),
    }
    (model_dir / "text_training_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("TEXT: Artifacts saved to %s", model_dir)
    return metrics


# ─── Dialogue Model ───────────────────────────────────────────────────────────

def train_dialogue_model(min_samples: int, model_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("STEP 2: DIALOGUE INTENT MODEL")
    log.info("=" * 60)
    t0 = time.monotonic()

    df = _load_best_dataset(DIALOGUE_DATASET_CANDIDATES, "DIALOGUE")
    text_col, label_col = _detect_columns(df, "DIALOGUE")
    df = _clean_df(df, text_col, label_col, min_samples, "DIALOGUE")

    X_raw = df[text_col].values
    y_raw = df[label_col].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    log.info("DIALOGUE split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=60_000,
        sublinear_tf=True,
        min_df=1,
    )
    X_train_tfidf = vec.fit_transform(X_train_raw)
    X_val_tfidf = vec.transform(X_val_raw)
    X_test_tfidf = vec.transform(X_test_raw)

    results = []
    best_clf = None
    best_val_f1 = -1.0

    candidates = [
        ("LinearSVC C=1.0", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3)),
        ("SGD modified_huber", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=300, random_state=42, n_jobs=-1)),
        ("LR", LogisticRegression(C=10.0, max_iter=1000, random_state=42, n_jobs=-1)),
    ]

    for name, clf in candidates:
        log.info("DIALOGUE: Training %s...", name)
        clf.fit(X_train_tfidf, y_train)
        r = _evaluate(clf, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, name, "DIALOGUE")
        results.append(r)
        if r["val_f1"] > best_val_f1:
            best_val_f1 = r["val_f1"]
            best_clf = clf

    best_result = max(results, key=lambda x: x["val_f1"])
    log.info("DIALOGUE: Best model: %s (val_f1=%.4f)", best_result["name"], best_result["val_f1"])

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, model_dir / "dialogue_intent_classifier.joblib")
    joblib.dump(vec, model_dir / "dialogue_intent_vectorizer.joblib")

    labels_map = {int(i): str(label) for i, label in enumerate(le.classes_)}
    (model_dir / "dialogue_intent_labels.json").write_text(json.dumps(labels_map, indent=2))

    metrics = {
        "model_name": best_result["name"],
        "train_macro_f1": round(best_result["train_f1"], 4),
        "val_macro_f1": round(best_result["val_f1"], 4),
        "test_macro_f1": round(best_result["test_f1"], 4),
        "val_accuracy": round(best_result["val_acc"], 4),
        "intent_classes": int(len(le.classes_)),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "all_models": [
            {"name": r["name"], "val_f1": round(r["val_f1"], 4), "test_f1": round(r["test_f1"], 4), "val_acc": round(r["val_acc"], 4)}
            for r in results
        ],
        "training_time_seconds": round(time.monotonic() - t0, 1),
    }
    (model_dir / "dialogue_training_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("DIALOGUE: Artifacts saved to %s", model_dir)
    return metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain all AI Healthcare Assistant models")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--dialogue-only", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-dialogue", action="store_true")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    summary = {}
    total_start = time.monotonic()

    run_text = not args.skip_text and not args.dialogue_only
    run_dialogue = not args.skip_dialogue and not args.text_only

    if run_text:
        try:
            summary["text"] = train_text_model(args.min_samples, args.model_dir)
        except Exception as e:
            log.error("TEXT model training failed: %s", e)
            summary["text"] = {"error": str(e)}

    if run_dialogue:
        try:
            summary["dialogue"] = train_dialogue_model(args.min_samples, args.model_dir)
        except Exception as e:
            log.error("DIALOGUE model training failed: %s", e)
            summary["dialogue"] = {"error": str(e)}

    total_time = round(time.monotonic() - total_start, 1)
    log.info("=" * 60)
    log.info("TRAINING COMPLETE in %.1fs", total_time)
    for model, result in summary.items():
        if "error" in result:
            log.error("  %s: FAILED — %s", model.upper(), result["error"])
        else:
            val_f1 = result.get("val_macro_f1", result.get("val_f1", "?"))
            test_f1 = result.get("test_macro_f1", result.get("test_f1", "?"))
            log.info("  %s: val_f1=%.4f  test_f1=%.4f", model.upper(), val_f1, test_f1)
    log.info("=" * 60)
    log.info("NOTE: Image model requires GPU. Run on Kaggle:")
    log.info("  backend/scripts/train_image_model.py --manifest data/image_dataset_combined/manifest.jsonl")
    log.info("  Or use: backend/scripts/fitzpatrick_train_fast.py for Fitzpatrick17k")


if __name__ == "__main__":
    main()
