 import argparse
import json
import re
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    user_candidates = ["user_text", "question", "input_text", "symptom_text", "text"]
    assistant_candidates = ["assistant_text", "answer", "response", "reply", "output_text"]
    intent_candidates = ["intent", "qtype", "label", "topic"]

    user_col = next((c for c in user_candidates if c in df.columns), None)
    assistant_col = next((c for c in assistant_candidates if c in df.columns), None)
    intent_col = next((c for c in intent_candidates if c in df.columns), None)

    if not user_col or not assistant_col:
        raise ValueError(
            "Dataset must include user/assistant text columns. "
            f"Supported user columns: {user_candidates}; assistant columns: {assistant_candidates}."
        )
    return {"user": user_col, "assistant": assistant_col, "intent": intent_col or ""}


def template_for_intent(intent: str) -> str:
    text = intent.lower()
    if "symptom" in text or "sign" in text:
        return "Thank you for sharing your symptoms clearly."
    if "treatment" in text or "therapy" in text:
        return "I understand your concern, and I will walk you through practical care options."
    if "diagn" in text or "test" in text or "exam" in text:
        return "That is an important question, and we can review it step by step."
    if "risk" in text or "suscept" in text or "cause" in text:
        return "Your concern is valid, and it helps to review risk factors carefully."
    if "prevent" in text:
        return "Prevention is a strong step, and we can focus on actions you can take today."
    if "outlook" in text or "prognosis" in text:
        return "I know this can feel stressful, and I will keep the guidance clear and direct."
    return "Thanks for sharing the details; I will provide clear guidance based on your input."


def main() -> None:
    parser = argparse.ArgumentParser(description="Train medical dialogue intent model for friendlier responses.")
    parser.add_argument("--dataset", required=True, help="Path to dialogue CSV.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dialogue dataset not found: {dataset_path}")

    raw = pd.read_csv(dataset_path)
    cols = detect_columns(raw)

    work = raw[[cols["user"], cols["assistant"]] + ([cols["intent"]] if cols["intent"] else [])].copy()
    rename_map = {
        cols["user"]: "user_text",
        cols["assistant"]: "assistant_text",
    }
    if cols["intent"]:
        rename_map[cols["intent"]] = "intent"
    work = work.rename(columns=rename_map)

    work["user_text"] = work["user_text"].map(clean_text)
    work["assistant_text"] = work["assistant_text"].astype(str).str.strip()
    if "intent" not in work.columns:
        work["intent"] = "general"
    work["intent"] = work["intent"].astype(str).str.strip().str.lower().replace("", "general")

    work = work[(work["user_text"] != "") & (work["assistant_text"] != "")]
    work = work.drop_duplicates(subset=["user_text", "assistant_text"]).reset_index(drop=True)

    counts = work["intent"].value_counts()
    keep_intents = counts[counts >= 2].index
    work = work[work["intent"].isin(keep_intents)].reset_index(drop=True)
    if len(work) < 100:
        raise ValueError("Dialogue dataset too small after filtering. Need at least 100 rows.")

    x = work["user_text"].values
    y = work["intent"].values

    can_stratify = bool((work["intent"].value_counts() >= 2).all())
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=-1)),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "samples": int(len(work)),
        "intent_classes": int(work["intent"].nunique()),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }

    print("Dialogue intent model metrics:")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, digits=4))

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    model_path = out_dir / "dialogue_intent_classifier.joblib"
    vectorizer_path = out_dir / "dialogue_intent_vectorizer.joblib"
    labels_path = out_dir / "dialogue_intent_labels.json"
    templates_path = out_dir / "dialogue_response_templates.json"
    metrics_path = out_dir / "dialogue_training_metrics.json"

    joblib.dump(clf, model_path)
    joblib.dump(tfidf, vectorizer_path)
    labels = [str(v) for v in clf.classes_]
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    templates = {label: template_for_intent(label) for label in labels}
    templates_path.write_text(json.dumps(templates, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved vectorizer: {vectorizer_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved templates: {templates_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
