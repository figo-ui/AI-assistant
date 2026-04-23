import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from train_text_model import build_symptom_text_frame, clean_text, strip_target_leakage


def parse_args() -> argparse.Namespace:
    backend_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate the text triage model on a held-out dataset.")
    parser.add_argument("--dataset", required=True, help="Evaluation CSV containing symptom_text and condition.")
    parser.add_argument(
        "--model-dir",
        default=str(backend_dir / "models"),
        help="Directory containing text model artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(backend_dir / "models" / "evaluation"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=200,
        help="Fail if fewer than this many matched evaluation rows remain.",
    )
    parser.add_argument(
        "--max-confusion-classes",
        type=int,
        default=25,
        help="Maximum number of classes to include in confusion matrix output.",
    )
    parser.add_argument(
        "--review-sample-size",
        type=int,
        default=250,
        help="How many cases to emit in the clinician review placeholder CSV.",
    )
    return parser.parse_args()


def _load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "text_classifier.joblib")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    labels_path = model_dir / "text_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else []
    svd_path = model_dir / "tfidf_svd.joblib"
    svd = joblib.load(svd_path) if svd_path.exists() else None
    return model, vectorizer, labels, svd


def _prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    if {"symptom_text", "condition"}.issubset(df.columns):
        prepared = df[["symptom_text", "condition"]].copy()
        prepared["condition"] = prepared["condition"].astype(str).str.strip()
        prepared["symptom_text"] = [
            strip_target_leakage(symptom_text=text, condition=condition)
            for text, condition in zip(prepared["symptom_text"], prepared["condition"])
        ]
        prepared["symptom_text"] = prepared["symptom_text"].map(clean_text)
        return prepared.loc[(prepared["symptom_text"] != "") & (prepared["condition"] != "")].reset_index(drop=True)
    return build_symptom_text_frame(df)


def _expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    results: List[Dict[str, float]] = []
    ece = 0.0
    total = max(1, len(confidences))
    for start, end in zip(edges[:-1], edges[1:]):
        if end >= 1.0:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences >= start) & (confidences < end)
        count = int(mask.sum())
        if count == 0:
            results.append(
                {
                    "bin_start": round(float(start), 2),
                    "bin_end": round(float(end), 2),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correctness[mask].mean())
        ece += (count / total) * abs(mean_confidence - accuracy)
        results.append(
            {
                "bin_start": round(float(start), 2),
                "bin_end": round(float(end), 2),
                "count": count,
                "mean_confidence": round(mean_confidence, 4),
                "accuracy": round(accuracy, 4),
            }
        )
    return round(float(ece), 4), results


def _multiclass_brier_score(y_true_idx: np.ndarray, probs: np.ndarray, labels: List[str]) -> float:
    target = np.zeros((len(y_true_idx), len(labels)), dtype=np.float64)
    target[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    prepared = _prepare_eval_frame(df)
    model, vectorizer, labels, svd = _load_artifacts(model_dir)
    model_labels = labels or [str(item) for item in getattr(model, "classes_", [])]
    if not model_labels:
        raise RuntimeError("Model labels could not be loaded from artifacts.")

    prepared = prepared[prepared["condition"].isin(model_labels)].reset_index(drop=True)
    if len(prepared) < int(args.min_cases):
        raise ValueError(
            f"Evaluation set only has {len(prepared)} rows overlapping model labels; need at least {args.min_cases}."
        )

    matrix = vectorizer.transform(prepared["symptom_text"].tolist())
    if svd is not None:
        matrix = svd.transform(matrix)

    probs = model.predict_proba(matrix)
    class_names = list(getattr(model, "classes_", model_labels))
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [str(class_names[idx]) for idx in pred_idx]
    top1_conf = probs[np.arange(len(pred_idx)), pred_idx]

    y_true = prepared["condition"].astype(str).tolist()
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_true_idx = np.asarray([class_to_idx[label] for label in y_true], dtype=np.int64)

    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3_correct = np.asarray(
        [y_true_idx[row_idx] in set(top3_idx[row_idx]) for row_idx in range(len(y_true_idx))],
        dtype=np.float64,
    )
    correctness = np.asarray([int(a == b) for a, b in zip(y_true, pred_labels)], dtype=np.float64)
    ece, calibration_bins = _expected_calibration_error(top1_conf, correctness, bins=10)

    top_labels = prepared["condition"].value_counts().head(int(args.max_confusion_classes)).index.tolist()
    confusion = confusion_matrix(
        [label for label in y_true if label in top_labels],
        [label for label, pred in zip(y_true, pred_labels) if label in top_labels for pred in [pred]],
        labels=top_labels,
    )
    confusion_df = pd.DataFrame(confusion, index=top_labels, columns=top_labels)
    confusion_path = out_dir / "text_confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    summary = {
        "dataset": str(dataset_path),
        "rows_evaluated": int(len(prepared)),
        "classes_evaluated": int(prepared["condition"].nunique()),
        "accuracy_top1": round(float(accuracy_score(y_true, pred_labels)), 4),
        "macro_f1_top1": round(float(f1_score(y_true, pred_labels, average="macro")), 4),
        "top3_accuracy": round(float(top3_correct.mean()), 4),
        "negative_log_likelihood": round(float(log_loss(y_true_idx, probs, labels=list(range(len(class_names))))), 4),
        "brier_score": round(_multiclass_brier_score(y_true_idx, probs, class_names), 4),
        "expected_calibration_error": ece,
        "confusion_matrix_csv": str(confusion_path),
        "clinician_review_placeholder_csv": str(out_dir / "clinician_review_placeholder.csv"),
        "notes": [
            "Evaluation rows were filtered to labels present in the trained model artifacts.",
            "This is a held-out external-style evaluation only if the dataset source differs from training.",
            "Clinician review CSV is a placeholder workflow artifact and not completed validation.",
        ],
    }
    (out_dir / "text_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "text_calibration_bins.json").write_text(json.dumps(calibration_bins, indent=2), encoding="utf-8")

    review_frame = prepared.copy()
    review_frame["predicted_condition"] = pred_labels
    review_frame["top1_confidence"] = np.round(top1_conf, 4)
    review_frame["review_status"] = "pending"
    review_frame["clinician_notes"] = ""
    review_frame["accepted_prediction"] = ""
    review_frame.head(int(args.review_sample_size)).to_csv(
        out_dir / "clinician_review_placeholder.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
