import argparse
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


WHITESPACE_RE = re.compile(r"\s+")
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


class XGBMulticlassWrapper:
    """
    Keeps a string-label interface while training XGBoost on numeric labels.
    Persisted via joblib and consumed at runtime by predict_proba().
    """

    def __init__(self, **params):
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Install it to use --model-type xgb.")
        self.model = XGBClassifier(**params)
        self.encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(np.asarray(y, dtype=str))
        self.classes_ = [str(v) for v in self.encoder.classes_]

        class_ids = np.unique(y_enc)
        class_weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_enc)
        weight_map = {int(k): float(v) for k, v in zip(class_ids, class_weights)}
        sample_weight = np.array([weight_map[int(v)] for v in y_enc], dtype=np.float32)

        self.model.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        pred_ids = self.model.predict(X)
        return self.encoder.inverse_transform(pred_ids.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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


def build_symptom_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_aliases = ["symptom_text", "text", "symptoms", "input_text", "description"]
    label_aliases = ["condition", "label", "disease", "Disease", "prognosis", "target"]

    text_col = next((col for col in text_aliases if col in df.columns), None)
    label_col = next((col for col in label_aliases if col in df.columns), None)

    if text_col and label_col:
        out = df[[text_col, label_col]].copy()
        out.columns = ["symptom_text", "condition"]
        out["condition"] = out["condition"].astype(str).str.strip()
        out["symptom_text"] = [
            strip_target_leakage(symptom_text=text, condition=condition)
            for text, condition in zip(out["symptom_text"], out["condition"])
        ]
        out = out.loc[(out["symptom_text"] != "") & (out["condition"] != "")]
        out = out.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
        return out

    if {"symptom_text", "condition"}.issubset(df.columns):
        out = df[["symptom_text", "condition"]].copy()
        out["condition"] = out["condition"].astype(str).str.strip()
        out["symptom_text"] = [
            strip_target_leakage(symptom_text=text, condition=condition)
            for text, condition in zip(out["symptom_text"], out["condition"])
        ]
        return out

    symptom_cols = [col for col in df.columns if col.lower().startswith("symptom_")]
    disease_col = None
    for candidate in ["Disease", "disease", "condition", "Condition", "prognosis", "label", "target"]:
        if candidate in df.columns:
            disease_col = candidate
            break

    if not symptom_cols or not disease_col:
        raise ValueError(
            "Dataset must include either text/label columns (e.g. text+label, symptom_text+condition) "
            "or a disease column plus Symptom_* columns."
        )

    work = df[symptom_cols + [disease_col]].copy()
    for col in symptom_cols:
        work[col] = work[col].fillna("").map(clean_text)

    work["symptom_text"] = (
        work[symptom_cols]
        .apply(lambda row: " ".join([v for v in row if v]), axis=1)
        .map(clean_text)
    )
    work["condition"] = work[disease_col].astype(str).str.strip()
    work["symptom_text"] = [
        strip_target_leakage(symptom_text=text, condition=condition)
        for text, condition in zip(work["symptom_text"], work["condition"])
    ]
    work = work.loc[work["symptom_text"] != ""]
    work = work.drop_duplicates(subset=["symptom_text", "condition"]).reset_index(drop=True)
    return work[["symptom_text", "condition"]]


def _distribution_report(y, out_path: Path, name: str) -> None:
    counts = pd.Series(y, dtype="string").value_counts()
    payload = {
        "name": name,
        "rows": int(len(y)),
        "classes": int(counts.shape[0]),
        "min_class_count": int(counts.min()),
        "median_class_count": float(counts.median()),
        "max_class_count": int(counts.max()),
        "max_min_ratio": round(float(counts.max() / max(1, counts.min())), 4),
        "top_20": {str(k): int(v) for k, v in counts.head(20).to_dict().items()},
        "bottom_20": {str(k): int(v) for k, v in counts.tail(20).to_dict().items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved class distribution report: {out_path}")


def _rebalance_under_over(
    X,
    y,
    *,
    majority_labels: list[str],
    majority_cap_fraction: float,
    minority_target: int,
    random_state: int,
):
    y_series = pd.Series(y, dtype="string")
    total_rows = len(y_series)
    cap_count = max(1, int(total_rows * majority_cap_fraction))

    under_strategy: dict[str, int] = {}
    for label in majority_labels:
        n_label = int((y_series == label).sum())
        if n_label > cap_count:
            under_strategy[label] = cap_count

    X_res, y_res = X, y_series
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    over_strategy = {str(cls): int(minority_target) for cls, n in counts.items() if int(n) < int(minority_target)}
    if over_strategy:
        ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X_res, y_res)

    return X_res, np.asarray(y_res, dtype=object)


def _rebalance_smote_svd(
    X_sparse,
    y,
    *,
    majority_labels: list[str],
    majority_cap_fraction: float,
    minority_target: int,
    random_state: int,
    svd_components: int,
):
    y_series = pd.Series(y, dtype="string")
    svd = TruncatedSVD(n_components=int(svd_components), random_state=random_state)
    X_dense = svd.fit_transform(X_sparse)

    total_rows = len(y_series)
    cap_count = max(1, int(total_rows * majority_cap_fraction))
    under_strategy: dict[str, int] = {}
    for label in majority_labels:
        n_label = int((y_series == label).sum())
        if n_label > cap_count:
            under_strategy[label] = cap_count

    X_res, y_res = X_dense, y_series
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    smote_strategy = {
        str(cls): int(minority_target)
        for cls, n in counts.items()
        if int(n) >= 6 and int(n) < int(minority_target)
    }
    if smote_strategy:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=3, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_res, y_res)

    return X_res, np.asarray(y_res, dtype=object), svd


def _rebalance_hybrid_smote_svd(
    X_sparse,
    y,
    *,
    majority_labels: list[str],
    majority_cap_fraction: float,
    minority_target: int,
    random_state: int,
    svd_components: int,
    smote_seed_floor: int = 6,
):
    y_series = pd.Series(y, dtype="string")
    svd = TruncatedSVD(n_components=int(svd_components), random_state=random_state)
    X_dense = svd.fit_transform(X_sparse)

    total_rows = len(y_series)
    cap_count = max(1, int(total_rows * majority_cap_fraction))
    under_strategy: dict[str, int] = {}
    for label in majority_labels:
        n_label = int((y_series == label).sum())
        if n_label > cap_count:
            under_strategy[label] = cap_count

    X_res, y_res = X_dense, y_series
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    seed_strategy = {
        str(cls): int(smote_seed_floor)
        for cls, n in counts.items()
        if int(n) < int(smote_seed_floor)
    }
    if seed_strategy:
        ros = RandomOverSampler(sampling_strategy=seed_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X_res, y_res)

    counts = Counter(y_res)
    smote_strategy = {
        str(cls): int(minority_target)
        for cls, n in counts.items()
        if int(n) >= int(smote_seed_floor) and int(n) < int(minority_target)
    }
    if smote_strategy:
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=max(1, int(smote_seed_floor) - 1),
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X_res, y_res)

    return X_res, np.asarray(y_res, dtype=object), svd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression text model.")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-type",
        choices=["lr", "xgb"],
        default="lr",
        help="Classifier backend (default: lr).",
    )
    parser.add_argument(
        "--rebalance-mode",
        choices=["none", "under_over", "smote_svd", "hybrid_smote_svd"],
        default="under_over",
        help="Dataset rebalancing strategy (default: under_over).",
    )
    parser.add_argument(
        "--majority-label",
        action="append",
        default=[],
        help="Majority class label to cap. Repeat flag for multiple labels.",
    )
    parser.add_argument(
        "--majority-cap-fraction",
        type=float,
        default=0.20,
        help="Maximum fraction allowed per majority label after undersampling.",
    )
    parser.add_argument(
        "--minority-target",
        type=int,
        default=200,
        help="Target minimum sample count per minority class after oversampling.",
    )
    parser.add_argument(
        "--smote-svd-components",
        type=int,
        default=384,
        help="SVD components for smote_svd mode.",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply calibrated probabilities (LR only).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    prepared = build_symptom_text_frame(df)

    x = prepared["symptom_text"].values
    y = prepared["condition"].values

    label_counts = prepared["condition"].value_counts()
    can_stratify = bool((label_counts >= 2).all())
    if not can_stratify:
        print("Warning: Some classes have <2 samples. Training without stratified split.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if can_stratify else None,
    )
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)

    x_fit = x_train
    y_fit = y_train
    x_cal = None
    y_cal = None
    if args.calibrate and args.model_type == "lr":
        label_counts_train = pd.Series(y_train).value_counts()
        can_stratify_train = bool((label_counts_train >= 2).all())
        x_fit, x_cal, y_fit, y_cal = train_test_split(
            x_train,
            y_train,
            test_size=0.15,
            random_state=args.random_state,
            stratify=y_train if can_stratify_train else None,
        )

    X_fit_vec = tfidf.fit_transform(x_fit)
    X_test_vec = tfidf.transform(x_test)
    X_cal_vec = tfidf.transform(x_cal) if x_cal is not None else None

    majority_labels = args.majority_label or ["COVID-19", "Suspected COVID-19"]
    svd = None
    if args.rebalance_mode == "under_over":
        X_train_reb, y_train_reb = _rebalance_under_over(
            X_fit_vec,
            y_fit,
            majority_labels=majority_labels,
            majority_cap_fraction=float(args.majority_cap_fraction),
            minority_target=int(args.minority_target),
            random_state=int(args.random_state),
        )
    elif args.rebalance_mode == "smote_svd":
        X_train_reb, y_train_reb, svd = _rebalance_smote_svd(
            X_fit_vec,
            y_fit,
            majority_labels=majority_labels,
            majority_cap_fraction=float(args.majority_cap_fraction),
            minority_target=int(args.minority_target),
            random_state=int(args.random_state),
            svd_components=int(args.smote_svd_components),
        )
    elif args.rebalance_mode == "hybrid_smote_svd":
        X_train_reb, y_train_reb, svd = _rebalance_hybrid_smote_svd(
            X_fit_vec,
            y_fit,
            majority_labels=majority_labels,
            majority_cap_fraction=float(args.majority_cap_fraction),
            minority_target=int(args.minority_target),
            random_state=int(args.random_state),
            svd_components=int(args.smote_svd_components),
        )
    else:
        X_train_reb, y_train_reb = X_fit_vec, np.asarray(y_fit, dtype=object)

    _distribution_report(
        y_fit,
        out_dir / "class_distribution_train_raw.json",
        name="train_raw",
    )
    _distribution_report(
        y_train_reb,
        out_dir / "class_distribution_train_rebalanced.json",
        name=f"train_rebalanced_{args.rebalance_mode}",
    )

    X_cal_model = svd.transform(X_cal_vec) if (svd is not None and X_cal_vec is not None) else X_cal_vec

    if args.model_type == "lr":
        base_clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
        base_clf.fit(X_train_reb, y_train_reb)
        clf = base_clf
        if args.calibrate and X_cal_vec is not None and y_cal is not None:
            try:
                from sklearn.frozen import FrozenEstimator  # sklearn>=1.6

                clf = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_clf),
                    method="sigmoid",
                    cv="prefit",
                )
            except ImportError:
                clf = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv="prefit")
            clf.fit(X_cal_model, y_cal)
    else:
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed. Run: pip install xgboost")
        clf = XGBMulticlassWrapper(
            objective="multi:softprob",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=int(args.random_state),
            n_jobs=-1,
        )
        clf.fit(X_train_reb, y_train_reb)

    X_test_model = svd.transform(X_test_vec) if svd is not None else X_test_vec
    y_pred = clf.predict(X_test_model)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation macro-F1: {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    model_path = out_dir / "text_classifier.joblib"
    vectorizer_path = out_dir / "tfidf_vectorizer.joblib"
    labels_path = out_dir / "text_labels.json"
    metrics_path = out_dir / "text_training_metrics.json"
    svd_path = out_dir / "tfidf_svd.joblib"

    joblib.dump(clf, model_path)
    joblib.dump(tfidf, vectorizer_path)
    labels = [str(v) for v in getattr(clf, "classes_", [])]
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    if svd is not None:
        joblib.dump(svd, svd_path)
        print(f"Saved SVD transformer: {svd_path}")
    elif svd_path.exists():
        svd_path.unlink(missing_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1_macro), 4),
                "samples": int(len(prepared)),
                "classes": int(len(labels)),
                "model_type": str(args.model_type),
                "rebalance_mode": str(args.rebalance_mode),
                "majority_labels": majority_labels,
                "majority_cap_fraction": round(float(args.majority_cap_fraction), 4),
                "minority_target": int(args.minority_target),
                "calibrated": bool(args.calibrate and args.model_type == "lr"),
                "uses_svd": bool(svd is not None),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved vectorizer: {vectorizer_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
