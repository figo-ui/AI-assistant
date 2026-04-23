"""
STEP 3: Model Training — Task 1: Triage Text Classification (v2 — fast)
Strategy:
  - Model A: TF-IDF (word, 1-2gram) + SGDClassifier (modified_huber) — fast, calibrated
  - Model B: TF-IDF (word+char) + SGDClassifier — richer features
  - Model C: TF-IDF (word) + LinearSVC (no calibration) — fastest, best F1 for many classes
  All trained on 60,553 rows / 115 classes
  Target: beat Macro F1 = 0.7709
"""
import sys, os, warnings, json, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, "data", "dataset_v1.0", "triage")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
section("Loading Data")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

with open(os.path.join(DATA_DIR, "label_names.json")) as f:
    label_names = json.load(f)

X_train = df_train["symptom_text"].astype(str).values
y_train = df_train["label_id"].values
X_val   = df_val["symptom_text"].astype(str).values
y_val   = df_val["label_id"].values
X_test  = df_test["symptom_text"].astype(str).values
y_test  = df_test["label_id"].values

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(f"Classes: {len(label_names)}")

# ── Vectorize ──────────────────────────────────────────────────────────────
section("Vectorizing (TF-IDF word 1-2gram)")
t0 = time.time()
vec_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=80000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    strip_accents="unicode",
)
X_tr_w = vec_word.fit_transform(X_train)
X_va_w = vec_word.transform(X_val)
X_te_w = vec_word.transform(X_test)
print(f"Word TF-IDF shape: {X_tr_w.shape}, time: {time.time()-t0:.1f}s")

# Char TF-IDF
t0 = time.time()
vec_char = TfidfVectorizer(
    ngram_range=(3, 4),
    max_features=30000,
    sublinear_tf=True,
    min_df=5,
    analyzer="char_wb",
    strip_accents="unicode",
)
X_tr_c = vec_char.fit_transform(X_train)
X_va_c = vec_char.transform(X_val)
X_te_c = vec_char.transform(X_test)
print(f"Char TF-IDF shape: {X_tr_c.shape}, time: {time.time()-t0:.1f}s")

# Combined
X_tr = hstack([X_tr_w, X_tr_c])
X_va = hstack([X_va_w, X_va_c])
X_te = hstack([X_te_w, X_te_c])
print(f"Combined shape: {X_tr.shape}")

# ── Model A: SGD (modified_huber) — word only ──────────────────────────────
section("Model A: SGD (modified_huber) — word TF-IDF")
t0 = time.time()
sgd_a = SGDClassifier(
    loss="modified_huber",
    alpha=5e-5,
    max_iter=300,
    tol=1e-4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
sgd_a.fit(X_tr_w, y_train)
t1 = time.time()

y_tr_a = sgd_a.predict(X_tr_w)
y_va_a = sgd_a.predict(X_va_w)
y_te_a = sgd_a.predict(X_te_w)

tr_f1_a = f1_score(y_train, y_tr_a, average="macro")
va_f1_a = f1_score(y_val, y_va_a, average="macro")
te_f1_a = f1_score(y_test, y_te_a, average="macro")
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_f1_a:.4f} | Val F1: {va_f1_a:.4f} | Test F1: {te_f1_a:.4f} | Gap: {tr_f1_a-va_f1_a:.4f}")

# ── Model B: SGD (modified_huber) — word+char ──────────────────────────────
section("Model B: SGD (modified_huber) — word+char TF-IDF")
t0 = time.time()
sgd_b = SGDClassifier(
    loss="modified_huber",
    alpha=5e-5,
    max_iter=300,
    tol=1e-4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
sgd_b.fit(X_tr, y_train)
t1 = time.time()

y_tr_b = sgd_b.predict(X_tr)
y_va_b = sgd_b.predict(X_va)
y_te_b = sgd_b.predict(X_te)

tr_f1_b = f1_score(y_train, y_tr_b, average="macro")
va_f1_b = f1_score(y_val, y_va_b, average="macro")
te_f1_b = f1_score(y_test, y_te_b, average="macro")
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_f1_b:.4f} | Val F1: {va_f1_b:.4f} | Test F1: {te_f1_b:.4f} | Gap: {tr_f1_b-va_f1_b:.4f}")

# ── Model C: LinearSVC — word+char (no calibration, fastest) ──────────────
section("Model C: LinearSVC — word+char TF-IDF")
t0 = time.time()
svc_c = LinearSVC(
    C=0.3,
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
svc_c.fit(X_tr, y_train)
t1 = time.time()

y_tr_c = svc_c.predict(X_tr)
y_va_c = svc_c.predict(X_va)
y_te_c = svc_c.predict(X_te)

tr_f1_c = f1_score(y_train, y_tr_c, average="macro")
va_f1_c = f1_score(y_val, y_va_c, average="macro")
te_f1_c = f1_score(y_test, y_te_c, average="macro")
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_f1_c:.4f} | Val F1: {va_f1_c:.4f} | Test F1: {te_f1_c:.4f} | Gap: {tr_f1_c-va_f1_c:.4f}")

# ── Model D: LinearSVC C=1.0 ───────────────────────────────────────────────
section("Model D: LinearSVC C=1.0 — word+char TF-IDF")
t0 = time.time()
svc_d = LinearSVC(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
svc_d.fit(X_tr, y_train)
t1 = time.time()

y_tr_d = svc_d.predict(X_tr)
y_va_d = svc_d.predict(X_va)
y_te_d = svc_d.predict(X_te)

tr_f1_d = f1_score(y_train, y_tr_d, average="macro")
va_f1_d = f1_score(y_val, y_va_d, average="macro")
te_f1_d = f1_score(y_test, y_te_d, average="macro")
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_f1_d:.4f} | Val F1: {va_f1_d:.4f} | Test F1: {te_f1_d:.4f} | Gap: {tr_f1_d-va_f1_d:.4f}")

# ── Select best ────────────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("SGD word-only",       va_f1_a, te_f1_a, tr_f1_a, sgd_a, "word"),
    ("SGD word+char",       va_f1_b, te_f1_b, tr_f1_b, sgd_b, "combined"),
    ("LinearSVC C=0.3",     va_f1_c, te_f1_c, tr_f1_c, svc_c, "combined"),
    ("LinearSVC C=1.0",     va_f1_d, te_f1_d, tr_f1_d, svc_d, "combined"),
]

print(f"\n{'Model':30s} | Val F1  | Test F1 | Train F1 | Gap")
print("-" * 70)
for name, vf1, tf1, trf1, _, mode in results:
    print(f"  {name:28s} | {vf1:.4f}  | {tf1:.4f}  | {trf1:.4f}   | {trf1-vf1:.4f}")

best_idx = max(range(len(results)), key=lambda i: results[i][1])
best_name, best_val_f1, best_test_f1, best_train_f1, best_model, best_mode = results[best_idx]
print(f"\nBest model: {best_name}")
print(f"  Val Macro F1:  {best_val_f1:.4f}")
print(f"  Test Macro F1: {best_test_f1:.4f}")
print(f"  Baseline:      0.7709")
print(f"  Improvement:   {best_val_f1 - 0.7709:+.4f}")

# ── Save best model ────────────────────────────────────────────────────────
section("Saving Best Triage Model")

joblib.dump(best_model, os.path.join(MODEL_DIR, "text_classifier.joblib"))
joblib.dump(vec_word, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(vec_char, os.path.join(MODEL_DIR, "tfidf_char_vectorizer.joblib"))

with open(os.path.join(MODEL_DIR, "text_labels.json"), "w") as f:
    json.dump(label_names, f, indent=2)

# Determine best predictions for metrics
if best_mode == "combined":
    y_va_best = best_model.predict(X_va)
    y_te_best = best_model.predict(X_te)
else:
    y_va_best = best_model.predict(X_va_w)
    y_te_best = best_model.predict(X_te_w)

metrics = {
    "model_name": best_name,
    "train_macro_f1": round(float(best_train_f1), 4),
    "val_macro_f1": round(float(best_val_f1), 4),
    "test_macro_f1": round(float(best_test_f1), 4),
    "val_accuracy": round(float(accuracy_score(y_val, y_va_best)), 4),
    "test_accuracy": round(float(accuracy_score(y_test, y_te_best)), 4),
    "classes": len(label_names),
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "baseline_macro_f1": 0.7709,
    "improvement": round(float(best_val_f1) - 0.7709, 4),
    "vectorizer_mode": best_mode,
    "all_models": [
        {"name": r[0], "val_f1": round(r[1], 4), "test_f1": round(r[2], 4), "train_f1": round(r[3], 4)}
        for r in results
    ]
}
with open(os.path.join(MODEL_DIR, "text_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved: text_classifier.joblib, tfidf_vectorizer.joblib, tfidf_char_vectorizer.joblib")
print(f"Saved: text_labels.json ({len(label_names)} classes)")
print(f"Saved: text_training_metrics.json")

# Per-class report
section("Per-Class Report (Best Model — Val Set)")
report = classification_report(y_val, y_va_best, target_names=label_names, output_dict=True)
per_class = [(label_names[i], report.get(label_names[i], {}).get("f1-score", 0.0))
             for i in range(len(label_names)) if label_names[i] in report]
per_class.sort(key=lambda x: x[1])
print("Worst 10 classes by F1:")
for name, f1 in per_class[:10]:
    print(f"  {name:55s}: {f1:.4f}")
print("Best 10 classes by F1:")
for name, f1 in per_class[-10:]:
    print(f"  {name:55s}: {f1:.4f}")

print(f"\n✓ Step 3 (Triage) complete.")
print(f"  Best: {best_name} | Val F1: {best_val_f1:.4f} | Test F1: {best_test_f1:.4f}")
