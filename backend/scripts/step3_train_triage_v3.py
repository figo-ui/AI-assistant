"""
STEP 3: Model Training — Task 1: Triage Text Classification (v3 — optimized)
Key insight: texts are very long (mean 87 words) — truncate to first 100 words
for faster vectorization. Use word TF-IDF only (char is too slow on long texts).
Models: SGD (fast) + LinearSVC (best F1 for text classification)
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, "data", "dataset_v1.0", "triage")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

np.random.seed(42)

def truncate_text(text, max_words=150):
    """Truncate to first max_words words to speed up vectorization."""
    words = str(text).split()
    return " ".join(words[:max_words])

# ── Load data ──────────────────────────────────────────────────────────────
section("Loading Data")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

with open(os.path.join(DATA_DIR, "label_names.json")) as f:
    label_names = json.load(f)

# Truncate texts for speed
X_train_raw = df_train["symptom_text"].astype(str).apply(truncate_text).values
X_val_raw   = df_val["symptom_text"].astype(str).apply(truncate_text).values
X_test_raw  = df_test["symptom_text"].astype(str).apply(truncate_text).values
y_train = df_train["label_id"].values
y_val   = df_val["label_id"].values
y_test  = df_test["label_id"].values

print(f"Train: {len(X_train_raw):,}, Val: {len(X_val_raw):,}, Test: {len(X_test_raw):,}")
print(f"Classes: {len(label_names)}")
avg_words = np.mean([len(t.split()) for t in X_train_raw[:1000]])
print(f"Avg words after truncation (sample): {avg_words:.1f}")

# ── Vectorize ──────────────────────────────────────────────────────────────
section("Vectorizing (TF-IDF word 1-2gram)")
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    strip_accents="unicode",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
)
X_tr = vec.fit_transform(X_train_raw)
X_va = vec.transform(X_val_raw)
X_te = vec.transform(X_test_raw)
print(f"TF-IDF shape: {X_tr.shape}, time: {time.time()-t0:.1f}s")

# ── Model A: SGD (modified_huber, alpha=1e-4) ──────────────────────────────
section("Model A: SGD modified_huber alpha=1e-4")
t0 = time.time()
sgd_a = SGDClassifier(
    loss="modified_huber",
    alpha=1e-4,
    max_iter=200,
    tol=1e-4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
sgd_a.fit(X_tr, y_train)
t1 = time.time()
y_tr_a = sgd_a.predict(X_tr)
y_va_a = sgd_a.predict(X_va)
y_te_a = sgd_a.predict(X_te)
tr_a = f1_score(y_train, y_tr_a, average="macro")
va_a = f1_score(y_val, y_va_a, average="macro")
te_a = f1_score(y_test, y_te_a, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_a:.4f} | Val: {va_a:.4f} | Test: {te_a:.4f} | Gap: {tr_a-va_a:.4f}")

# ── Model B: SGD (modified_huber, alpha=5e-5) ──────────────────────────────
section("Model B: SGD modified_huber alpha=5e-5")
t0 = time.time()
sgd_b = SGDClassifier(
    loss="modified_huber",
    alpha=5e-5,
    max_iter=200,
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
tr_b = f1_score(y_train, y_tr_b, average="macro")
va_b = f1_score(y_val, y_va_b, average="macro")
te_b = f1_score(y_test, y_te_b, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_b:.4f} | Val: {va_b:.4f} | Test: {te_b:.4f} | Gap: {tr_b-va_b:.4f}")

# ── Model C: LinearSVC C=0.5 ───────────────────────────────────────────────
section("Model C: LinearSVC C=0.5")
t0 = time.time()
svc_c = LinearSVC(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
svc_c.fit(X_tr, y_train)
t1 = time.time()
y_tr_c = svc_c.predict(X_tr)
y_va_c = svc_c.predict(X_va)
y_te_c = svc_c.predict(X_te)
tr_c = f1_score(y_train, y_tr_c, average="macro")
va_c = f1_score(y_val, y_va_c, average="macro")
te_c = f1_score(y_test, y_te_c, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_c:.4f} | Val: {va_c:.4f} | Test: {te_c:.4f} | Gap: {tr_c-va_c:.4f}")

# ── Model D: LinearSVC C=1.0 ───────────────────────────────────────────────
section("Model D: LinearSVC C=1.0")
t0 = time.time()
svc_d = LinearSVC(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
svc_d.fit(X_tr, y_train)
t1 = time.time()
y_tr_d = svc_d.predict(X_tr)
y_va_d = svc_d.predict(X_va)
y_te_d = svc_d.predict(X_te)
tr_d = f1_score(y_train, y_tr_d, average="macro")
va_d = f1_score(y_val, y_va_d, average="macro")
te_d = f1_score(y_test, y_te_d, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_d:.4f} | Val: {va_d:.4f} | Test: {te_d:.4f} | Gap: {tr_d-va_d:.4f}")

# ── Model E: LinearSVC C=2.0 ───────────────────────────────────────────────
section("Model E: LinearSVC C=2.0")
t0 = time.time()
svc_e = LinearSVC(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
svc_e.fit(X_tr, y_train)
t1 = time.time()
y_tr_e = svc_e.predict(X_tr)
y_va_e = svc_e.predict(X_va)
y_te_e = svc_e.predict(X_te)
tr_e = f1_score(y_train, y_tr_e, average="macro")
va_e = f1_score(y_val, y_va_e, average="macro")
te_e = f1_score(y_test, y_te_e, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_e:.4f} | Val: {va_e:.4f} | Test: {te_e:.4f} | Gap: {tr_e-va_e:.4f}")

# ── Select best ────────────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("SGD alpha=1e-4",   va_a, te_a, tr_a, sgd_a, y_va_a, y_te_a),
    ("SGD alpha=5e-5",   va_b, te_b, tr_b, sgd_b, y_va_b, y_te_b),
    ("LinearSVC C=0.5",  va_c, te_c, tr_c, svc_c, y_va_c, y_te_c),
    ("LinearSVC C=1.0",  va_d, te_d, tr_d, svc_d, y_va_d, y_te_d),
    ("LinearSVC C=2.0",  va_e, te_e, tr_e, svc_e, y_va_e, y_te_e),
]

print(f"\n{'Model':25s} | Val F1  | Test F1 | Train F1 | Gap")
print("-" * 65)
for name, vf1, tf1, trf1, _, __, ___ in results:
    print(f"  {name:23s} | {vf1:.4f}  | {tf1:.4f}  | {trf1:.4f}   | {trf1-vf1:.4f}")

best_idx = max(range(len(results)), key=lambda i: results[i][0])
best_name, best_val_f1, best_test_f1, best_train_f1, best_model, best_y_va, best_y_te = results[best_idx]
print(f"\nBest model: {best_name}")
print(f"  Val Macro F1:  {best_val_f1:.4f}")
print(f"  Test Macro F1: {best_test_f1:.4f}")
print(f"  Baseline:      0.7709")
print(f"  Improvement:   {best_val_f1 - 0.7709:+.4f}")

# ── Save best model ────────────────────────────────────────────────────────
section("Saving Best Triage Model")

joblib.dump(best_model, os.path.join(MODEL_DIR, "text_classifier.joblib"))
joblib.dump(vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

# Remove old char vectorizer if exists
char_path = os.path.join(MODEL_DIR, "tfidf_char_vectorizer.joblib")
if os.path.exists(char_path):
    os.remove(char_path)

with open(os.path.join(MODEL_DIR, "text_labels.json"), "w") as f:
    json.dump(label_names, f, indent=2)

metrics = {
    "model_name": best_name,
    "train_macro_f1": round(float(best_train_f1), 4),
    "val_macro_f1": round(float(best_val_f1), 4),
    "test_macro_f1": round(float(best_test_f1), 4),
    "val_accuracy": round(float(accuracy_score(y_val, best_y_va)), 4),
    "test_accuracy": round(float(accuracy_score(y_test, best_y_te)), 4),
    "classes": len(label_names),
    "train_samples": int(len(X_train_raw)),
    "val_samples": int(len(X_val_raw)),
    "test_samples": int(len(X_test_raw)),
    "baseline_macro_f1": 0.7709,
    "improvement": round(float(best_val_f1) - 0.7709, 4),
    "vectorizer_mode": "word_1_2gram_truncated150",
    "all_models": [
        {"name": r[0], "val_f1": round(r[1], 4), "test_f1": round(r[2], 4), "train_f1": round(r[3], 4)}
        for r in results
    ]
}
with open(os.path.join(MODEL_DIR, "text_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved: text_classifier.joblib, tfidf_vectorizer.joblib")
print(f"Saved: text_labels.json ({len(label_names)} classes)")
print(f"Saved: text_training_metrics.json")

# Per-class report
section("Per-Class Report (Best Model — Val Set)")
report = classification_report(y_val, best_y_va, target_names=label_names, output_dict=True)
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
