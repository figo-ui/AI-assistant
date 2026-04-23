"""
STEP 3: Model Training — Task 2: Dialogue Intent Classification
Strategy:
  - TF-IDF + LinearSVC (fast, high accuracy for intent classification)
  - 73,416 rows, 9 intent classes
  - Target: beat Accuracy = 0.9994, Macro F1 = 0.9763
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

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, "data", "dataset_v1.0", "dialogue")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
section("Loading Dialogue Data")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

with open(os.path.join(DATA_DIR, "intent_names.json")) as f:
    intent_names = json.load(f)

X_train = df_train["text"].astype(str).values
y_train = df_train["label_id"].values
X_val   = df_val["text"].astype(str).values
y_val   = df_val["label_id"].values
X_test  = df_test["text"].astype(str).values
y_test  = df_test["label_id"].values

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(f"Intent classes: {len(intent_names)}: {intent_names}")
print(f"Class distribution (train):")
for i, name in enumerate(intent_names):
    count = (y_train == i).sum()
    print(f"  {i}: {name:20s} — {count:,}")

# ── Vectorize ──────────────────────────────────────────────────────────────
section("Vectorizing")
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    strip_accents="unicode",
)
X_tr = vec.fit_transform(X_train)
X_va = vec.transform(X_val)
X_te = vec.transform(X_test)
print(f"TF-IDF shape: {X_tr.shape}, time: {time.time()-t0:.1f}s")

# ── Model A: LinearSVC C=1.0 ───────────────────────────────────────────────
section("Model A: LinearSVC C=1.0")
t0 = time.time()
svc_a = LinearSVC(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
svc_a.fit(X_tr, y_train)
t1 = time.time()
y_va_a = svc_a.predict(X_va)
y_te_a = svc_a.predict(X_te)
y_tr_a = svc_a.predict(X_tr)
tr_a = f1_score(y_train, y_tr_a, average="macro")
va_a = f1_score(y_val, y_va_a, average="macro")
te_a = f1_score(y_test, y_te_a, average="macro")
acc_a = accuracy_score(y_val, y_va_a)
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_a:.4f} | Val F1: {va_a:.4f} | Test F1: {te_a:.4f} | Val Acc: {acc_a:.4f}")

# ── Model B: LinearSVC C=0.5 ───────────────────────────────────────────────
section("Model B: LinearSVC C=0.5")
t0 = time.time()
svc_b = LinearSVC(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
svc_b.fit(X_tr, y_train)
t1 = time.time()
y_va_b = svc_b.predict(X_va)
y_te_b = svc_b.predict(X_te)
y_tr_b = svc_b.predict(X_tr)
tr_b = f1_score(y_train, y_tr_b, average="macro")
va_b = f1_score(y_val, y_va_b, average="macro")
te_b = f1_score(y_test, y_te_b, average="macro")
acc_b = accuracy_score(y_val, y_va_b)
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_b:.4f} | Val F1: {va_b:.4f} | Test F1: {te_b:.4f} | Val Acc: {acc_b:.4f}")

# ── Model C: SGD modified_huber ────────────────────────────────────────────
section("Model C: SGD modified_huber")
t0 = time.time()
sgd_c = SGDClassifier(
    loss="modified_huber",
    alpha=1e-4,
    max_iter=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
sgd_c.fit(X_tr, y_train)
t1 = time.time()
y_va_c = sgd_c.predict(X_va)
y_te_c = sgd_c.predict(X_te)
y_tr_c = sgd_c.predict(X_tr)
tr_c = f1_score(y_train, y_tr_c, average="macro")
va_c = f1_score(y_val, y_va_c, average="macro")
te_c = f1_score(y_test, y_te_c, average="macro")
acc_c = accuracy_score(y_val, y_va_c)
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_c:.4f} | Val F1: {va_c:.4f} | Test F1: {te_c:.4f} | Val Acc: {acc_c:.4f}")

# ── Model D: LinearSVC C=1.0 + Calibration (for probabilities) ────────────
section("Model D: LinearSVC C=1.0 + Platt Calibration")
t0 = time.time()
svc_d_base = LinearSVC(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
svc_d = CalibratedClassifierCV(svc_d_base, cv=3, method="sigmoid")
svc_d.fit(X_tr, y_train)
t1 = time.time()
y_va_d = svc_d.predict(X_va)
y_te_d = svc_d.predict(X_te)
y_tr_d = svc_d.predict(X_tr)
tr_d = f1_score(y_train, y_tr_d, average="macro")
va_d = f1_score(y_val, y_va_d, average="macro")
te_d = f1_score(y_test, y_te_d, average="macro")
acc_d = accuracy_score(y_val, y_va_d)
print(f"Time: {t1-t0:.1f}s | Train F1: {tr_d:.4f} | Val F1: {va_d:.4f} | Test F1: {te_d:.4f} | Val Acc: {acc_d:.4f}")

# ── Select best ────────────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("LinearSVC C=1.0",       va_a, te_a, tr_a, acc_a, svc_a, y_va_a, y_te_a),
    ("LinearSVC C=0.5",       va_b, te_b, tr_b, acc_b, svc_b, y_va_b, y_te_b),
    ("SGD modified_huber",    va_c, te_c, tr_c, acc_c, sgd_c, y_va_c, y_te_c),
    ("LinearSVC+Calibration", va_d, te_d, tr_d, acc_d, svc_d, y_va_d, y_te_d),
]

print(f"\n{'Model':30s} | Val F1  | Test F1 | Train F1 | Val Acc")
print("-" * 70)
for name, vf1, tf1, trf1, acc, _, __, ___ in results:
    print(f"  {name:28s} | {vf1:.4f}  | {tf1:.4f}  | {trf1:.4f}   | {acc:.4f}")

best_idx = max(range(len(results)), key=lambda i: results[i][0])
best_name, best_val_f1, best_test_f1, best_train_f1, best_acc, best_model, best_y_va, best_y_te = results[best_idx]
print(f"\nBest model: {best_name}")
print(f"  Val Macro F1:  {best_val_f1:.4f}")
print(f"  Val Accuracy:  {best_acc:.4f}")
print(f"  Baseline F1:   0.9763, Baseline Acc: 0.9994")
print(f"  F1 Improvement: {best_val_f1 - 0.9763:+.4f}")

# ── Save best model ────────────────────────────────────────────────────────
section("Saving Best Dialogue Model")

joblib.dump(best_model, os.path.join(MODEL_DIR, "dialogue_intent_classifier.joblib"))
joblib.dump(vec, os.path.join(MODEL_DIR, "dialogue_intent_vectorizer.joblib"))

with open(os.path.join(MODEL_DIR, "dialogue_intent_labels.json"), "w") as f:
    json.dump(intent_names, f, indent=2)

metrics = {
    "model_name": best_name,
    "train_macro_f1": round(float(best_train_f1), 4),
    "val_macro_f1": round(float(best_val_f1), 4),
    "test_macro_f1": round(float(best_test_f1), 4),
    "val_accuracy": round(float(best_acc), 4),
    "test_accuracy": round(float(accuracy_score(y_test, best_y_te)), 4),
    "intent_classes": len(intent_names),
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "baseline_macro_f1": 0.9763,
    "baseline_accuracy": 0.9994,
    "improvement_f1": round(float(best_val_f1) - 0.9763, 4),
    "all_models": [
        {"name": r[0], "val_f1": round(r[1], 4), "test_f1": round(r[2], 4), "val_acc": round(r[4], 4)}
        for r in results
    ]
}
with open(os.path.join(MODEL_DIR, "dialogue_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved: dialogue_intent_classifier.joblib")
print(f"Saved: dialogue_intent_vectorizer.joblib")
print(f"Saved: dialogue_intent_labels.json ({len(intent_names)} intents)")
print(f"Saved: dialogue_training_metrics.json")

# Per-class report
section("Per-Class Report (Best Model — Val Set)")
print(classification_report(y_val, best_y_va, target_names=intent_names))

print(f"\n✓ Step 3 (Dialogue) complete.")
print(f"  Best: {best_name} | Val F1: {best_val_f1:.4f} | Val Acc: {best_acc:.4f}")
