"""
STEP 3: Dialogue Intent Classification (v3 — MedQuAD only, beat baseline)
The baseline was trained on MedQuAD with 17 intents and got 0.9994 accuracy.
We train on the same data but with a better model and more features.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

np.random.seed(42)

# ── Load MedQuAD only ──────────────────────────────────────────────────────
section("Loading MedQuAD Data")
medquad_path = os.path.join(DATA, "raw", "dialogue", "medquad_clinical_qa.csv")
df = pd.read_csv(medquad_path)
print(f"MedQuAD: {df.shape}")

# Use user_text as input
df_clean = df[["user_text", "intent"]].copy()
df_clean.columns = ["text", "intent"]
df_clean = df_clean.dropna()
df_clean["intent"] = df_clean["intent"].str.strip().str.lower()
df_clean = df_clean[df_clean["text"].str.strip().str.len() > 5]
print(f"After cleaning: {df_clean.shape}")
print(f"Intent distribution:\n{df_clean['intent'].value_counts().to_string()}")

# Remove classes with < 5 samples (can't stratify)
vc = df_clean["intent"].value_counts()
valid_intents = vc[vc >= 5].index.tolist()
df_clean = df_clean[df_clean["intent"].isin(valid_intents)].copy()
print(f"After removing rare intents: {df_clean.shape}")

# Encode
le = LabelEncoder()
df_clean["label_id"] = le.fit_transform(df_clean["intent"])
intent_names = le.classes_.tolist()
print(f"\nIntents ({len(intent_names)}): {intent_names}")

# Split — same as baseline (80/20)
X = df_clean["text"].values
y = df_clean["label_id"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)
print(f"Split: {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test")

# ── Vectorize ──────────────────────────────────────────────────────────────
section("Vectorizing")
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True,
    min_df=1,
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
svc_a = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
svc_a.fit(X_tr, y_train)
t1 = time.time()
y_va_a = svc_a.predict(X_va); y_te_a = svc_a.predict(X_te); y_tr_a = svc_a.predict(X_tr)
tr_a = f1_score(y_train, y_tr_a, average="macro")
va_a = f1_score(y_val, y_va_a, average="macro")
te_a = f1_score(y_test, y_te_a, average="macro")
acc_a = accuracy_score(y_val, y_va_a)
print(f"Time: {t1-t0:.1f}s | Train: {tr_a:.4f} | Val: {va_a:.4f} | Test: {te_a:.4f} | Acc: {acc_a:.4f}")

# ── Model B: LinearSVC C=5.0 ───────────────────────────────────────────────
section("Model B: LinearSVC C=5.0")
t0 = time.time()
svc_b = LinearSVC(C=5.0, max_iter=2000, class_weight="balanced", random_state=42)
svc_b.fit(X_tr, y_train)
t1 = time.time()
y_va_b = svc_b.predict(X_va); y_te_b = svc_b.predict(X_te); y_tr_b = svc_b.predict(X_tr)
tr_b = f1_score(y_train, y_tr_b, average="macro")
va_b = f1_score(y_val, y_va_b, average="macro")
te_b = f1_score(y_test, y_te_b, average="macro")
acc_b = accuracy_score(y_val, y_va_b)
print(f"Time: {t1-t0:.1f}s | Train: {tr_b:.4f} | Val: {va_b:.4f} | Test: {te_b:.4f} | Acc: {acc_b:.4f}")

# ── Model C: SGD ───────────────────────────────────────────────────────────
section("Model C: SGD modified_huber")
t0 = time.time()
sgd_c = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=300, class_weight="balanced",
                       random_state=42, n_jobs=-1)
sgd_c.fit(X_tr, y_train)
t1 = time.time()
y_va_c = sgd_c.predict(X_va); y_te_c = sgd_c.predict(X_te); y_tr_c = sgd_c.predict(X_tr)
tr_c = f1_score(y_train, y_tr_c, average="macro")
va_c = f1_score(y_val, y_va_c, average="macro")
te_c = f1_score(y_test, y_te_c, average="macro")
acc_c = accuracy_score(y_val, y_va_c)
print(f"Time: {t1-t0:.1f}s | Train: {tr_c:.4f} | Val: {va_c:.4f} | Test: {te_c:.4f} | Acc: {acc_c:.4f}")

# ── Select best ────────────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("LinearSVC C=1.0",     va_a, te_a, tr_a, acc_a, svc_a, y_va_a, y_te_a),
    ("LinearSVC C=5.0",     va_b, te_b, tr_b, acc_b, svc_b, y_va_b, y_te_b),
    ("SGD modified_huber",  va_c, te_c, tr_c, acc_c, sgd_c, y_va_c, y_te_c),
]

print(f"\n{'Model':25s} | Val F1  | Test F1 | Train F1 | Val Acc")
print("-" * 65)
for name, vf1, tf1, trf1, acc, _, __, ___ in results:
    print(f"  {name:23s} | {vf1:.4f}  | {tf1:.4f}  | {trf1:.4f}   | {acc:.4f}")

best_idx = max(range(len(results)), key=lambda i: results[i][0])
best_name, best_val_f1, best_test_f1, best_train_f1, best_acc, best_model, best_y_va, best_y_te = results[best_idx]
print(f"\nBest model: {best_name}")
print(f"  Val Macro F1:  {best_val_f1:.4f}")
print(f"  Val Accuracy:  {best_acc:.4f}")
print(f"  Baseline F1:   0.9763, Baseline Acc: 0.9994")
print(f"  F1 Improvement: {best_val_f1 - 0.9763:+.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
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

print(f"Saved: dialogue_intent_classifier.joblib, dialogue_intent_vectorizer.joblib")
print(f"Saved: dialogue_intent_labels.json ({len(intent_names)} intents)")
print(f"Saved: dialogue_training_metrics.json")

section("Per-Class Report (Best Model — Val Set)")
print(classification_report(y_val, best_y_va, target_names=intent_names))

print(f"\n✓ Step 3 (Dialogue v3) complete.")
print(f"  Best: {best_name} | Val F1: {best_val_f1:.4f} | Val Acc: {best_acc:.4f}")
