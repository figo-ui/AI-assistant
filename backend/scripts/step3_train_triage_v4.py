"""
STEP 3: Model Training — Task 1: Triage Text Classification (v4 — smart data selection)
Key insight: The unified dataset has MIMIC-style long clinical texts that are noisy.
The processed dataset (28k rows, 113 classes) is clean symptom text.
Strategy: Use ONLY the clean processed dataset + filter unified to symptom-style rows only.
Truncate to 50 words max. Use LinearSVC which is the gold standard for text classification.
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
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

np.random.seed(42)

def truncate_text(text, max_words=50):
    words = str(text).split()
    return " ".join(words[:max_words])

# ── Load and prepare clean data ────────────────────────────────────────────
section("Loading Clean Data")

# Primary: processed dataset (clean symptom text, 113 classes)
proc_path = os.path.join(DATA, "processed", "expanded_symptom_condition_processed_min5_rebalanced.csv")
df_proc = pd.read_csv(proc_path)
print(f"Processed dataset: {df_proc.shape}")

# Secondary: unified dataset — filter to SHORT symptom-style texts only
triage_path = os.path.join(DATA, "unified", "ULTIMATE_TRIAGE_KNOWLEDGE.csv")
df_unified = pd.read_csv(triage_path)

# Filter unified: keep only rows with 3-50 words (symptom-style, not MIMIC clinical notes)
df_unified["word_count"] = df_unified["symptom_text"].astype(str).str.split().str.len()
df_unified_short = df_unified[(df_unified["word_count"] >= 3) & (df_unified["word_count"] <= 50)].copy()
print(f"Unified (short symptom texts 3-50 words): {df_unified_short.shape}")

# Filter unified to conditions with >= 50 samples
vc = df_unified_short["condition"].value_counts()
valid_conds = set(vc[vc >= 50].index.tolist())
df_unified_short = df_unified_short[df_unified_short["condition"].isin(valid_conds)].copy()
print(f"Unified after condition filter: {df_unified_short.shape}")

# Combine
df_proc_clean = df_proc[["symptom_text", "condition"]].copy()
df_unified_clean = df_unified_short[["symptom_text", "condition"]].copy()

df_combined = pd.concat([df_proc_clean, df_unified_clean], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["symptom_text"])
df_combined["symptom_text"] = df_combined["symptom_text"].astype(str).str.strip()
df_combined["condition"] = df_combined["condition"].astype(str).str.strip()

# Final class filter: >= 20 samples
vc_final = df_combined["condition"].value_counts()
valid_final = set(vc_final[vc_final >= 20].index.tolist())
df_final = df_combined[df_combined["condition"].isin(valid_final)].copy()
print(f"Final combined dataset: {df_final.shape}, {df_final['condition'].nunique()} classes")

# Truncate texts
df_final["symptom_text"] = df_final["symptom_text"].apply(truncate_text)
avg_words = df_final["symptom_text"].str.split().str.len().mean()
print(f"Avg words after truncation: {avg_words:.1f}")

# Encode labels
le = LabelEncoder()
df_final["label_id"] = le.fit_transform(df_final["condition"])
label_names = le.classes_.tolist()
print(f"Label classes: {len(label_names)}")

# Split
X = df_final["symptom_text"].values
y = df_final["label_id"].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Split: {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test")

# ── Vectorize ──────────────────────────────────────────────────────────────
section("Vectorizing")
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    strip_accents="unicode",
)
X_tr = vec.fit_transform(X_train)
X_va = vec.transform(X_val)
X_te = vec.transform(X_test)
print(f"TF-IDF shape: {X_tr.shape}, time: {time.time()-t0:.1f}s")

# ── Model A: SGD ───────────────────────────────────────────────────────────
section("Model A: SGD modified_huber")
t0 = time.time()
sgd_a = SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=200, tol=1e-4,
                       class_weight="balanced", random_state=42, n_jobs=-1)
sgd_a.fit(X_tr, y_train)
t1 = time.time()
y_va_a = sgd_a.predict(X_va); y_te_a = sgd_a.predict(X_te); y_tr_a = sgd_a.predict(X_tr)
tr_a = f1_score(y_train, y_tr_a, average="macro")
va_a = f1_score(y_val, y_va_a, average="macro")
te_a = f1_score(y_test, y_te_a, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_a:.4f} | Val: {va_a:.4f} | Test: {te_a:.4f} | Gap: {tr_a-va_a:.4f}")

# ── Model B: LinearSVC C=0.3 ───────────────────────────────────────────────
section("Model B: LinearSVC C=0.3")
t0 = time.time()
svc_b = LinearSVC(C=0.3, max_iter=2000, class_weight="balanced", random_state=42)
svc_b.fit(X_tr, y_train)
t1 = time.time()
y_va_b = svc_b.predict(X_va); y_te_b = svc_b.predict(X_te); y_tr_b = svc_b.predict(X_tr)
tr_b = f1_score(y_train, y_tr_b, average="macro")
va_b = f1_score(y_val, y_va_b, average="macro")
te_b = f1_score(y_test, y_te_b, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_b:.4f} | Val: {va_b:.4f} | Test: {te_b:.4f} | Gap: {tr_b-va_b:.4f}")

# ── Model C: LinearSVC C=1.0 ───────────────────────────────────────────────
section("Model C: LinearSVC C=1.0")
t0 = time.time()
svc_c = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
svc_c.fit(X_tr, y_train)
t1 = time.time()
y_va_c = svc_c.predict(X_va); y_te_c = svc_c.predict(X_te); y_tr_c = svc_c.predict(X_tr)
tr_c = f1_score(y_train, y_tr_c, average="macro")
va_c = f1_score(y_val, y_va_c, average="macro")
te_c = f1_score(y_test, y_te_c, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_c:.4f} | Val: {va_c:.4f} | Test: {te_c:.4f} | Gap: {tr_c-va_c:.4f}")

# ── Model D: LinearSVC C=2.0 ───────────────────────────────────────────────
section("Model D: LinearSVC C=2.0")
t0 = time.time()
svc_d = LinearSVC(C=2.0, max_iter=2000, class_weight="balanced", random_state=42)
svc_d.fit(X_tr, y_train)
t1 = time.time()
y_va_d = svc_d.predict(X_va); y_te_d = svc_d.predict(X_te); y_tr_d = svc_d.predict(X_tr)
tr_d = f1_score(y_train, y_tr_d, average="macro")
va_d = f1_score(y_val, y_va_d, average="macro")
te_d = f1_score(y_test, y_te_d, average="macro")
print(f"Time: {t1-t0:.1f}s | Train: {tr_d:.4f} | Val: {va_d:.4f} | Test: {te_d:.4f} | Gap: {tr_d-va_d:.4f}")

# ── Select best ────────────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("SGD alpha=1e-4",  va_a, te_a, tr_a, sgd_a, y_va_a, y_te_a),
    ("LinearSVC C=0.3", va_b, te_b, tr_b, svc_b, y_va_b, y_te_b),
    ("LinearSVC C=1.0", va_c, te_c, tr_c, svc_c, y_va_c, y_te_c),
    ("LinearSVC C=2.0", va_d, te_d, tr_d, svc_d, y_va_d, y_te_d),
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

# ── Save ───────────────────────────────────────────────────────────────────
section("Saving Best Triage Model")

joblib.dump(best_model, os.path.join(MODEL_DIR, "text_classifier.joblib"))
joblib.dump(vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

# Remove old char vectorizer if exists
for fname in ["tfidf_char_vectorizer.joblib", "tfidf_svd.joblib"]:
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        os.remove(p)

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
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "baseline_macro_f1": 0.7709,
    "improvement": round(float(best_val_f1) - 0.7709, 4),
    "vectorizer_mode": "word_1_2gram_truncated50",
    "all_models": [
        {"name": r[0], "val_f1": round(r[1], 4), "test_f1": round(r[2], 4), "train_f1": round(r[3], 4)}
        for r in results
    ]
}
with open(os.path.join(MODEL_DIR, "text_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Also save updated splits
out_dir = os.path.join(BASE, "data", "dataset_v1.0", "triage")
df_tr = pd.DataFrame({"symptom_text": X_train, "label_id": y_train, "condition": le.inverse_transform(y_train)})
df_va = pd.DataFrame({"symptom_text": X_val,   "label_id": y_val,   "condition": le.inverse_transform(y_val)})
df_te = pd.DataFrame({"symptom_text": X_test,  "label_id": y_test,  "condition": le.inverse_transform(y_test)})
df_tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
df_va.to_csv(os.path.join(out_dir, "val.csv"), index=False)
df_te.to_csv(os.path.join(out_dir, "test.csv"), index=False)
with open(os.path.join(out_dir, "label_names.json"), "w") as f:
    json.dump(label_names, f, indent=2)

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
