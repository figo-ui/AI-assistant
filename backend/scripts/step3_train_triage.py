"""
STEP 3: Model Training — Task 1: Triage Text Classification
Strategy:
  - Baseline: TF-IDF + LogisticRegression (reproduce existing approach)
  - Advanced: TF-IDF + LinearSVC with calibration (faster, better for many classes)
  - Best: TF-IDF (char+word ngrams) + SGDClassifier with L2 regularization
  - All trained on 60,553 rows / 115 classes
  - Target: beat Macro F1 = 0.7709
"""
import sys, os, warnings, json, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                              confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

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

# Compute class weights
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(label_names)),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

# ── Model 1: Baseline — TF-IDF word + LR (reproduce existing) ─────────────
section("Model 1: TF-IDF (word) + LogisticRegression [BASELINE]")
t0 = time.time()

vec1 = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=60000,
    sublinear_tf=True,
    min_df=3,
    analyzer="word",
    strip_accents="unicode",
)
X_train_v1 = vec1.fit_transform(X_train)
X_val_v1   = vec1.transform(X_val)
X_test_v1  = vec1.transform(X_test)

lr1 = LogisticRegression(
    C=5.0,
    max_iter=300,
    solver="lbfgs",
    multi_class="auto",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
lr1.fit(X_train_v1, y_train)
t1 = time.time()

y_val_pred1 = lr1.predict(X_val_v1)
y_test_pred1 = lr1.predict(X_test_v1)
y_train_pred1 = lr1.predict(X_train_v1)

train_f1_1 = f1_score(y_train, y_train_pred1, average="macro")
val_f1_1   = f1_score(y_val, y_val_pred1, average="macro")
test_f1_1  = f1_score(y_test, y_test_pred1, average="macro")
val_acc_1  = accuracy_score(y_val, y_val_pred1)

print(f"Training time: {t1-t0:.1f}s")
print(f"Train Macro F1: {train_f1_1:.4f}")
print(f"Val   Macro F1: {val_f1_1:.4f}")
print(f"Test  Macro F1: {test_f1_1:.4f}")
print(f"Val   Accuracy: {val_acc_1:.4f}")
print(f"Overfitting gap (train-val): {train_f1_1 - val_f1_1:.4f}")

# ── Model 2: TF-IDF (word+char) + LinearSVC + Calibration ─────────────────
section("Model 2: TF-IDF (word+char ngrams) + LinearSVC + Calibration [ADVANCED]")
t0 = time.time()

# Word-level TF-IDF
vec2_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=80000,
    sublinear_tf=True,
    min_df=3,
    analyzer="word",
    strip_accents="unicode",
)
# Char-level TF-IDF (captures morphological patterns)
vec2_char = TfidfVectorizer(
    ngram_range=(3, 4),
    max_features=30000,
    sublinear_tf=True,
    min_df=5,
    analyzer="char_wb",
    strip_accents="unicode",
)

from scipy.sparse import hstack

X_train_word = vec2_word.fit_transform(X_train)
X_train_char = vec2_char.fit_transform(X_train)
X_train_v2 = hstack([X_train_word, X_train_char])

X_val_word = vec2_word.transform(X_val)
X_val_char = vec2_char.transform(X_val)
X_val_v2 = hstack([X_val_word, X_val_char])

X_test_word = vec2_word.transform(X_test)
X_test_char = vec2_char.transform(X_test)
X_test_v2 = hstack([X_test_word, X_test_char])

svc2 = LinearSVC(
    C=0.5,
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
# Wrap with Platt scaling for probability output — use cv=2 for speed
cal2 = CalibratedClassifierCV(svc2, cv=2, method="sigmoid")
cal2.fit(X_train_v2, y_train)
t1 = time.time()

y_val_pred2   = cal2.predict(X_val_v2)
y_test_pred2  = cal2.predict(X_test_v2)
y_train_pred2 = cal2.predict(X_train_v2)

train_f1_2 = f1_score(y_train, y_train_pred2, average="macro")
val_f1_2   = f1_score(y_val, y_val_pred2, average="macro")
test_f1_2  = f1_score(y_test, y_test_pred2, average="macro")
val_acc_2  = accuracy_score(y_val, y_val_pred2)

print(f"Training time: {t1-t0:.1f}s")
print(f"Train Macro F1: {train_f1_2:.4f}")
print(f"Val   Macro F1: {val_f1_2:.4f}")
print(f"Test  Macro F1: {test_f1_2:.4f}")
print(f"Val   Accuracy: {val_acc_2:.4f}")
print(f"Overfitting gap (train-val): {train_f1_2 - val_f1_2:.4f}")

# ── Model 3: TF-IDF + SGD with L2 (fast, regularized) ─────────────────────
section("Model 3: TF-IDF (word+char) + SGDClassifier [REGULARIZED]")
t0 = time.time()

sgd3 = SGDClassifier(
    loss="modified_huber",
    alpha=1e-4,
    max_iter=200,
    tol=1e-4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
sgd3.fit(X_train_v2, y_train)
t1 = time.time()

y_val_pred3   = sgd3.predict(X_val_v2)
y_test_pred3  = sgd3.predict(X_test_v2)
y_train_pred3 = sgd3.predict(X_train_v2)

train_f1_3 = f1_score(y_train, y_train_pred3, average="macro")
val_f1_3   = f1_score(y_val, y_val_pred3, average="macro")
test_f1_3  = f1_score(y_test, y_test_pred3, average="macro")
val_acc_3  = accuracy_score(y_val, y_val_pred3)

print(f"Training time: {t1-t0:.1f}s")
print(f"Train Macro F1: {train_f1_3:.4f}")
print(f"Val   Macro F1: {val_f1_3:.4f}")
print(f"Test  Macro F1: {test_f1_3:.4f}")
print(f"Val   Accuracy: {val_acc_3:.4f}")
print(f"Overfitting gap (train-val): {train_f1_3 - val_f1_3:.4f}")

# ── Select best model ──────────────────────────────────────────────────────
section("Model Selection")
results = [
    ("LR (word TF-IDF)",         val_f1_1, test_f1_1, train_f1_1, lr1, vec1, None),
    ("LinearSVC+Cal (word+char)", val_f1_2, test_f1_2, train_f1_2, cal2, (vec2_word, vec2_char), "combined"),
    ("SGD (word+char)",           val_f1_3, test_f1_3, train_f1_3, sgd3, (vec2_word, vec2_char), "combined"),
]

for name, vf1, tf1, trf1, _, __, ___ in results:
    print(f"  {name:35s} | Val F1: {vf1:.4f} | Test F1: {tf1:.4f} | Train F1: {trf1:.4f} | Gap: {trf1-vf1:.4f}")

best_idx = max(range(len(results)), key=lambda i: results[i][1])
best_name, best_val_f1, best_test_f1, best_train_f1, best_model, best_vec, best_mode = results[best_idx]
print(f"\nBest model: {best_name} (Val F1: {best_val_f1:.4f})")
print(f"Baseline was: Macro F1 = 0.7709 on 11,076 samples")
print(f"Improvement: {best_val_f1 - 0.7709:+.4f}")

# ── Save best model ────────────────────────────────────────────────────────
section("Saving Best Triage Model")

if best_mode == "combined":
    vec_word, vec_char = best_vec
    joblib.dump(best_model, os.path.join(MODEL_DIR, "text_classifier.joblib"))
    joblib.dump(vec_word, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(vec_char, os.path.join(MODEL_DIR, "tfidf_char_vectorizer.joblib"))
    print("Saved: text_classifier.joblib, tfidf_vectorizer.joblib, tfidf_char_vectorizer.joblib")
else:
    joblib.dump(best_model, os.path.join(MODEL_DIR, "text_classifier.joblib"))
    joblib.dump(best_vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    print("Saved: text_classifier.joblib, tfidf_vectorizer.joblib")

with open(os.path.join(MODEL_DIR, "text_labels.json"), "w") as f:
    json.dump(label_names, f, indent=2)
print(f"Saved: text_labels.json ({len(label_names)} classes)")

# Save training metrics
metrics = {
    "model_name": best_name,
    "train_macro_f1": round(float(best_train_f1), 4),
    "val_macro_f1": round(float(best_val_f1), 4),
    "test_macro_f1": round(float(best_test_f1), 4),
    "val_accuracy": round(float(accuracy_score(y_val, 
        results[best_idx][4].predict(X_val_v2 if best_mode == "combined" else X_val_v1))), 4),
    "classes": len(label_names),
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "baseline_macro_f1": 0.7709,
    "improvement": round(float(best_val_f1) - 0.7709, 4),
    "vectorizer_mode": best_mode or "word_only",
    "all_models": [
        {"name": r[0], "val_f1": round(r[1], 4), "test_f1": round(r[2], 4), "train_f1": round(r[3], 4)}
        for r in results
    ]
}
with open(os.path.join(MODEL_DIR, "text_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: text_training_metrics.json")

# Per-class report for best model
section("Per-Class Report (Best Model — Val Set)")
if best_mode == "combined":
    y_val_best = results[best_idx][4].predict(X_val_v2)
else:
    y_val_best = results[best_idx][4].predict(X_val_v1)

report = classification_report(y_val, y_val_best, target_names=label_names, output_dict=True)
# Show worst 10 classes
per_class = [(label_names[i], report.get(label_names[i], {}).get("f1-score", 0.0)) 
             for i in range(len(label_names)) if label_names[i] in report]
per_class.sort(key=lambda x: x[1])
print("Worst 10 classes by F1:")
for name, f1 in per_class[:10]:
    print(f"  {name:50s}: {f1:.4f}")
print("Best 10 classes by F1:")
for name, f1 in per_class[-10:]:
    print(f"  {name:50s}: {f1:.4f}")

print(f"\n✓ Step 3 (Triage) complete. Best model: {best_name}")
print(f"  Val Macro F1: {best_val_f1:.4f} (baseline: 0.7709, improvement: {best_val_f1-0.7709:+.4f})")
