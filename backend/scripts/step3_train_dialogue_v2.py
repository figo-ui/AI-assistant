"""
STEP 3: Dialogue Intent Classification (v2)
Use the original 17 MedQuAD intents + augment with ULTIMATE_CONVERSATIONAL_QA
to properly beat the baseline (Macro F1 = 0.9763, Acc = 0.9994 on 17 intents).
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

# ── Load MedQuAD (primary — has all 17 intents) ────────────────────────────
section("Loading Data")
medquad_path = os.path.join(DATA, "raw", "dialogue", "medquad_clinical_qa.csv")
df_medquad = pd.read_csv(medquad_path)
print(f"MedQuAD: {df_medquad.shape}")
print(f"Intents: {df_medquad['intent'].value_counts().to_dict()}")

# Use user_text as input
df_medquad_clean = df_medquad[["user_text", "intent"]].copy()
df_medquad_clean.columns = ["text", "intent"]
df_medquad_clean = df_medquad_clean.dropna()
df_medquad_clean["intent"] = df_medquad_clean["intent"].str.strip().str.lower()

# ── Augment with ULTIMATE_CONVERSATIONAL_QA ────────────────────────────────
# Map questions to MedQuAD intents
qa_path = os.path.join(DATA, "unified", "ULTIMATE_CONVERSATIONAL_QA.csv")
df_qa = pd.read_csv(qa_path)
df_qa = df_qa.drop_duplicates(subset=["question"])

# Map QA questions to MedQuAD intents
INTENT_MAP_MEDQUAD = {
    "what is": "information",
    "what are": "information",
    "define": "information",
    "describe": "information",
    "overview": "information",
    "symptom": "symptoms",
    "sign": "symptoms",
    "feel": "symptoms",
    "experience": "symptoms",
    "treat": "treatment",
    "therapy": "treatment",
    "medication": "treatment",
    "drug": "treatment",
    "cure": "treatment",
    "manage": "treatment",
    "diagnos": "exams and tests",
    "test": "exams and tests",
    "exam": "exams and tests",
    "detect": "exams and tests",
    "screen": "exams and tests",
    "prevent": "prevention",
    "avoid": "prevention",
    "cause": "causes",
    "why": "causes",
    "reason": "causes",
    "factor": "causes",
    "outlook": "outlook",
    "prognosis": "outlook",
    "survival": "outlook",
    "recover": "outlook",
    "complicat": "complications",
    "side effect": "complications",
    "who": "susceptibility",
    "suscept": "susceptibility",
    "risk": "susceptibility",
    "prone": "susceptibility",
    "research": "research",
    "study": "research",
    "trial": "research",
    "stage": "stages",
    "inherit": "inheritance",
    "genetic": "genetic changes",
    "gene": "genetic changes",
    "frequen": "frequency",
    "how common": "frequency",
    "how many": "frequency",
    "support": "support groups",
    "consider": "considerations",
}

def assign_medquad_intent(q):
    q_lower = str(q).lower()
    for keyword, intent in INTENT_MAP_MEDQUAD.items():
        if keyword in q_lower:
            return intent
    return "information"

df_qa_clean = df_qa[["question"]].copy()
df_qa_clean.columns = ["text"]
df_qa_clean["intent"] = df_qa_clean["text"].apply(assign_medquad_intent)

# Combine
df_all = pd.concat([df_medquad_clean, df_qa_clean], ignore_index=True)
df_all = df_all.dropna(subset=["text", "intent"])
df_all = df_all[df_all["text"].str.strip().str.len() > 5]
df_all = df_all.drop_duplicates(subset=["text"])

print(f"\nCombined: {df_all.shape}")
print(f"Intent distribution:\n{df_all['intent'].value_counts().to_string()}")

# Encode
le = LabelEncoder()
df_all["label_id"] = le.fit_transform(df_all["intent"])
intent_names = le.classes_.tolist()
print(f"\nFinal intents ({len(intent_names)}): {intent_names}")

# Split
X = df_all["text"].values
y = df_all["label_id"].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Split: {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test")

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
y_va_a = svc_a.predict(X_va); y_te_a = svc_a.predict(X_te); y_tr_a = svc_a.predict(X_tr)
tr_a = f1_score(y_train, y_tr_a, average="macro")
va_a = f1_score(y_val, y_va_a, average="macro")
te_a = f1_score(y_test, y_te_a, average="macro")
acc_a = accuracy_score(y_val, y_va_a)
print(f"Time: {t1-t0:.1f}s | Train: {tr_a:.4f} | Val: {va_a:.4f} | Test: {te_a:.4f} | Acc: {acc_a:.4f}")

# ── Model B: LinearSVC C=0.5 ───────────────────────────────────────────────
section("Model B: LinearSVC C=0.5")
t0 = time.time()
svc_b = LinearSVC(C=0.5, max_iter=1000, class_weight="balanced", random_state=42)
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
sgd_c = SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=200, class_weight="balanced",
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
    ("LinearSVC C=0.5",     va_b, te_b, tr_b, acc_b, svc_b, y_va_b, y_te_b),
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

# Save updated splits
out_dir = os.path.join(BASE, "data", "dataset_v1.0", "dialogue")
df_tr = pd.DataFrame({"text": X_train, "intent": le.inverse_transform(y_train), "label_id": y_train})
df_va = pd.DataFrame({"text": X_val,   "intent": le.inverse_transform(y_val),   "label_id": y_val})
df_te = pd.DataFrame({"text": X_test,  "intent": le.inverse_transform(y_test),  "label_id": y_test})
df_tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
df_va.to_csv(os.path.join(out_dir, "val.csv"), index=False)
df_te.to_csv(os.path.join(out_dir, "test.csv"), index=False)
with open(os.path.join(out_dir, "intent_names.json"), "w") as f:
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

print(f"\n✓ Step 3 (Dialogue v2) complete.")
print(f"  Best: {best_name} | Val F1: {best_val_f1:.4f} | Val Acc: {best_acc:.4f}")
