"""
STEP 2: Restructure to the BEST Format
- Task 1 (Triage): Filter unified dataset to 87 conditions (>=50 samples),
  merge with processed dataset, deduplicate, save as Parquet + CSV
- Task 2 (Dialogue): Use medquad_clinical_qa with intent labels,
  augment with ULTIMATE_CONVERSATIONAL_QA (assign intents via keyword mapping)
- Task 3 (Image): Use dermamnist_64.npz with class weights computed,
  save split metadata
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "data", "dataset_v1.0")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "triage"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "dialogue"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "imaging"), exist_ok=True)

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: TRIAGE TEXT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
section("TASK 1: TRIAGE TEXT — Restructuring")

# Load unified triage dataset
triage_path = os.path.join(DATA, "unified", "ULTIMATE_TRIAGE_KNOWLEDGE.csv")
df_unified = pd.read_csv(triage_path)
print(f"Unified triage loaded: {df_unified.shape}")

# Load processed dataset (already clean, 113 classes)
proc_path = os.path.join(DATA, "processed", "expanded_symptom_condition_processed_min5_rebalanced.csv")
df_proc = pd.read_csv(proc_path)
print(f"Processed dataset loaded: {df_proc.shape}")

# Load existing labels to understand target class set
labels_path = os.path.join(BASE, "backend", "models", "text_labels.json")
with open(labels_path) as f:
    existing_labels = json.load(f)
print(f"Existing model labels: {len(existing_labels)} classes")

# ── Step 2a: Filter unified dataset to conditions with >= 50 samples ──────
vc = df_unified["condition"].value_counts()
valid_conditions_50 = set(vc[vc >= 50].index.tolist())
print(f"\nConditions with >= 50 samples in unified: {len(valid_conditions_50)}")

df_unified_filtered = df_unified[df_unified["condition"].isin(valid_conditions_50)].copy()
print(f"Unified after filtering: {df_unified_filtered.shape}")

# ── Step 2b: Clean text — remove MIMIC-style clinical noise ───────────────
# MIMIC rows have very short or clinical-code-style text
# Filter: keep rows where text has >= 3 words and looks like symptoms
def is_clean_symptom_text(text):
    text = str(text).strip().lower()
    if len(text.split()) < 3:
        return False
    # Reject pure lab/procedure text
    noise_patterns = [
        "r/o vancomycin", "negative by eia", "positive by eia",
        "gram stain", "urine culture", "blood culture", "specimen",
        "reference range", "formulary", "dose_val", "furosemide r/o"
    ]
    for pat in noise_patterns:
        if pat in text:
            return False
    return True

mask_clean = df_unified_filtered["symptom_text"].apply(is_clean_symptom_text)
df_unified_clean = df_unified_filtered[mask_clean].copy()
print(f"After noise filtering: {df_unified_clean.shape}")
print(f"Removed {df_unified_filtered.shape[0] - df_unified_clean.shape[0]} noisy rows")

# ── Step 2c: Normalize condition names ────────────────────────────────────
# Standardize casing
df_unified_clean["condition"] = df_unified_clean["condition"].str.strip()
df_unified_clean["symptom_text"] = df_unified_clean["symptom_text"].str.strip()

# ── Step 2d: Prepare processed dataset ────────────────────────────────────
# The processed dataset uses 'condition' column — normalize
df_proc_clean = df_proc[["symptom_text", "condition"]].copy()
df_proc_clean["symptom_text"] = df_proc_clean["symptom_text"].str.strip()
df_proc_clean["condition"] = df_proc_clean["condition"].str.strip()

# ── Step 2e: Merge datasets ────────────────────────────────────────────────
# Combine unified (filtered) + processed
df_combined = pd.concat([df_unified_clean[["symptom_text", "condition"]],
                          df_proc_clean[["symptom_text", "condition"]]], 
                         ignore_index=True)
print(f"\nCombined before dedup: {df_combined.shape}")

# Remove exact duplicates
df_combined = df_combined.drop_duplicates(subset=["symptom_text", "condition"])
print(f"After dedup: {df_combined.shape}")

# ── Step 2f: Final class filtering — keep classes with >= 30 samples ──────
vc_combined = df_combined["condition"].value_counts()
valid_final = set(vc_combined[vc_combined >= 30].index.tolist())
df_final = df_combined[df_combined["condition"].isin(valid_final)].copy()
print(f"After final class filter (>=30 samples): {df_final.shape}")
print(f"Final unique conditions: {df_final['condition'].nunique()}")

# ── Step 2g: Encode labels ─────────────────────────────────────────────────
le = LabelEncoder()
df_final["label_id"] = le.fit_transform(df_final["condition"])
label_names = le.classes_.tolist()
print(f"Label encoder classes: {len(label_names)}")

# ── Step 2h: Train/Val/Test split (stratified) ────────────────────────────
# 70% train, 15% val, 15% test
X = df_final["symptom_text"].values
y = df_final["label_id"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# ── Step 2i: Save splits ───────────────────────────────────────────────────
df_train = pd.DataFrame({"symptom_text": X_train, "label_id": y_train,
                          "condition": le.inverse_transform(y_train)})
df_val   = pd.DataFrame({"symptom_text": X_val,   "label_id": y_val,
                          "condition": le.inverse_transform(y_val)})
df_test  = pd.DataFrame({"symptom_text": X_test,  "label_id": y_test,
                          "condition": le.inverse_transform(y_test)})

triage_dir = os.path.join(OUT_DIR, "triage")
df_train.to_csv(os.path.join(triage_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(triage_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(triage_dir, "test.csv"), index=False)
df_final.to_csv(os.path.join(triage_dir, "full.csv"), index=False)

# Save as Parquet too
df_train.to_parquet(os.path.join(triage_dir, "train.parquet"), index=False)
df_val.to_parquet(os.path.join(triage_dir, "val.parquet"), index=False)
df_test.to_parquet(os.path.join(triage_dir, "test.parquet"), index=False)

# Save label map
with open(os.path.join(triage_dir, "label_names.json"), "w") as f:
    json.dump(label_names, f, indent=2)

# Save class distribution
vc_train = pd.Series(y_train).value_counts().sort_index()
class_dist = {label_names[i]: int(vc_train.get(i, 0)) for i in range(len(label_names))}
with open(os.path.join(triage_dir, "class_distribution.json"), "w") as f:
    json.dump(class_dist, f, indent=2)

print(f"\nTriage dataset saved to: {triage_dir}")
print(f"  train.csv/parquet, val.csv/parquet, test.csv/parquet, full.csv")
print(f"  label_names.json ({len(label_names)} classes)")

# Class balance stats
vc_final = df_final["condition"].value_counts()
print(f"\nFinal class balance:")
print(f"  Min samples: {vc_final.min()} ({vc_final.idxmin()})")
print(f"  Max samples: {vc_final.max()} ({vc_final.idxmax()})")
print(f"  Mean: {vc_final.mean():.1f}, Median: {vc_final.median():.1f}")
print(f"  Imbalance ratio: {vc_final.max()/vc_final.min():.1f}x")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: DIALOGUE INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
section("TASK 2: DIALOGUE INTENT — Restructuring")

# Load medquad with intent labels (primary source)
medquad_path = os.path.join(DATA, "raw", "dialogue", "medquad_clinical_qa.csv")
df_medquad = pd.read_csv(medquad_path)
print(f"MedQuAD loaded: {df_medquad.shape}")
print(f"Columns: {list(df_medquad.columns)}")
print(f"Intent distribution:\n{df_medquad['intent'].value_counts().to_string()}")

# Load existing intent labels
intent_labels_path = os.path.join(BASE, "backend", "models", "dialogue_intent_labels.json")
with open(intent_labels_path) as f:
    existing_intents = json.load(f)
print(f"\nExisting intent labels: {existing_intents}")

# Load grok dialogue (has intent column)
grok_dial_path = os.path.join(DATA, "raw", "grok", "triage_dialogue_reasoning.csv")
df_grok = pd.read_csv(grok_dial_path)
print(f"\nGrok dialogue loaded: {df_grok.shape}")
print(f"Grok intent distribution:\n{df_grok['intent'].value_counts().to_string()}")

# ── Map intents to existing label set ─────────────────────────────────────
# MedQuAD intents: information, symptoms, treatment, diagnosis, prevention, 
#                  causes, research, outlook, susceptibility, exams and tests, 
#                  inheritance, complications, support, genetic changes, stages
# Map to existing intents
INTENT_MAP = {
    "information": "information",
    "symptoms": "symptoms",
    "treatment": "treatment",
    "diagnosis": "diagnosis",
    "prevention": "prevention",
    "causes": "causes",
    "research": "information",
    "outlook": "prognosis",
    "susceptibility": "risk_factors",
    "exams and tests": "diagnosis",
    "inheritance": "information",
    "complications": "complications",
    "support": "information",
    "genetic changes": "information",
    "stages": "information",
    "clinical_qa": "information",
    "triage": "triage",
    "emergency": "emergency",
    "greeting": "greeting",
    "farewell": "farewell",
    "clarification": "clarification",
    "medication": "treatment",
    "referral": "referral",
}

# Prepare medquad
df_medquad_clean = df_medquad[["user_text", "intent"]].copy()
df_medquad_clean.columns = ["text", "intent"]
df_medquad_clean["intent"] = df_medquad_clean["intent"].str.strip().str.lower()
df_medquad_clean["intent"] = df_medquad_clean["intent"].map(INTENT_MAP).fillna("information")

# Prepare grok dialogue
df_grok_clean = df_grok[["user_text", "intent"]].copy()
df_grok_clean.columns = ["text", "intent"]
df_grok_clean["intent"] = df_grok_clean["intent"].str.strip().str.lower()
df_grok_clean["intent"] = df_grok_clean["intent"].map(INTENT_MAP).fillna("information")

# Load ULTIMATE_CONVERSATIONAL_QA — assign intents via keyword mapping
qa_path = os.path.join(DATA, "unified", "ULTIMATE_CONVERSATIONAL_QA.csv")
df_qa = pd.read_csv(qa_path)
df_qa = df_qa.drop_duplicates(subset=["question"])
print(f"\nULTIMATE_CONVERSATIONAL_QA (deduped): {df_qa.shape}")

def assign_intent_from_question(q):
    q_lower = str(q).lower()
    if any(w in q_lower for w in ["what is", "what are", "define", "describe", "overview"]):
        return "information"
    if any(w in q_lower for w in ["symptom", "sign", "feel", "experience"]):
        return "symptoms"
    if any(w in q_lower for w in ["treat", "therapy", "medication", "drug", "cure", "manage"]):
        return "treatment"
    if any(w in q_lower for w in ["diagnos", "test", "exam", "detect", "screen"]):
        return "diagnosis"
    if any(w in q_lower for w in ["prevent", "avoid", "risk reduc"]):
        return "prevention"
    if any(w in q_lower for w in ["cause", "why", "reason", "factor"]):
        return "causes"
    if any(w in q_lower for w in ["outlook", "prognosis", "survival", "recover"]):
        return "prognosis"
    if any(w in q_lower for w in ["complicat", "side effect", "danger"]):
        return "complications"
    if any(w in q_lower for w in ["who", "suscept", "risk", "prone"]):
        return "risk_factors"
    return "information"

df_qa_clean = df_qa[["question"]].copy()
df_qa_clean.columns = ["text"]
df_qa_clean["intent"] = df_qa_clean["text"].apply(assign_intent_from_question)

# Combine all dialogue sources
df_dialogue_all = pd.concat([
    df_medquad_clean,
    df_grok_clean,
    df_qa_clean,
], ignore_index=True)

df_dialogue_all = df_dialogue_all.dropna(subset=["text", "intent"])
df_dialogue_all = df_dialogue_all[df_dialogue_all["text"].str.strip().str.len() > 5]
df_dialogue_all = df_dialogue_all.drop_duplicates(subset=["text"])

print(f"\nCombined dialogue dataset: {df_dialogue_all.shape}")
print(f"Intent distribution:\n{df_dialogue_all['intent'].value_counts().to_string()}")

# Encode intents
le_intent = LabelEncoder()
df_dialogue_all["label_id"] = le_intent.fit_transform(df_dialogue_all["intent"])
intent_names = le_intent.classes_.tolist()
print(f"\nFinal intent classes ({len(intent_names)}): {intent_names}")

# Train/Val/Test split
X_d = df_dialogue_all["text"].values
y_d = df_dialogue_all["label_id"].values

X_d_train, X_d_temp, y_d_train, y_d_temp = train_test_split(
    X_d, y_d, test_size=0.30, random_state=42, stratify=y_d
)
X_d_val, X_d_test, y_d_val, y_d_test = train_test_split(
    X_d_temp, y_d_temp, test_size=0.50, random_state=42, stratify=y_d_temp
)

print(f"\nDialogue split sizes:")
print(f"  Train: {len(X_d_train):,}")
print(f"  Val:   {len(X_d_val):,}")
print(f"  Test:  {len(X_d_test):,}")

# Save dialogue splits
dial_dir = os.path.join(OUT_DIR, "dialogue")
df_d_train = pd.DataFrame({"text": X_d_train, "intent": le_intent.inverse_transform(y_d_train), "label_id": y_d_train})
df_d_val   = pd.DataFrame({"text": X_d_val,   "intent": le_intent.inverse_transform(y_d_val),   "label_id": y_d_val})
df_d_test  = pd.DataFrame({"text": X_d_test,  "intent": le_intent.inverse_transform(y_d_test),  "label_id": y_d_test})

df_d_train.to_csv(os.path.join(dial_dir, "train.csv"), index=False)
df_d_val.to_csv(os.path.join(dial_dir, "val.csv"), index=False)
df_d_test.to_csv(os.path.join(dial_dir, "test.csv"), index=False)
df_d_train.to_parquet(os.path.join(dial_dir, "train.parquet"), index=False)
df_d_val.to_parquet(os.path.join(dial_dir, "val.parquet"), index=False)
df_d_test.to_parquet(os.path.join(dial_dir, "test.parquet"), index=False)

with open(os.path.join(dial_dir, "intent_names.json"), "w") as f:
    json.dump(intent_names, f, indent=2)

print(f"Dialogue dataset saved to: {dial_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: IMAGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
section("TASK 3: IMAGE — Restructuring")

# Load dermamnist_64.npz (64px — better resolution for transfer learning)
npz_path = os.path.join(DATA, "raw", "imaging_legacy", "dermamnist_64.npz")
data = np.load(npz_path)

train_images = data["train_images"]  # (7007, 64, 64, 3)
train_labels = data["train_labels"].flatten()
val_images   = data["val_images"]    # (1003, 64, 64, 3)
val_labels   = data["val_labels"].flatten()
test_images  = data["test_images"]   # (2005, 64, 64, 3)
test_labels  = data["test_labels"].flatten()

print(f"Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}")

# Class names (DermaMNIST standard)
derm_class_names = [
    "actinic keratoses and intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions"
]

# Compute class weights for imbalanced training
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(7),
    y=train_labels
)
print(f"\nClass weights (balanced):")
for i, (name, w) in enumerate(zip(derm_class_names, class_weights)):
    count = (train_labels == i).sum()
    print(f"  {i}: {name} — {count} samples — weight: {w:.3f}")

# Normalize images to [0, 1]
train_images_norm = train_images.astype(np.float32) / 255.0
val_images_norm   = val_images.astype(np.float32) / 255.0
test_images_norm  = test_images.astype(np.float32) / 255.0

# Compute per-channel mean and std from training set
mean = train_images_norm.mean(axis=(0, 1, 2))
std  = train_images_norm.std(axis=(0, 1, 2))
print(f"\nTraining set statistics:")
print(f"  Mean (per channel): {mean.tolist()}")
print(f"  Std  (per channel): {std.tolist()}")

# Save imaging metadata
img_dir = os.path.join(OUT_DIR, "imaging")
img_meta = {
    "class_names": derm_class_names,
    "num_classes": 7,
    "train_samples": int(len(train_labels)),
    "val_samples": int(len(val_labels)),
    "test_samples": int(len(test_labels)),
    "image_size": 64,
    "channels": 3,
    "class_weights": class_weights.tolist(),
    "normalization": {
        "mean": mean.tolist(),
        "std": std.tolist()
    },
    "class_distribution": {
        "train": {derm_class_names[i]: int((train_labels == i).sum()) for i in range(7)},
        "val":   {derm_class_names[i]: int((val_labels == i).sum()) for i in range(7)},
        "test":  {derm_class_names[i]: int((test_labels == i).sum()) for i in range(7)},
    }
}

with open(os.path.join(img_dir, "imaging_metadata.json"), "w") as f:
    json.dump(img_meta, f, indent=2)

# Save normalized arrays
np.save(os.path.join(img_dir, "train_images.npy"), train_images_norm)
np.save(os.path.join(img_dir, "train_labels.npy"), train_labels)
np.save(os.path.join(img_dir, "val_images.npy"), val_images_norm)
np.save(os.path.join(img_dir, "val_labels.npy"), val_labels)
np.save(os.path.join(img_dir, "test_images.npy"), test_images_norm)
np.save(os.path.join(img_dir, "test_labels.npy"), test_labels)

print(f"\nImaging dataset saved to: {img_dir}")
print(f"  train/val/test images + labels as .npy")
print(f"  imaging_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 COMPLETE — DATASET RESTRUCTURING SUMMARY")

print(f"""
TASK 1 — TRIAGE TEXT:
  Input:  45,763 rows (unified) + 28,169 rows (processed)
  Output: {len(df_final):,} rows, {df_final['condition'].nunique()} classes
  Split:  {len(X_train):,} train / {len(X_val):,} val / {len(X_test):,} test
  Format: CSV + Parquet
  Path:   {triage_dir}

TASK 2 — DIALOGUE INTENT:
  Input:  16,165 (medquad) + 300 (grok) + {len(df_qa):,} (unified QA)
  Output: {len(df_dialogue_all):,} rows, {len(intent_names)} intent classes
  Split:  {len(X_d_train):,} train / {len(X_d_val):,} val / {len(X_d_test):,} test
  Format: CSV + Parquet
  Path:   {dial_dir}

TASK 3 — IMAGE:
  Input:  dermamnist_64.npz (7,007 train / 1,003 val / 2,005 test)
  Output: Normalized .npy arrays + metadata JSON
  Classes: 7 (DermaMNIST)
  Path:   {img_dir}
""")
