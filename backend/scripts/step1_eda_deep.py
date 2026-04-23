"""
STEP 1 (continued): Deep EDA on triage dataset — understand the 1,102 conditions
and determine the best filtering strategy to get to ~88-113 clinically relevant classes.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")

DIVIDER = "=" * 70

def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

# ─── Load triage dataset ───────────────────────────────────────────────────
section("DEEP EDA: ULTIMATE_TRIAGE_KNOWLEDGE")
triage_path = os.path.join(DATA, "unified", "ULTIMATE_TRIAGE_KNOWLEDGE.csv")
df = pd.read_csv(triage_path)
print(f"Shape: {df.shape}")

# Text length analysis
df["text_len"] = df["symptom_text"].astype(str).str.len()
df["word_count"] = df["symptom_text"].astype(str).str.split().str.len()
print(f"\nText length stats:")
print(df[["text_len", "word_count"]].describe().to_string())

# Condition distribution
vc = df["condition"].value_counts()
print(f"\nTotal unique conditions: {len(vc)}")
print(f"Conditions with >= 100 samples: {(vc >= 100).sum()}")
print(f"Conditions with >= 50 samples:  {(vc >= 50).sum()}")
print(f"Conditions with >= 20 samples:  {(vc >= 20).sum()}")
print(f"Conditions with >= 10 samples:  {(vc >= 10).sum()}")
print(f"Conditions with >= 5 samples:   {(vc >= 5).sum()}")
print(f"Conditions with < 5 samples:    {(vc < 5).sum()}")

# Top 30 conditions
print(f"\nTop 30 conditions:")
print(vc.head(30).to_string())

# Bottom 20 conditions
print(f"\nBottom 20 conditions (rarest):")
print(vc.tail(20).to_string())

# Check for MIMIC-style clinical codes vs clean condition names
print(f"\nSample conditions (first 50 unique):")
for c in vc.index[:50]:
    print(f"  {c}: {vc[c]}")

# Check text quality — look for clinical codes vs symptom text
print(f"\nSample symptom texts (first 10):")
for t in df["symptom_text"].head(10).tolist():
    print(f"  > {str(t)[:150]}")

# Check for MIMIC-style text (procedure codes, lab results)
mimic_indicators = ["r/o", "negative by eia", "furosemide", "vancomycin", "specimen", "culture"]
mimic_mask = df["symptom_text"].astype(str).str.lower().str.contains("|".join(mimic_indicators), na=False)
print(f"\nRows with MIMIC-style clinical text: {mimic_mask.sum():,} ({mimic_mask.mean()*100:.1f}%)")

# Check for clean symptom text
symptom_indicators = ["pain", "fever", "cough", "headache", "nausea", "vomiting", "fatigue", "dizziness"]
symptom_mask = df["symptom_text"].astype(str).str.lower().str.contains("|".join(symptom_indicators), na=False)
print(f"Rows with symptom-style text: {symptom_mask.sum():,} ({symptom_mask.mean()*100:.1f}%)")

# ─── Load existing text labels to understand target classes ───────────────
section("EXISTING TEXT LABELS (baseline model)")
labels_path = os.path.join(BASE, "backend", "models", "text_labels.json")
if os.path.exists(labels_path):
    with open(labels_path) as f:
        existing_labels = json.load(f)
    print(f"Existing model classes: {len(existing_labels)}")
    print(f"Sample labels: {existing_labels[:20]}")
    
    # How many of these exist in the unified dataset?
    unified_conditions = set(df["condition"].str.strip().str.lower().tolist())
    existing_lower = [l.strip().lower() for l in existing_labels]
    overlap = sum(1 for l in existing_lower if l in unified_conditions)
    print(f"\nOverlap with unified dataset: {overlap}/{len(existing_labels)} ({overlap/len(existing_labels)*100:.1f}%)")

# ─── Load processed dataset to compare ────────────────────────────────────
section("PROCESSED DATASET COMPARISON")
proc_path = os.path.join(DATA, "processed", "expanded_symptom_condition_processed_min5_rebalanced.csv")
df_proc = pd.read_csv(proc_path)
print(f"Processed dataset shape: {df_proc.shape}")
print(f"Unique conditions: {df_proc['condition'].nunique()}")
vc_proc = df_proc["condition"].value_counts()
print(f"Top 10 conditions in processed:")
print(vc_proc.head(10).to_string())

# ─── QA Dataset deep analysis ─────────────────────────────────────────────
section("DEEP EDA: ULTIMATE_CONVERSATIONAL_QA")
qa_path = os.path.join(DATA, "unified", "ULTIMATE_CONVERSATIONAL_QA.csv")
df_qa = pd.read_csv(qa_path)
print(f"Shape: {df_qa.shape}")

df_qa["q_len"] = df_qa["question"].astype(str).str.len()
df_qa["a_len"] = df_qa["answer"].astype(str).str.len()
print(f"\nQuestion length: mean={df_qa['q_len'].mean():.1f}, median={df_qa['q_len'].median():.1f}")
print(f"Answer length:   mean={df_qa['a_len'].mean():.1f}, median={df_qa['a_len'].median():.1f}")

# Check for duplicates in questions
q_dup = df_qa["question"].duplicated().sum()
print(f"\nDuplicate questions: {q_dup:,} ({q_dup/len(df_qa)*100:.2f}%)")

# Sample Q&A pairs
print(f"\nSample Q&A pairs:")
for _, row in df_qa.head(5).iterrows():
    print(f"  Q: {str(row['question'])[:100]}")
    print(f"  A: {str(row['answer'])[:100]}")
    print()

# ─── Imaging labels deep analysis ─────────────────────────────────────────
section("DEEP EDA: ULTIMATE_IMAGING_LABELS")
img_path = os.path.join(DATA, "unified", "ULTIMATE_IMAGING_LABELS.csv")
df_img = pd.read_csv(img_path)
print(f"Shape: {df_img.shape}")

vc_cond = df_img["condition"].value_counts()
vc_cat = df_img["category"].value_counts()
print(f"\nUnique conditions: {df_img['condition'].nunique()}")
print(f"Unique categories: {df_img['category'].nunique()}")
print(f"\nCategory distribution:")
print(vc_cat.to_string())
print(f"\nTop 20 conditions:")
print(vc_cond.head(20).to_string())
print(f"\nBottom 10 conditions:")
print(vc_cond.tail(10).to_string())
print(f"\nConditions with < 10 samples: {(vc_cond < 10).sum()}")
print(f"Conditions with < 5 samples:  {(vc_cond < 5).sum()}")

# ─── DermaMNIST class names ────────────────────────────────────────────────
section("DERMAMNIST CLASS NAMES")
img_labels_path = os.path.join(BASE, "backend", "models", "image_labels.json")
if os.path.exists(img_labels_path):
    with open(img_labels_path) as f:
        img_labels = json.load(f)
    print(f"Image labels: {img_labels}")

# DermaMNIST standard class names
derm_classes = {
    0: "actinic keratoses",
    1: "basal cell carcinoma",
    2: "benign keratosis-like lesions",
    3: "dermatofibroma",
    4: "melanoma",
    5: "melanocytic nevi",
    6: "vascular lesions"
}
print(f"\nDermaMNIST standard classes:")
npz = np.load(os.path.join(DATA, "raw", "imaging_legacy", "dermamnist.npz"))
train_labels = npz["train_labels"].flatten()
for cls_id, cls_name in derm_classes.items():
    count = (train_labels == cls_id).sum()
    print(f"  {cls_id}: {cls_name} — {count} train samples")

# Imbalance ratio
counts = [(train_labels == i).sum() for i in range(7)]
print(f"\nImbalance ratio: {max(counts)/min(counts):.1f}x (max/min)")
print(f"Class weights needed: {[round(max(counts)/c, 2) for c in counts]}")

section("DIAGNOSIS COMPLETE")
print("""
KEY FINDINGS:
=============

TASK 1 — TRIAGE TEXT CLASSIFICATION:
  - ULTIMATE_TRIAGE_KNOWLEDGE has 45,763 rows but 1,102 unique conditions
  - Many conditions are MIMIC-style clinical codes (DRG descriptions), not clean symptom→condition pairs
  - Need to filter to clinically relevant conditions with >= 50 samples
  - Conditions with >= 50 samples: see above
  - The processed dataset (28,169 rows, 113 classes) is cleaner but smaller
  - Strategy: filter unified dataset to conditions with >= 50 samples, 
    then merge with processed dataset for best coverage

TASK 2 — DIALOGUE INTENT CLASSIFICATION:
  - ULTIMATE_CONVERSATIONAL_QA has 91,269 Q&A pairs (question + answer)
  - This is a Q&A dataset, NOT an intent classification dataset
  - The existing model classifies 15-17 intents (information, symptoms, treatment, etc.)
  - Strategy: use medquad_clinical_qa (16,165 rows with intent labels) as primary
    and augment with ULTIMATE_CONVERSATIONAL_QA for richer training

TASK 3 — SKIN LESION IMAGE CLASSIFICATION:
  - DermaMNIST: 7,007 train / 1,003 val / 2,005 test, 7 classes
  - Severe class imbalance: class 5 (melanocytic nevi) = 4,693 vs class 3 (dermatofibroma) = 80
  - Imbalance ratio: ~58x
  - ULTIMATE_IMAGING_LABELS has metadata only (no pixel data)
  - Strategy: use dermamnist_64.npz (64px) with class weighting + augmentation
    and EfficientNet-B0 transfer learning
""")
