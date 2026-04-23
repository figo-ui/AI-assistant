"""
STEP 1: Dataset Ingestion & Diagnosis
Full EDA across all datasets for the healthcare AI pipeline.
"""
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")

DIVIDER = "=" * 70

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def subsection(title):
    print(f"\n--- {title} ---")

def analyze_csv(path, name, text_col=None, label_col=None, max_rows=None):
    section(f"EDA: {name}")
    try:
        df = pd.read_csv(path, nrows=max_rows)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None

    print(f"  Path       : {path}")
    print(f"  Shape      : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Dtypes     :\n{df.dtypes.to_string()}")

    subsection("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    mv = pd.DataFrame({"count": missing, "pct": missing_pct})
    mv = mv[mv["count"] > 0]
    if mv.empty:
        print("  No missing values.")
    else:
        print(mv.to_string())

    subsection("Duplicates")
    n_dup = df.duplicated().sum()
    print(f"  Exact duplicate rows: {n_dup:,} ({n_dup/len(df)*100:.2f}%)")

    subsection("Statistical Summary (numeric)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(df[num_cols].describe().to_string())
    else:
        print("  No numeric columns.")

    if text_col and text_col in df.columns:
        subsection(f"Text Column: '{text_col}'")
        lengths = df[text_col].dropna().astype(str).str.len()
        word_counts = df[text_col].dropna().astype(str).str.split().str.len()
        print(f"  Char length  — mean: {lengths.mean():.1f}, median: {lengths.median():.1f}, "
              f"min: {lengths.min()}, max: {lengths.max()}")
        print(f"  Word count   — mean: {word_counts.mean():.1f}, median: {word_counts.median():.1f}, "
              f"min: {word_counts.min()}, max: {word_counts.max()}")
        print(f"  Empty/null   : {df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == '').sum()}")
        print(f"  Sample texts :")
        for t in df[text_col].dropna().head(3).tolist():
            print(f"    > {str(t)[:120]}")

    if label_col and label_col in df.columns:
        subsection(f"Label Column: '{label_col}'")
        vc = df[label_col].value_counts()
        n_classes = len(vc)
        print(f"  Unique classes : {n_classes}")
        print(f"  Class balance  — min: {vc.min()}, max: {vc.max()}, "
              f"mean: {vc.mean():.1f}, std: {vc.std():.1f}")
        imbalance_ratio = vc.max() / max(vc.min(), 1)
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
        print(f"  Top-10 classes:")
        print(vc.head(10).to_string())
        print(f"  Bottom-10 classes:")
        print(vc.tail(10).to_string())
        # Classes with < 10 samples
        rare = vc[vc < 10]
        print(f"  Classes with <10 samples: {len(rare)}")
        rare50 = vc[vc < 50]
        print(f"  Classes with <50 samples: {len(rare50)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIFIED DATASETS
# ─────────────────────────────────────────────────────────────────────────────

section("UNIFIED DATASETS — PRIMARY TARGETS")

# 1a. ULTIMATE_TRIAGE_KNOWLEDGE
triage_path = os.path.join(DATA, "unified", "ULTIMATE_TRIAGE_KNOWLEDGE.csv")
df_triage = analyze_csv(triage_path, "ULTIMATE_TRIAGE_KNOWLEDGE (Triage/Text)",
                        text_col=None, label_col=None)

if df_triage is not None:
    subsection("Column inspection for triage dataset")
    for col in df_triage.columns:
        sample_vals = df_triage[col].dropna().head(3).tolist()
        print(f"  {col}: {sample_vals}")

    # Try to auto-detect text and label columns
    str_cols = df_triage.select_dtypes(include=["object"]).columns.tolist()
    print(f"\n  String columns: {str_cols}")
    for col in str_cols:
        vc = df_triage[col].value_counts()
        print(f"  '{col}' — unique: {df_triage[col].nunique()}, "
              f"top: {vc.index[0] if len(vc) > 0 else 'N/A'} ({vc.iloc[0] if len(vc) > 0 else 0})")

# 1b. ULTIMATE_CONVERSATIONAL_QA
qa_path = os.path.join(DATA, "unified", "ULTIMATE_CONVERSATIONAL_QA.csv")
df_qa = analyze_csv(qa_path, "ULTIMATE_CONVERSATIONAL_QA (Dialogue)",
                    text_col=None, label_col=None)

if df_qa is not None:
    subsection("Column inspection for QA dataset")
    for col in df_qa.columns:
        sample_vals = df_qa[col].dropna().head(2).tolist()
        print(f"  {col}: {[str(v)[:100] for v in sample_vals]}")

# 1c. ULTIMATE_IMAGING_LABELS
img_path = os.path.join(DATA, "unified", "ULTIMATE_IMAGING_LABELS.csv")
df_img = analyze_csv(img_path, "ULTIMATE_IMAGING_LABELS (Image)",
                     text_col=None, label_col=None)

if df_img is not None:
    subsection("Column inspection for imaging dataset")
    for col in df_img.columns:
        sample_vals = df_img[col].dropna().head(3).tolist()
        print(f"  {col}: {[str(v)[:80] for v in sample_vals]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROCESSED DATASETS
# ─────────────────────────────────────────────────────────────────────────────

section("PROCESSED DATASETS — SUPPLEMENTARY")

proc1 = os.path.join(DATA, "processed", "expanded_symptom_condition_processed_min5_rebalanced.csv")
df_proc1 = analyze_csv(proc1, "expanded_symptom_condition_processed_min5_rebalanced",
                       text_col=None, label_col=None)
if df_proc1 is not None:
    for col in df_proc1.columns[:5]:
        print(f"  {col}: {df_proc1[col].head(2).tolist()}")

proc2 = os.path.join(DATA, "processed", "integrated_important_symptom_condition_processed_min5.csv")
df_proc2 = analyze_csv(proc2, "integrated_important_symptom_condition_processed_min5",
                       text_col=None, label_col=None)
if df_proc2 is not None:
    for col in df_proc2.columns[:5]:
        print(f"  {col}: {df_proc2[col].head(2).tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RAW DIALOGUE / CLINICAL
# ─────────────────────────────────────────────────────────────────────────────

section("RAW DIALOGUE / CLINICAL DATASETS")

mimic_path = os.path.join(DATA, "raw", "clinical", "MIMIC_IV_Transcript.csv")
df_mimic = analyze_csv(mimic_path, "MIMIC_IV_Transcript", text_col=None, label_col=None)
if df_mimic is not None:
    for col in df_mimic.columns:
        print(f"  {col}: {df_mimic[col].head(2).tolist()}")

medquad_path = os.path.join(DATA, "raw", "dialogue", "medquad_clinical_qa.csv")
df_medquad = analyze_csv(medquad_path, "medquad_clinical_qa", text_col=None, label_col=None)
if df_medquad is not None:
    for col in df_medquad.columns:
        print(f"  {col}: {[str(v)[:80] for v in df_medquad[col].head(2).tolist()]}")

grok_sup = os.path.join(DATA, "raw", "grok", "triage_supervised.csv")
df_grok_sup = analyze_csv(grok_sup, "triage_supervised (grok)", text_col=None, label_col=None)
if df_grok_sup is not None:
    for col in df_grok_sup.columns:
        print(f"  {col}: {[str(v)[:80] for v in df_grok_sup[col].head(2).tolist()]}")

grok_dial = os.path.join(DATA, "raw", "grok", "triage_dialogue_reasoning.csv")
df_grok_dial = analyze_csv(grok_dial, "triage_dialogue_reasoning (grok)", text_col=None, label_col=None)
if df_grok_dial is not None:
    for col in df_grok_dial.columns:
        print(f"  {col}: {[str(v)[:80] for v in df_grok_dial[col].head(2).tolist()]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. IMAGING — NPZ FILES
# ─────────────────────────────────────────────────────────────────────────────

section("IMAGING DATASETS — NPZ")

for npz_name in ["dermamnist.npz", "dermamnist_64.npz"]:
    npz_path = os.path.join(DATA, "raw", "imaging_legacy", npz_name)
    print(f"\n  File: {npz_path}")
    try:
        data = np.load(npz_path)
        keys = list(data.keys())
        print(f"  Keys: {keys}")
        for k in keys:
            arr = data[k]
            print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")
        # Class distribution
        for split in ["train_labels", "val_labels", "test_labels"]:
            if split in data:
                labels = data[split].flatten()
                unique, counts = np.unique(labels, return_counts=True)
                print(f"  {split} distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXISTING BASELINES SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("EXISTING BASELINE MODELS SUMMARY")

print("""
  Task 1 — Triage Text Classification:
    Model     : TF-IDF + LogisticRegression (sklearn)
    Macro F1  : 0.7709 (reported), 0.7553 (model card)
    Classes   : 88 (reported) / 113 (model card)
    Samples   : 11,076 (reported)
    Artifacts : text_classifier.joblib, tfidf_vectorizer.joblib

  Task 2 — Dialogue Intent Classification:
    Model     : TF-IDF + sklearn classifier
    Accuracy  : 0.9994 (reported) / 0.9825 (metrics json)
    Macro F1  : 0.9763
    Classes   : 15-17 intent classes
    Samples   : 16,164 (reported) / 37,089 (metrics json)
    Artifacts : dialogue_intent_classifier.joblib

  Task 3 — Skin Lesion Image Classification:
    Model     : DermCNN (custom PyTorch CNN)
    Test Macro F1  : 0.307
    Test Accuracy  : 0.5022
    Classes   : 7 (DermaMNIST)
    Samples   : 7,007 train / 1,003 val / 2,005 test
    Artifacts : skin_cnn_torch.pt
""")

section("EDA COMPLETE — DIAGNOSIS SUMMARY WILL FOLLOW")
print("All datasets loaded and analyzed. See above for full statistics.")
