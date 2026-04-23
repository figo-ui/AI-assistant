import pandas as pd
from pathlib import Path
import os
import json
import re

# patterns adapted from MIT-LCP/mimic-code
NOTE_CLEAN_RE = re.compile(r'\[\s*\*\s*.*?\s*\*\s*\]') # Removes MIMIC de-identification stubs like [** Name **]

def preprocess_mimic_notes(text: str) -> str:
    """Cleans raw clinical notes using MIT-LCP standards."""
    if not isinstance(text, str): return ""
    text = NOTE_CLEAN_RE.sub(" ", text) # Remove stubs
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def build_mimic_training_pipeline():
    """
    Implements the bridge between the MIT-LCP/mimic-code structure and our Chatbot.
    Expects data in data/dataset_hub/datasets/EHR/mimic-iv/raw/
    """
    base_path = Path("data/dataset_hub/datasets/EHR/mimic-iv/raw")
    output_path = Path("data/mimic_integrated_training_set.csv")
    
    # Required files from MIMIC-IV distribution
    notes_file = base_path / "note" / "discharge.csv"
    diagnoses_file = base_path / "hosp" / "diagnoses_icd.csv"
    dict_file = base_path / "hosp" / "d_icd_diagnoses.csv"
    
    if not notes_file.exists() or not diagnoses_file.exists():
        print(f"ERROR: MIMIC-IV files not found in {base_path}")
        print("Please download discharge.csv and diagnoses_icd.csv from PhysioNet first.")
        return

    print("Loading MIMIC-IV Clinical Notes (this may take time)...")
    notes = pd.read_csv(notes_file, usecols=["subject_id", "hadm_id", "text"])
    
    print("Loading Diagnoses and Dictionary...")
    diag = pd.read_csv(diagnoses_file, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    d_diag = pd.read_csv(dict_file, usecols=["icd_code", "icd_version", "long_title"])
    
    # Join logic from mimic-code patterns
    print("Joining concepts...")
    merged = diag.merge(d_diag, on=["icd_code", "icd_version"], how="left")
    
    # We take primary diagnoses (usually seq_num=1 in MIMIC)
    # For simplicity in this adapter, we take the top diagnosis per admission
    notes_with_diag = notes.merge(merged, on=["subject_id", "hadm_id"], how="inner")
    
    print("Pre-processing text notes...")
    notes_with_diag["symptom_text"] = notes_with_diag["text"].apply(preprocess_mimic_notes)
    notes_with_diag = notes_with_diag.rename(columns={"long_title": "condition"})
    
    # Final cleanup for our model format
    final_set = notes_with_diag[["symptom_text", "condition"]].dropna()
    
    print(f"MIMIC pipeline complete. Exporting {len(final_set)} records to {output_path}")
    final_set.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    build_mimic_training_pipeline()
