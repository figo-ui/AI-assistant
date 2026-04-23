import shutil
import os
from pathlib import Path

def reorganize():
    root = Path("data")
    
    # Define Target Folders
    raw_clinical = root / "raw" / "clinical"
    raw_dialogue = root / "raw" / "dialogue"
    meta_dir = root / "meta"
    proc_dir = root / "processed"
    
    # 1. MOVE MIMIC
    mimic_src = root / "dataset_hub" / "datasets" / "EHR" / "mimic-iv" / "raw" / "MIMIC_IV_Trasncript.csv"
    if mimic_src.exists():
        print(f"Moving MIMIC to {raw_clinical}")
        shutil.move(str(mimic_src), str(raw_clinical / "MIMIC_IV_Transcript.csv"))

    # 2. MOVE MedQuAD
    medquad_src = root / "dialogue" / "medquad_dialogue_pairs.csv"
    if medquad_src.exists():
        print(f"Moving MedQuAD to {raw_dialogue}")
        shutil.move(str(medquad_src), str(raw_dialogue / "medquad_clinical_qa.csv"))

    # 3. MOVE Metadata (JSON maps, summaries)
    for f in root.glob("*.json"):
        print(f"Moving Metadata: {f.name}")
        shutil.move(str(f), str(meta_dir / f.name))

    # 4. MOVE Triage CSVs to Processed
    triage_patterns = ["integrated_*.csv", "expanded_*.csv", "disease_symptom*.csv", "synthea*.csv"]
    for pattern in triage_patterns:
        for f in root.glob(pattern):
            print(f"Moving Triage Data: {f.name}")
            shutil.move(str(f), str(proc_dir / f.name))

    print("\nDATA FOLDER REORGANIZATION COMPLETE.")

if __name__ == "__main__":
    reorganize()
