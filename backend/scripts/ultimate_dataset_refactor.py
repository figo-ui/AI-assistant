import pandas as pd
from pathlib import Path
import re
import os

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'\[\s*\*\s*.*?\s*\*\s*\]', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def ultimate_aggregator():
    root = Path("data")
    unified_dir = root / "unified"
    unified_dir.mkdir(exist_ok=True)
    
    triage_frames = []
    qa_frames = []
    imaging_frames = []
    
    # HAM10000 label mapping
    HAM_MAP = {
        'akiec': 'Actinic Keratosis',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratotic Lesion',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevus',
        'vasc': 'Vascular Lesion'
    }

    print("\n--- Deep Scan: Discovering all Medical Knowledge ---")
    targets = [root / "raw", root / "processed"]
    for target in targets:
        if not target.exists(): continue
        for f in target.rglob("*.csv"):
            print(f"Analyzing: {f.name}...")
            try:
                df = pd.read_csv(f, low_memory=False, nrows=500000)
                
                # BRANCH A: IMAGING DETECTION
                if "fitzpatrick_scale" in df.columns or "image_id" in df.columns or "dx" in df.columns:
                    print(f"  -> Detected Imaging Metadata in {f.name}")
                    temp = pd.DataFrame()
                    if "image_id" in df.columns: # HAM10000 style
                        temp["image_id"] = df["image_id"]
                        temp["condition"] = df["dx"].map(HAM_MAP).fillna(df["dx"])
                        temp["category"] = df.get("dx_type", "Unknown")
                    else: # Fitzpatrick style
                        temp["image_id"] = df.get("md5hash", df.get("url_alphanum", "unknown"))
                        temp["condition"] = df["label"]
                        temp["category"] = df.get("three_partition_label", "Unknown")
                    
                    temp["condition"] = temp["condition"].astype(str).str.strip().str.title()
                    imaging_frames.append(temp)

                # BRANCH B: SYMPTOM/TRIAGE DETECTION
                elif ("description" in df.columns and "drg_type" in df.columns) or \
                     ("symptom_text" in df.columns) or ("prognosis" in df.columns):
                    
                    print(f"  -> Detected Triage logic in {f.name}")
                    if "description" in df.columns and "drg_type" in df.columns:
                        cols = [c for c in ["drug", "test_name", "comments"] if c in df.columns]
                        temp = pd.DataFrame()
                        temp["symptom_text"] = df[cols].fillna("").agg(" ".join, axis=1)
                        temp["condition"] = df["description"]
                    else:
                        text_col = next((c for c in ["symptom_text", "text", "description"] if c in df.columns), None)
                        label_col = next((c for c in ["condition", "label", "prognosis"] if c in df.columns), None)
                        if text_col and label_col:
                            temp = df[[text_col, label_col]].rename(columns={text_col: "symptom_text", label_col: "condition"})
                    
                    temp["symptom_text"] = temp["symptom_text"].apply(clean_text)
                    temp["condition"] = temp["condition"].astype(str).str.strip().str.title()
                    triage_frames.append(temp)

                # BRANCH C: DIALOGUE/QA DETECTION
                elif ("user_text" in df.columns and "assistant_text" in df.columns) or \
                     ("question" in df.columns and "answer" in df.columns):
                    
                    print(f"  -> Detected Dialogue logic in {f.name}")
                    q_col = next((c for c in ["user_text", "question", "prompt"] if c in df.columns), None)
                    a_col = next((c for c in ["assistant_text", "answer", "response"] if c in df.columns), None)
                    temp = df[[q_col, a_col]].rename(columns={q_col: "question", a_col: "answer"})
                    qa_frames.append(temp)
            
            except Exception as e:
                print(f"  !! Error processing {f.name}: {e}")

    # FINAL EXPORT
    if triage_frames:
        pd.concat(triage_frames, ignore_index=True).drop_duplicates().dropna().to_csv(unified_dir / "ULTIMATE_TRIAGE_KNOWLEDGE.csv", index=False)
        print("SUCCESS: ULTIMATE TRIAGE KNOWLEDGE created.")

    if qa_frames:
        pd.concat(qa_frames, ignore_index=True).drop_duplicates().dropna().to_csv(unified_dir / "ULTIMATE_CONVERSATIONAL_QA.csv", index=False)
        print("SUCCESS: ULTIMATE CONVERSATIONAL QA created.")

    if imaging_frames:
        pd.concat(imaging_frames, ignore_index=True).drop_duplicates().dropna().to_csv(unified_dir / "ULTIMATE_IMAGING_LABELS.csv", index=False)
        print("SUCCESS: ULTIMATE IMAGING LABELS created.")

    print("\nALL DATA REFACTORED AND UNIFIED.")

if __name__ == "__main__":
    ultimate_aggregator()
