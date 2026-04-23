import subprocess
import os
from pathlib import Path
import time
import sys

def orchestrate():
    root = Path(".")
    scripts = root / "backend" / "scripts"
    unified_dir = root / "data" / "unified"
    
    # 1. GENERATE UNIFIED DATA (if not already running)
    print("Step 1: Ensuring Unified Datasets are generated...")
    # We assume the background process f8393c or 90711a is running, 
    # but we will run a clean synchronous version to be sure.
    subprocess.run([sys.executable, str(scripts / "ultimate_dataset_refactor.py")], check=True)

    # 2. DEFINE MASTER PATHS
    triage_path = unified_dir / "ULTIMATE_TRIAGE_KNOWLEDGE.csv"
    dialogue_path = unified_dir / "ULTIMATE_CONVERSATIONAL_QA.csv"
    
    if not triage_path.exists():
        print("ERROR: Ultimate Triage Knowledge not found. Check refactor logs.")
        return

    # 3. TRAIN TEXT MODELS (TRIAGE)
    print("\nStep 2: Starting Triage Model Training (XGBoost Pipeline)...")
    triage_cmd = [
        sys.executable,
        str(scripts / "preprocess_and_train.py"),
        "--input", str(triage_path),
        "--model-type", "xgboost",
        "--min-samples", "3",
        "--rebalance-mode", "smote"
    ]
    subprocess.run(triage_cmd, check=True)

    # 4. TRAIN DIALOGUE MODELS
    print("\nStep 3: Starting Dialogue Intent Training...")
    dialogue_cmd = [
        sys.executable,
        str(scripts / "preprocess_and_train.py"),
        "--input", str(dialogue_path),
        "--train-dialogue",
        "--dialogue-dataset", str(dialogue_path)
    ]
    # Note: Using unified dialogue path for both building and training
    subprocess.run(dialogue_cmd, check=True)

    print("\n✅ TRAINING COMPLETE. All models updated with production unified data.")

if __name__ == "__main__":
    orchestrate()
