import shutil
import os
from pathlib import Path

def aggressive_cleanup():
    root = Path("data")
    raw_dir = root / "raw"
    
    # 1. MOVES
    moves = {
        "dataset_hub": raw_dir / "hub",
        "dialogue": raw_dir / "dialogue_legacy",
        "grok_medical_corpus_smoke": raw_dir / "grok",
        "image_datasets": raw_dir / "imaging_legacy",
        "kaggle": raw_dir / "kaggle",
        "open_datasets": raw_dir / "open",
        "physionet": raw_dir / "physionet"
    }
    
    for old_name, new_path in moves.items():
        old_path = root / old_name
        if old_path.exists():
            print(f"Moving {old_name} to {new_path}")
            if new_path.exists():
                # Merge if exists
                for item in old_path.iterdir():
                    shutil.move(str(item), str(new_path / item.name))
                old_path.rmdir()
            else:
                shutil.move(str(old_path), str(new_path))

    # 2. Cleanup leftover files in root of data
    for f in root.iterdir():
        if f.is_file() and f.name not in ["DATASET_CARD.md"]:
            if f.suffix == ".json":
                print(f"Archiving metadata file: {f.name}")
                shutil.move(str(f), str(root / "meta" / f.name))
            elif f.suffix == ".csv":
                print(f"Archiving processed file: {f.name}")
                shutil.move(str(f), str(root / "processed" / f.name))

    print("\nAGGRESSIVE CLEANUP COMPLETE.")

if __name__ == "__main__":
    aggressive_cleanup()
