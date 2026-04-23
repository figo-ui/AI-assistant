"""
TIMO — Step 2: Fitzpatrick17k Image Download + Manifest Builder
================================================================
DIAGNOSIS NOTE:
  - www.dermaamin.com (12,631 rows, 76% of dataset) returns 404 for ALL images.
    The site has taken down its clinical image gallery.
  - atlasdermatologico.com.br (3,905 rows, 24%) has 98% success rate.

DECISION:
  - Use atlas-only subset: ~3,826 downloadable images, 92 unique labels.
  - All 92 labels have ≥ 10 samples in the atlas subset.
  - This is still a significant improvement over DermaMNIST (7 classes, 28×28).
  - If test macro-F1 < 0.55 after training, Step 6 will integrate HAM10000/ISIC.

Pipeline:
  1. Filter bad QC + null URLs + dermaamin domain
  2. Download atlas images concurrently (8 workers)
  3. Stratified train(70%)/val(15%)/test(15%) split
  4. Write manifest.jsonl
  5. Save processed CSV
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedShuffleSplit

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path(r"C:\Users\hp\Desktop\AI assistant")
CSV_PATH = WORKSPACE / "data/raw/hub/datasets/Imaging/New folder/fitzpatrick17k (1).csv"
IMAGE_DIR = WORKSPACE / "data/raw/hub/datasets/Imaging/fitzpatrick17k_images"
PROCESSED_CSV = WORKSPACE / "data/processed/fitzpatrick17k_processed_v1.csv"

BACKEND_DIR = WORKSPACE / "backend"
MANIFEST_DIR = BACKEND_DIR / "data/image_dataset_combined"
MANIFEST_PATH = MANIFEST_DIR / "manifest.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_SAMPLES_PER_CLASS = 10
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
MAX_WORKERS = 8
REQUEST_TIMEOUT = 20
MAX_RETRIES = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("timo.step2")

BAD_QC = {"3 Wrongly labelled", "4 Other"}
ACCESSIBLE_DOMAINS = {"atlasdermatologico.com.br"}


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Raw shape: {df.shape}")

    # Exclude bad QC
    before = len(df)
    df = df[~df["qc"].isin(BAD_QC)]
    log.info(f"After bad-QC filter: {len(df)} rows (removed {before - len(df)})")

    # Exclude null URLs
    before = len(df)
    df = df[df["url"].notna() & (df["url"].str.strip() != "")]
    log.info(f"After null-URL filter: {len(df)} rows (removed {before - len(df)})")

    # Filter to accessible domains only
    df["domain"] = df["url"].apply(lambda u: urlparse(str(u)).netloc)
    before = len(df)
    df = df[df["domain"].isin(ACCESSIBLE_DOMAINS)]
    log.info(
        f"After domain filter (accessible only): {len(df)} rows "
        f"(removed {before - len(df)} dermaamin.com rows — site returns 404 for all images)"
    )

    # Exclude classes with < MIN_SAMPLES_PER_CLASS
    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
    before = len(df)
    df = df[df["label"].isin(valid_labels)]
    log.info(
        f"After min-{MIN_SAMPLES_PER_CLASS}-samples filter: {len(df)} rows, "
        f"{df['label'].nunique()} classes (removed {before - len(df)} rows)"
    )

    df = df.reset_index(drop=True)
    log.info(f"Final filtered dataset: {len(df)} rows, {df['label'].nunique()} classes")
    return df


def _ext_from_url(url: str) -> str:
    url_lower = url.lower().split("?")[0]
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        if url_lower.endswith(ext):
            return ext
    return ".jpg"


def _dest_path(row: pd.Series, image_dir: Path) -> Path:
    ext = _ext_from_url(str(row["url"]))
    return image_dir / f"{row['md5hash']}{ext}"


def _download_one(row: pd.Series, image_dir: Path) -> Tuple[str, Optional[Path], str]:
    dest = _dest_path(row, image_dir)
    if dest.exists() and dest.stat().st_size > 500:
        return row["md5hash"], dest, "exists"

    url = str(row["url"]).strip()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, stream=True)
            if resp.status_code == 200:
                content = resp.content
                dest.write_bytes(content)
                if dest.stat().st_size < 500:
                    dest.unlink(missing_ok=True)
                    return row["md5hash"], None, "skip_too_small"
                return row["md5hash"], dest, "downloaded"
            elif resp.status_code in {404, 410, 403, 401}:
                return row["md5hash"], None, f"skip_http_{resp.status_code}"
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                return row["md5hash"], None, f"skip_http_{resp.status_code}"
        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return row["md5hash"], None, f"skip_exception_{type(exc).__name__}"

    return row["md5hash"], None, "skip_max_retries"


def download_images(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    image_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {len(df)} images → {image_dir}")
    log.info(f"Workers: {MAX_WORKERS}, timeout: {REQUEST_TIMEOUT}s, retries: {MAX_RETRIES}")

    results: Dict[str, Optional[Path]] = {}
    statuses: Dict[str, str] = {}

    rows_list = [df.iloc[i] for i in range(len(df))]
    done = 0
    skipped = 0
    existed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, row, image_dir): row["md5hash"] for row in rows_list}
        for future in as_completed(futures):
            md5, path, status = future.result()
            results[md5] = path
            statuses[md5] = status
            if status == "downloaded":
                done += 1
            elif status == "exists":
                existed += 1
            else:
                skipped += 1
            total_processed = done + existed + skipped
            if total_processed % 200 == 0:
                log.info(
                    f"  Progress: {total_processed}/{len(df)} | "
                    f"downloaded={done} existed={existed} skipped={skipped}"
                )

    log.info(f"Download complete: downloaded={done}, existed={existed}, skipped={skipped}")

    df = df.copy()
    df["local_path"] = df["md5hash"].map(lambda h: str(results.get(h)) if results.get(h) else None)
    df["download_status"] = df["md5hash"].map(statuses)

    before = len(df)
    df = df[df["local_path"].notna()].reset_index(drop=True)
    log.info(f"Rows with valid local image: {len(df)} (dropped {before - len(df)} failed downloads)")

    return df


def build_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building stratified train/val/test split...")
    df = df.copy()
    df["split"] = "train"

    # First split off test (15%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
    train_val_idx, test_idx = next(sss1.split(df, df["label"]))
    df.loc[df.index[test_idx], "split"] = "test"

    # Then split val from train_val (~17.6% of train_val = 15% of total)
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_val_df = df[df["split"] == "train"].copy()
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
    try:
        train_idx2, val_idx2 = next(sss2.split(train_val_df, train_val_df["label"]))
        val_global_idx = train_val_df.index[val_idx2]
        df.loc[val_global_idx, "split"] = "val"
    except ValueError as e:
        log.warning(f"Stratified val split failed ({e}), using random split")
        val_mask = train_val_df.sample(frac=val_fraction, random_state=SEED).index
        df.loc[val_mask, "split"] = "val"

    split_counts = df["split"].value_counts()
    log.info(f"Split counts: {split_counts.to_dict()}")
    return df


def write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing manifest → {manifest_path}")
    with manifest_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "image_path": str(row["local_path"]),
                "label": str(row["label"]),
                "split": str(row["split"]),
                "source": "fitzpatrick17k",
            }
            f.write(json.dumps(record) + "\n")
    log.info(f"Manifest written: {len(df)} rows")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    log.info("=" * 60)
    log.info("TIMO STEP 2 — Fitzpatrick17k Download & Manifest")
    log.info("=" * 60)
    log.info("NOTE: dermaamin.com (76% of dataset) returns 404 for all images.")
    log.info("Using atlasdermatologico.com.br subset only (3,905 rows, 98% success rate).")
    log.info("=" * 60)

    # Step 2a: Load & filter
    df = load_and_filter(CSV_PATH)

    # Step 2b: Download images
    df = download_images(df, IMAGE_DIR)

    # Re-check class counts after download failures
    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
    before = len(df)
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    log.info(
        f"After post-download class filter: {len(df)} rows, "
        f"{df['label'].nunique()} classes (removed {before - len(df)})"
    )

    # Step 2c: Stratified split
    df = build_stratified_split(df)

    # Step 2d: Save processed CSV
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    log.info(f"Processed CSV saved → {PROCESSED_CSV}")

    # Step 2e: Write manifest
    write_manifest(df, MANIFEST_PATH)

    # Summary
    log.info("=" * 60)
    log.info("STEP 2 COMPLETE — SUMMARY")
    log.info(f"  Total rows in manifest: {len(df)}")
    log.info(f"  Classes: {df['label'].nunique()}")
    log.info(f"  Train: {(df['split']=='train').sum()}")
    log.info(f"  Val:   {(df['split']=='val').sum()}")
    log.info(f"  Test:  {(df['split']=='test').sum()}")
    log.info(f"  Manifest: {MANIFEST_PATH}")
    log.info(f"  Processed CSV: {PROCESSED_CSV}")
    log.info("=" * 60)

    print("\n=== CLASS DISTRIBUTION IN MANIFEST ===")
    print(df["label"].value_counts().to_string())
    print("\n=== SPLIT DISTRIBUTION ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).T.describe())


if __name__ == "__main__":
    main()
