import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_SPLITS = {"train", "val", "test"}

UNIFIED_LABEL_MAP = {
    "akiec": "actinic keratosis",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
    "scc": "squamous cell carcinoma",
    "seborrheic keratosis": "seborrheic keratosis",
    "eczema": "eczema",
    "contact dermatitis": "contact dermatitis",
    "atopic dermatitis": "atopic dermatitis",
    "psoriasis": "psoriasis",
    "acne": "acne",
    "rosacea": "rosacea",
    "tinea": "fungal infection",
    "impetigo": "impetigo",
    "urticaria": "urticaria",
    "cellulitis": "cellulitis",
    "chemical burn": "chemical burn",
    "lupus": "lupus rash",
    "vitiligo": "vitiligo",
}


@dataclass
class SourceConfig:
    source_id: str
    display_name: str
    raw_dir: str
    metadata_patterns: Tuple[str, ...]
    label_columns: Tuple[str, ...]
    image_columns: Tuple[str, ...]
    split_columns: Tuple[str, ...]
    source_label_map: Dict[str, str]
    hf_repo_id: Optional[str] = None


SOURCE_CONFIGS: Dict[str, SourceConfig] = {
    "ham10000": SourceConfig(
        source_id="ham10000",
        display_name="HAM10000",
        raw_dir="HAM10000",
        metadata_patterns=("**/*HAM10000_metadata*.csv", "**/*metadata*.csv"),
        label_columns=("dx", "label", "diagnosis"),
        image_columns=("image_id", "image", "image_name"),
        split_columns=("split", "subset"),
        source_label_map={
            "akiec": "actinic keratosis",
            "bcc": "basal cell carcinoma",
            "bkl": "benign keratosis",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "nv": "melanocytic nevus",
            "vasc": "vascular lesion",
        },
    ),
    "isic2020": SourceConfig(
        source_id="isic2020",
        display_name="ISIC 2020 / Archive",
        raw_dir="ISIC2020",
        metadata_patterns=("**/*train*.csv", "**/*metadata*.csv"),
        label_columns=("diagnosis", "label", "target", "dx"),
        image_columns=("image_name", "image", "image_id"),
        split_columns=("split", "subset"),
        source_label_map={
            "melanoma": "melanoma",
            "nevus": "melanocytic nevus",
            "basal cell carcinoma": "basal cell carcinoma",
            "actinic keratosis": "actinic keratosis",
            "benign keratosis": "benign keratosis",
            "vascular lesion": "vascular lesion",
            "squamous cell carcinoma": "squamous cell carcinoma",
        },
    ),
    "pad_ufes_20": SourceConfig(
        source_id="pad_ufes_20",
        display_name="PAD-UFES-20",
        raw_dir="PAD-UFES-20",
        metadata_patterns=("**/*metadata*.csv", "**/*.csv"),
        label_columns=("diagnostic", "diagnosis", "label", "lesion_type"),
        image_columns=("img_id", "image_id", "image", "image_name"),
        split_columns=("split", "subset"),
        source_label_map={
            "bcc": "basal cell carcinoma",
            "melanoma": "melanoma",
            "seborrheic keratosis": "seborrheic keratosis",
            "squamous cell carcinoma": "squamous cell carcinoma",
            "basal cell carcinoma": "basal cell carcinoma",
            "nevus": "melanocytic nevus",
            "psoriasis": "psoriasis",
            "eczema": "eczema",
            "contact dermatitis": "contact dermatitis",
            "chemical burn": "chemical burn",
        },
    ),
    "fitzpatrick17k": SourceConfig(
        source_id="fitzpatrick17k",
        display_name="Fitzpatrick17k",
        raw_dir="Fitzpatrick17k",
        metadata_patterns=("**/*metadata*.csv", "**/*.csv"),
        label_columns=("label", "nine_partition_label", "three_partition_label", "disease"),
        image_columns=("md5hash", "image", "image_id", "file_name"),
        split_columns=("split", "subset"),
        source_label_map={
            "eczema": "eczema",
            "psoriasis": "psoriasis",
            "acne": "acne",
            "vitiligo": "vitiligo",
            "rosacea": "rosacea",
            "contact dermatitis": "contact dermatitis",
            "fungal infection": "fungal infection",
            "urticaria": "urticaria",
            "cellulitis": "cellulitis",
            "impetigo": "impetigo",
            "lupus": "lupus rash",
        },
    ),
}


def parse_args() -> argparse.Namespace:
    backend_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Stage skin-image datasets and build a combined manifest."
    )
    parser.add_argument(
        "--raw-root",
        default=str(backend_dir / "data" / "image_sources"),
        help="Directory containing raw dataset folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(backend_dir / "data" / "image_dataset_combined"),
        help="Combined dataset output directory.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="hardlink",
        help="How to materialize images inside the combined dataset.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--enable-hf-download",
        action="store_true",
        help="Attempt Hugging Face snapshot downloads for configured source repos.",
    )
    return parser.parse_args()


def normalize_label(value: str, source_map: Dict[str, str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in source_map:
        return source_map[text]
    if text in UNIFIED_LABEL_MAP:
        return UNIFIED_LABEL_MAP[text]
    return None


def stable_split(key: str, *, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / float(16 ** 8)
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def find_first(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate.is_file():
                return candidate
    return None


def resolve_image_path(root: Path, image_name: str) -> Optional[Path]:
    candidate = Path(image_name)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix:
        probe = root / candidate
        if probe.exists():
            return probe.resolve()
    else:
        for suffix in IMAGE_EXTENSIONS:
            probe = root / f"{image_name}{suffix}"
            if probe.exists():
                return probe.resolve()
    patterns = [f"**/{candidate.name}"]
    if not candidate.suffix:
        patterns = [f"**/{image_name}{suffix}" for suffix in IMAGE_EXTENSIONS]
    for pattern in patterns:
        for match in root.glob(pattern):
            if match.is_file():
                return match.resolve()
    return None


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            os.link(str(source), str(destination))
            return
        except Exception:
            pass
    shutil.copy2(source, destination)


def pick_column(frame: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        for column in frame.columns:
            if str(column).strip().lower() == candidate.lower():
                return str(column)
    return None


def iter_records(root: Path, config: SourceConfig, val_ratio: float, test_ratio: float) -> List[Dict[str, str]]:
    metadata_path = find_first(root, config.metadata_patterns)
    if metadata_path is None:
        return []
    frame = pd.read_csv(metadata_path)
    if frame.empty:
        return []

    label_col = pick_column(frame, config.label_columns)
    image_col = pick_column(frame, config.image_columns)
    split_col = pick_column(frame, config.split_columns)
    if label_col is None or image_col is None:
        return []

    records: List[Dict[str, str]] = []
    for _, row in frame.iterrows():
        unified_label = normalize_label(str(row.get(label_col, "")), config.source_label_map)
        if unified_label is None:
            continue
        raw_image_key = str(row.get(image_col, "")).strip()
        if not raw_image_key:
            continue
        image_path = resolve_image_path(root, raw_image_key)
        if image_path is None:
            continue
        raw_split = str(row.get(split_col, "")).strip().lower() if split_col else ""
        split = raw_split if raw_split in VALID_SPLITS else stable_split(raw_image_key, val_ratio=val_ratio, test_ratio=test_ratio)
        records.append(
            {
                "image_path": str(image_path),
                "label": unified_label,
                "split": split,
                "source": config.source_id,
            }
        )
    return records


def stage_hf_sources(raw_root: Path) -> None:
    if snapshot_download is None:
        return
    for config in SOURCE_CONFIGS.values():
        if not config.hf_repo_id:
            continue
        target_dir = raw_root / config.raw_dir
        if target_dir.exists():
            continue
        snapshot_download(
            repo_id=config.hf_repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )


def write_combined_dataset(records: List[Dict[str, str]], output_root: Path, copy_mode: str) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    images_root = output_root / "images"
    manifest_path = output_root / "manifest.jsonl"

    split_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            source_path = Path(record["image_path"])
            ext = source_path.suffix.lower() or ".jpg"
            target = images_root / record["split"] / record["label"] / f"{record['source']}_{index}{ext}"
            copy_or_link(source_path, target, copy_mode)
            payload = {**record, "image_path": str(target.resolve())}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            split_counts[payload["split"]] = split_counts.get(payload["split"], 0) + 1
            class_counts[payload["label"]] = class_counts.get(payload["label"], 0) + 1
            source_counts[payload["source"]] = source_counts.get(payload["source"], 0) + 1

    summary = {
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "taxonomy.json").write_text(
        json.dumps({"labels": sorted(class_counts.keys())}, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.enable_hf_download:
        stage_hf_sources(raw_root)

    all_records: List[Dict[str, str]] = []
    pending_sources: List[str] = []
    for config in SOURCE_CONFIGS.values():
        source_root = raw_root / config.raw_dir
        if not source_root.exists():
            pending_sources.append(config.source_id)
            continue
        records = iter_records(source_root, config, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        if not records:
            pending_sources.append(config.source_id)
            continue
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No image records were staged. Place raw dataset folders under the raw root and rerun this command."
        )

    summary = write_combined_dataset(records=all_records, output_root=output_root, copy_mode=args.copy_mode)
    summary["pending_sources"] = pending_sources
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
