import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "dataset_hub" / "healthcare_dataset_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "dataset_hub" / "datasets"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "dataset_hub" / "DATASET_ACQUISITION_SUMMARY.md"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "dataset_hub" / "metadata_cache.json"

CATEGORY_ORDER = ["EHR", "Imaging", "Dialogue", "Surveys", "Audio", "Genomics"]
TEXT_EXTENSIONS = {".txt", ".md", ".jsonl", ".tsv", ".csv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetStatus:
    dataset_id: str
    name: str
    category: str
    local_path: str
    status: str
    access: str
    note: str
    processed: str
    files_detected: int


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "dataset"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def session_with_retries(retries: int, timeout: int) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "healthcare-dataset-acquirer/1.0"})
    session.request_timeout = timeout  # type: ignore[attr-defined]
    session.request_retries = retries  # type: ignore[attr-defined]
    return session


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    retries: int,
    timeout: int,
    stream: bool = False,
) -> requests.Response:
    delay = 1.0
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method=method, url=url, timeout=timeout, stream=stream)
            if response.status_code >= 500:
                raise requests.HTTPError(f"{response.status_code} server error for {url}")
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 12.0)
            else:
                break
    raise RuntimeError(f"Failed request after {retries} attempts: {url} ({last_error})")


def file_name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name or "download.bin"


def download_url(
    session: requests.Session,
    url: str,
    target_dir: Path,
    retries: int,
    timeout: int,
    max_bytes: int,
    sleep_seconds: float,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    response = request_with_retry(session, "GET", url, retries=retries, timeout=timeout, stream=True)
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        raise RuntimeError(f"Skipping {url}: size {content_length} exceeds limit {max_bytes} bytes")
    filename = file_name_from_url(url)
    destination = target_dir / filename
    downloaded = 0
    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 512):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                output.close()
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"Download exceeded size cap for {url}")
            output.write(chunk)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return destination


def run_git_clone(repo_url: str, target_dir: Path) -> Path:
    def is_valid_repo(path: Path) -> bool:
        try:
            probe = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return probe.stdout.strip().lower() == "true"
        except Exception:
            return False

    target_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = target_dir / slugify(Path(urlparse(repo_url).path).stem)
    git_dir = clone_dir / ".git"
    if clone_dir.exists() and git_dir.exists() and is_valid_repo(clone_dir):
        return clone_dir
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)

    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        [
            "git",
            "-c",
            "http.lowSpeedLimit=1000",
            "-c",
            "http.lowSpeedTime=30",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            repo_url,
            str(clone_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    if not is_valid_repo(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(f"Cloned folder is not a valid git repo: {clone_dir}")
    return clone_dir


def collect_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if "processed" in path.parts:
            continue
        files.append(path)
    return files


def normalize_columns(columns: Iterable[str]) -> List[str]:
    normalized = []
    for col in columns:
        text = str(col).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        normalized.append(text.strip("_") or "column")
    return normalized


def preprocess_tabular(dataset_dir: Path, max_files: int = 4, max_rows: int = 8000) -> str:
    tabular_files = [p for p in collect_files(dataset_dir) if p.suffix.lower() in {".csv", ".tsv"}]
    json_files = [p for p in collect_files(dataset_dir) if p.suffix.lower() == ".json"]
    if not tabular_files and not json_files:
        return "skipped (no tabular/json files found)"

    processed_dir = dataset_dir / "processed" / "tabular"
    processed_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for src in tabular_files[:max_files]:
        sep = "\t" if src.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(src, sep=sep, low_memory=False, nrows=max_rows)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = normalize_columns(df.columns)
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                median = df[column].median()
                if pd.notna(median):
                    df[column] = df[column].fillna(median)
                std = df[column].std()
                if pd.notna(std) and std > 0:
                    df[column] = (df[column] - df[column].mean()) / std
            else:
                df[column] = df[column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        out_file = processed_dir / f"{src.stem}_normalized.csv"
        df.to_csv(out_file, index=False)
        converted += 1

    for src in json_files[:2]:
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = None
        if isinstance(payload, list):
            rows = payload[:max_rows]
        elif isinstance(payload, dict):
            for key in ("dataset", "results", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value[:max_rows]
                    break
            if rows is None:
                rows = [payload]
        if not rows:
            continue
        try:
            df = pd.json_normalize(rows)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = normalize_columns(df.columns)
        out_file = processed_dir / f"{src.stem}_normalized.csv"
        df.head(max_rows).to_csv(out_file, index=False)
        converted += 1

    return f"tabular normalized files: {converted}"


def guess_dialogue_columns(columns: List[str]) -> tuple[str | None, str | None]:
    question_aliases = ["question", "query", "prompt", "input", "utterance", "patient"]
    answer_aliases = ["answer", "response", "output", "doctor", "assistant"]

    q_col = next((c for c in columns if c.lower() in question_aliases), None)
    a_col = next((c for c in columns if c.lower() in answer_aliases), None)
    return q_col, a_col


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def preprocess_dialogue(dataset_dir: Path, max_rows: int = 15000) -> str:
    text_files = []
    for path in collect_files(dataset_dir):
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text_files.append(path)
            continue
        if suffix == "":
            text_files.append(path)
    if not text_files:
        return "skipped (no dialogue text files found)"

    processed_dir = dataset_dir / "processed" / "dialogue"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_file = processed_dir / "dialogue_cleaned.jsonl"
    written = 0

    with out_file.open("w", encoding="utf-8") as sink:
        for src in text_files[:6]:
            if src.suffix.lower() in {".csv", ".tsv"}:
                sep = "\t" if src.suffix.lower() == ".tsv" else ","
                try:
                    df = pd.read_csv(src, sep=sep, low_memory=False, nrows=max_rows)
                except Exception:
                    continue
                q_col, a_col = guess_dialogue_columns([str(c) for c in df.columns])
                if not q_col or not a_col:
                    continue
                for _, row in df[[q_col, a_col]].dropna().iterrows():
                    prompt = clean_text(row[q_col])
                    response = clean_text(row[a_col])
                    if not prompt or not response:
                        continue
                    item = {
                        "prompt": prompt,
                        "response": response,
                        "prompt_tokens": prompt.split(),
                        "response_tokens": response.split(),
                    }
                    sink.write(json.dumps(item, ensure_ascii=True) + "\n")
                    written += 1
                    if written >= max_rows:
                        return f"dialogue cleaned rows: {written}"
            elif src.suffix.lower() in {".txt", ".md", ""}:
                lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in lines:
                    text = clean_text(line)
                    if len(text) < 20:
                        continue
                    item = {
                        "prompt": text,
                        "response": "",
                        "prompt_tokens": text.split(),
                        "response_tokens": [],
                    }
                    sink.write(json.dumps(item, ensure_ascii=True) + "\n")
                    written += 1
                    if written >= max_rows:
                        return f"dialogue cleaned rows: {written}"

    return f"dialogue cleaned rows: {written}"


def preprocess_imaging(dataset_dir: Path, max_images: int = 200) -> str:
    image_files = [p for p in collect_files(dataset_dir) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        return "skipped (no compatible image files found)"

    output_dir = dataset_dir / "processed" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for src in image_files[:max_images]:
        try:
            with Image.open(src) as image:
                rgb = image.convert("RGB")
                resized = rgb.resize((512, 512))
                dst = output_dir / f"{src.stem}.jpg"
                resized.save(dst, format="JPEG", quality=92)
                converted += 1
        except Exception:
            continue
    return f"image files converted: {converted}"


def write_request_stub(dataset_dir: Path, dataset: Dict) -> None:
    lines = [
        f"Dataset: {dataset['name']}",
        f"Access: {dataset.get('access', '')}",
        f"Homepage: {dataset.get('homepage_url', '')}",
        "",
        "Action required:",
        "1. Visit the homepage and complete the required registration/request flow.",
        "2. Accept terms/licensing and any DUA if required.",
        "3. Download data into this dataset folder under raw/.",
        "4. Re-run the acquisition script to process and refresh metadata.",
    ]
    if dataset.get("source_urls"):
        lines.append("")
        lines.append("Reference URLs:")
        for url in dataset["source_urls"]:
            lines.append(f"- {url}")
    (dataset_dir / "REQUEST_REQUIRED.txt").write_text("\n".join(lines), encoding="utf-8")


def preprocess_by_category(category: str, dataset_dir: Path) -> str:
    if category in {"EHR", "Surveys", "Genomics"}:
        return preprocess_tabular(dataset_dir)
    if category == "Dialogue":
        return preprocess_dialogue(dataset_dir)
    if category == "Imaging":
        return preprocess_imaging(dataset_dir)
    if category == "Audio":
        return "skipped (audio preprocessing hook not enabled in this run)"
    return "skipped"


def cache_dataset_metadata(dataset_dir: Path, dataset: Dict, processed_note: str, status: str, note: str) -> Dict:
    files = collect_files(dataset_dir)
    file_rows = []
    for path in files[:2000]:
        rel = path.relative_to(dataset_dir).as_posix()
        file_rows.append({"path": rel, "size_bytes": path.stat().st_size})
    return {
        "dataset_id": dataset["id"],
        "name": dataset["name"],
        "category": dataset["category"],
        "status": status,
        "note": note,
        "processed": processed_note,
        "updated_at": utc_now(),
        "files_count": len(files),
        "files": file_rows,
    }


def render_report(statuses: List[DatasetStatus], report_path: Path) -> None:
    lines = [
        "# Healthcare Dataset Acquisition Summary",
        "",
        f"Generated at: {utc_now()}",
        "",
        "## Dataset Status",
        "",
        "| Category | Dataset | Status | Access | Local Path | Preprocessing | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(statuses, key=lambda item: (CATEGORY_ORDER.index(item.category), item.name))
    for item in ordered:
        lines.append(
            f"| {item.category} | {item.name} | {item.status} | {item.access} | `{item.local_path}` | {item.processed} | {item.note} |"
        )

    lines.extend(
        [
            "",
            "## Directory Layout",
            "",
            "- `data/dataset_hub/datasets/EHR/`",
            "- `data/dataset_hub/datasets/Imaging/`",
            "- `data/dataset_hub/datasets/Dialogue/`",
            "- `data/dataset_hub/datasets/Surveys/`",
            "- `data/dataset_hub/datasets/Audio/`",
            "- `data/dataset_hub/datasets/Genomics/`",
            "",
            "## Access Notes",
            "",
            "- Datasets with `manual_request` require registration, credentialing, or data use agreements.",
            "- Only open-access or academic-free datasets are included in the manifest.",
            "- For very large datasets, this run stores metadata and request instructions by default.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire, organize, and preprocess healthcare datasets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--mode",
        choices=["metadata", "download"],
        default="metadata",
        help="metadata: create structure + request stubs + metadata cache. download: fetch supported URLs too.",
    )
    parser.add_argument("--include-category", action="append", default=[])
    parser.add_argument("--include-id", action="append", default=[])
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-download-gb", type=float, default=1.0)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument(
        "--disable-git-clone",
        action="store_true",
        help="Treat git_clone datasets as manual request in this run.",
    )
    return parser.parse_args()


def should_include(dataset: Dict, include_categories: List[str], include_ids: List[str]) -> bool:
    if include_categories and dataset["category"] not in include_categories:
        return False
    if include_ids and dataset["id"] not in include_ids:
        return False
    return True


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    datasets = manifest["datasets"]
    max_bytes = int(args.max_download_gb * (1024 ** 3))
    session = session_with_retries(retries=args.retry, timeout=args.timeout)

    statuses: List[DatasetStatus] = []
    metadata_cache: List[Dict] = []

    for category in CATEGORY_ORDER:
        (args.output_root / category).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        if not should_include(dataset, args.include_category, args.include_id):
            continue

        dataset_dir = args.output_root / dataset["category"] / slugify(dataset["name"])
        raw_dir = dataset_dir / "raw"
        if args.force_clean and dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        safe_write_json(dataset_dir / "dataset_info.json", dataset)

        status = "prepared"
        note = "metadata prepared"
        processed = "skipped"

        method = dataset.get("download_method", "manual_request")
        urls = dataset.get("download_urls", [])

        try:
            if args.mode == "download":
                if method == "direct_http" and urls:
                    downloaded = []
                    for url in urls:
                        file_path = download_url(
                            session=session,
                            url=url,
                            target_dir=raw_dir,
                            retries=args.retry,
                            timeout=args.timeout,
                            max_bytes=max_bytes,
                            sleep_seconds=args.sleep_seconds,
                        )
                        downloaded.append(file_path.name)
                    status = "downloaded"
                    note = f"downloaded {len(downloaded)} file(s)"
                elif method == "git_clone" and urls:
                    if args.disable_git_clone:
                        write_request_stub(dataset_dir, dataset)
                        status = "request_required"
                        note = "git clone disabled for this run; manual fetch required"
                    else:
                        for repo in urls:
                            run_git_clone(repo, raw_dir)
                        status = "downloaded"
                        note = "repository cloned"
                elif method == "manual_request":
                    write_request_stub(dataset_dir, dataset)
                    status = "request_required"
                    note = "manual access/request required"
                else:
                    write_request_stub(dataset_dir, dataset)
                    status = "request_required"
                    note = "no direct download automation for this dataset"
            else:
                if method == "manual_request":
                    write_request_stub(dataset_dir, dataset)
                    status = "request_required"
                    note = "manual access/request required"
                else:
                    status = "metadata_only"
                    note = "download skipped in metadata mode"
        except Exception as exc:  # noqa: BLE001
            if method == "git_clone":
                write_request_stub(dataset_dir, dataset)
                status = "request_required"
                note = f"git clone failed; manual fetch required ({exc})"
            else:
                status = "error"
                note = str(exc)

        if status in {"downloaded", "metadata_only"}:
            processed = preprocess_by_category(dataset["category"], dataset_dir)
        elif status == "request_required":
            processed = "pending (awaiting dataset download)"

        metadata_cache.append(cache_dataset_metadata(dataset_dir, dataset, processed, status, note))
        statuses.append(
            DatasetStatus(
                dataset_id=dataset["id"],
                name=dataset["name"],
                category=dataset["category"],
                local_path=str(dataset_dir),
                status=status,
                access=dataset.get("access", ""),
                note=note,
                processed=processed,
                files_detected=metadata_cache[-1]["files_count"],
            )
        )

    safe_write_json(
        args.cache_path,
        {
            "generated_at": utc_now(),
            "manifest": str(args.manifest),
            "mode": args.mode,
            "results": metadata_cache,
        },
    )
    render_report(statuses=statuses, report_path=args.report_path)
    print(f"Summary report written: {args.report_path}")
    print(f"Metadata cache written: {args.cache_path}")


if __name__ == "__main__":
    main()
