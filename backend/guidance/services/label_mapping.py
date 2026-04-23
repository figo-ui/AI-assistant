import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from django.conf import settings


ID_PATTERN = re.compile(r"^(?:condition\s*)?(\d+)$", re.IGNORECASE)


def _normalize_key(value: str) -> str:
    return str(value).strip().lower()


def _extract_numeric_key(value: str) -> str:
    match = ID_PATTERN.match(str(value).strip())
    return match.group(1) if match else ""


@lru_cache(maxsize=1)
def load_condition_map() -> Dict[str, str]:
    path = Path(getattr(settings, "CONDITION_NAME_MAP_PATH", ""))
    if not path or not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(data, dict):
            return {
                _normalize_key(k): str(v).strip()
                for k, v in data.items()
                if str(v).strip()
            }
        return {}

    if path.suffix.lower() == ".csv":
        mapping: Dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return {}
                names = [n.strip() for n in reader.fieldnames]
                key_col = "label_id" if "label_id" in names else names[0]
                value_col = "condition" if "condition" in names else names[1] if len(names) > 1 else names[0]
                for row in reader:
                    key = _normalize_key(str(row.get(key_col, "")))
                    value = str(row.get(value_col, "")).strip()
                    if key and value:
                        mapping[key] = value
        except Exception:
            return {}
        return mapping

    return {}


def map_condition_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name:
        return name

    mapping = load_condition_map()
    if not mapping:
        return name

    direct = mapping.get(_normalize_key(name))
    if direct:
        return direct

    numeric = _extract_numeric_key(name)
    if numeric:
        mapped = mapping.get(_normalize_key(numeric))
        if mapped:
            return mapped

    return name


def map_prediction_list(predictions: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in predictions:
        condition = map_condition_name(str(item.get("condition", "")))
        out.append(
            {
                **item,
                "condition": condition,
            }
        )
    return out
