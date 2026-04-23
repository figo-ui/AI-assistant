import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


WHITESPACE_RE = re.compile(r"\s+")


def clean_symptom_text(text: str) -> str:
    text = text.strip().lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def prepare_image_tensor(image_path: Path, image_size: int) -> Optional[np.ndarray]:
    try:
        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    except Exception:
        return None

    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
