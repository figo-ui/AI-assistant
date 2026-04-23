import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
from django.conf import settings

from .preprocess import clean_symptom_text


DEFAULT_TEMPLATES = {
    "high": "Thank you for sharing these details quickly.",
    "medium": "Thank you for explaining your symptoms clearly.",
    "low": "Thanks for sharing your symptoms.",
}


@lru_cache(maxsize=1)
def _load_dialogue_artifacts():
    model_path = Path(settings.DIALOGUE_INTENT_MODEL_PATH)
    vectorizer_path = Path(settings.DIALOGUE_INTENT_VECTORIZER_PATH)
    templates_path = Path(settings.DIALOGUE_RESPONSE_TEMPLATES_PATH)

    model = None
    vectorizer = None
    templates: Dict[str, str] = {}

    if model_path.exists() and vectorizer_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

    if templates_path.exists():
        templates = json.loads(templates_path.read_text(encoding="utf-8"))

    return model, vectorizer, templates


def build_supportive_opening(symptom_text: str, risk_level: str) -> str:
    fallback = DEFAULT_TEMPLATES.get(str(risk_level or "").lower(), DEFAULT_TEMPLATES["low"])
    model, vectorizer, templates = _load_dialogue_artifacts()

    if model is None or vectorizer is None or not templates:
        return fallback

    cleaned = clean_symptom_text(symptom_text)
    if not cleaned:
        return fallback

    matrix = vectorizer.transform([cleaned])
    intent = str(model.predict(matrix)[0]).strip().lower()
    intent_template = str(templates.get(intent, "")).strip()
    if not intent_template:
        return fallback

    return intent_template
