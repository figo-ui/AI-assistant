import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from django.conf import settings

from .clinical_safety import apply_prediction_safety_overrides
from .preprocess import clean_symptom_text


KEYWORD_PRIOR_MAP = {
    "eczema": ["itchy", "dry skin", "rash", "red patches"],
    "psoriasis": ["scaly", "silvery", "thick plaques"],
    "acne": ["pimples", "blackheads", "whiteheads", "oily skin"],
    "fungal infection": ["ring", "itching", "peeling", "spread"],
    "contact dermatitis": ["allergy", "soap", "cosmetic", "irritation"],
}

URINARY_KEYWORD_BOOST = {
    "Escherichia coli urinary tract infection": [
        "burning urination",
        "burning micturition",
        "urinary",
        "urine",
        "kidney pain",
        "flank pain",
    ],
    "Cystitis": [
        "burning urination",
        "burning micturition",
        "urine",
        "bladder",
        "lower abdomen",
    ],
    "Diabetic renal disease (disorder)": [
        "kidney",
        "renal",
        "flank",
    ],
}

UTI_RELATED_CLASS_HINTS = [
    "urinary tract infection",
    "cystitis",
    "pyeloneph",
    "kidney infection",
    "escherichia coli urinary tract infection",
    "renal infection",
]

RESPIRATORY_TERMS = [
    "cough",
    "sore throat",
    "runny nose",
    "shortness of breath",
    "trouble breathing",
    "breath",
    "wheezing",
    "loss of taste",
    "loss of smell",
]

STROKE_TERMS = [
    "slurred speech",
    "one side weakness",
    "one-side weakness",
    "one sided weakness",
    "unilateral weakness",
    "facial droop",
    "face droop",
    "cannot speak",
    "difficulty speaking",
]

CARDIAC_TERMS = [
    "chest pain",
    "pressure in chest",
    "chest pressure",
    "pain in chest",
    "left arm pain",
    "sweating",
    "sweat",
]

SEVERE_RESP_TERMS = [
    "trouble breathing",
    "shortness of breath",
    "breathless",
    "cannot breathe",
]

UTI_TERMS = [
    "burning urination",
    "burning micturition",
    "painful urination",
    "frequent urine",
    "frequent urination",
    "urinary frequency",
    "urine often",
    "lower abdomen pain",
    "suprapubic",
    "dysuria",
]

KIDNEY_TERMS = [
    "kidney",
    "flank",
    "lower back",
    "back pain",
    "loin",
]

FEVER_TERMS = [
    "fever",
    "high temperature",
    "temperature",
    "chills",
]

DERMATITIS_TERMS = [
    "itchy",
    "itching",
    "rash",
    "red skin",
    "red rash",
]

EXPOSURE_TERMS = [
    "chemical",
    "soap",
    "detergent",
    "cosmetic",
    "cream",
    "lotion",
    "after exposure",
    "allergy",
    "irritation",
]

PNEUMONIA_TERMS = [
    "cough",
    "fever",
    "chest pain",
    "pain when breathing",
    "trouble breathing",
    "shortness of breath",
    "phlegm",
]

URI_TERMS = [
    "sore throat",
    "runny nose",
    "mild cough",
    "low fever",
    "congestion",
    "sneezing",
]

PANIC_TERMS = [
    "fear",
    "panic",
    "fast heartbeat",
    "palpitations",
    "chest tightness",
    "feeling anxious",
]

MALARIA_TERMS = [
    "malaria",
    "mosquito",
    "fever",
    "chills",
    "headache",
]


@lru_cache(maxsize=1)
def _load_text_artifacts():
    model_path = Path(settings.TEXT_MODEL_PATH)
    vectorizer_path = Path(settings.TFIDF_VECTORIZER_PATH)
    labels_path = Path(settings.TEXT_LABELS_PATH)
    svd_path = Path(getattr(settings, "TEXT_SVD_PATH", ""))

    if not model_path.exists() or not vectorizer_path.exists():
        return None, None, [], None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    svd = joblib.load(svd_path) if svd_path and svd_path.exists() else None
    labels: List[str] = []

    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    elif hasattr(model, "classes_"):
        labels = [str(v) for v in model.classes_]

    return model, vectorizer, labels, svd


def _normalize_distribution(scores: Dict[str, float]) -> List[Dict[str, float]]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return []

    preds = [
        {"condition": k, "probability": round(max(v, 0.0) / total, 4)}
        for k, v in scores.items()
    ]
    preds.sort(key=lambda item: item["probability"], reverse=True)
    return preds


def _contains_any(text: str, terms: List[str]) -> bool:
    return any(term in text for term in terms)


def _find_label_indices(labels: List[str], exact_names: List[str], partial_names: List[str]) -> List[int]:
    indices: List[int] = []
    lower_labels = [str(label).strip().lower() for label in labels]
    for idx, name in enumerate(lower_labels):
        if name in [item.lower() for item in exact_names]:
            indices.append(idx)
            continue
        if any(partial in name for partial in [item.lower() for item in partial_names]):
            indices.append(idx)
    return sorted(set(indices))


def _apply_probability_floor(
    adjusted: np.ndarray,
    labels: List[str],
    *,
    exact_names: List[str],
    partial_names: List[str],
    floor: float,
) -> None:
    indices = _find_label_indices(labels, exact_names=exact_names, partial_names=partial_names)
    if not indices:
        return

    floor = max(0.0, min(0.95, float(floor)))
    share = floor / len(indices)
    non_target_total = float(np.sum(adjusted)) - float(np.sum(adjusted[indices]))
    if non_target_total <= 0:
        adjusted[:] = 0.0
        for idx in indices:
            adjusted[idx] = share
        return

    scale = max(0.0, 1.0 - floor) / non_target_total
    for idx in range(len(adjusted)):
        if idx not in indices:
            adjusted[idx] *= scale
    for idx in indices:
        adjusted[idx] = max(adjusted[idx], share)


def _heuristic_prediction(symptom_text: str, top_k: int):
    text = clean_symptom_text(symptom_text)
    scores: Dict[str, float] = {condition: 0.05 for condition in KEYWORD_PRIOR_MAP.keys()}
    for condition, keywords in KEYWORD_PRIOR_MAP.items():
        for keyword in keywords:
            if keyword in text:
                scores[condition] += 1.0

    predictions = _normalize_distribution(scores)[:top_k]
    confidence = predictions[0]["probability"] if predictions else 0.0
    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": "text-heuristic-v1",
    }


def _keyword_boost_distribution(
    clean_text: str,
    labels: List[str],
    probs: List[float],
) -> List[float]:
    if not labels or not probs:
        return probs

    label_to_idx = {str(label).strip(): idx for idx, label in enumerate(labels)}
    lower_labels = [str(label).strip().lower() for label in labels]
    adjusted = np.asarray([float(p) for p in probs], dtype=np.float64)
    adjusted = np.clip(adjusted, 1e-12, 1.0)

    kidney_pattern = _contains_any(clean_text, FEVER_TERMS) and _contains_any(clean_text, KIDNEY_TERMS)
    lower_uti_pattern = _contains_any(clean_text, UTI_TERMS)
    stroke_pattern = _contains_any(clean_text, STROKE_TERMS)
    cardiac_pattern = _contains_any(clean_text, CARDIAC_TERMS) and (
        _contains_any(clean_text, SEVERE_RESP_TERMS)
        or "sweating" in clean_text
        or "sweat" in clean_text
    )
    dermatitis_pattern = _contains_any(clean_text, DERMATITIS_TERMS) and _contains_any(clean_text, EXPOSURE_TERMS)
    pneumonia_pattern = (
        "cough" in clean_text
        and _contains_any(clean_text, FEVER_TERMS)
        and ("chest pain" in clean_text or _contains_any(clean_text, SEVERE_RESP_TERMS) or "pain when breathing" in clean_text)
    )
    uri_pattern = (
        _contains_any(clean_text, ["sore throat", "runny nose"])
        and ("cough" in clean_text or "mild cough" in clean_text or "low fever" in clean_text)
    )
    panic_pattern = _contains_any(clean_text, ["chest tightness", "fast heartbeat", "palpitations"]) and _contains_any(
        clean_text, ["fear", "panic", "anxious"]
    )
    malaria_pattern = (
        ("malaria" in clean_text or "mosquito" in clean_text)
        and _contains_any(clean_text, ["fever", "chills", "headache"])
    )
    has_respiratory = _contains_any(clean_text, RESPIRATORY_TERMS)

    for condition, keywords in URINARY_KEYWORD_BOOST.items():
        idx = label_to_idx.get(condition)
        if idx is None:
            continue
        hits = sum(1 for keyword in keywords if keyword in clean_text)
        if hits > 0:
            adjusted[idx] *= min(6.0, 1.0 + 1.2 * hits)
        if kidney_pattern:
            adjusted[idx] *= 8.0

    if kidney_pattern:
        for idx, name in enumerate(lower_labels):
            if any(hint in name for hint in UTI_RELATED_CLASS_HINTS):
                adjusted[idx] *= 5.0

    if lower_uti_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Urinary tract infection"],
            partial_names=["urinary tract infection"],
            floor=0.42,
        )

    if kidney_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Urinary tract infection"],
            partial_names=["urinary tract infection", "pyeloneph", "renal infection"],
            floor=0.55,
        )

    if stroke_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Stroke", "Paralysis (brain hemorrhage)"],
            partial_names=["stroke", "brain hemorrhage", "paralysis"],
            floor=0.72,
        )

    if cardiac_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Heart attack"],
            partial_names=["heart attack"],
            floor=0.58,
        )

    if dermatitis_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Allergy", "Drug Reaction"],
            partial_names=["allergy", "drug reaction"],
            floor=0.58,
        )
        for idx, name in enumerate(lower_labels):
            if "fungal infection" in name:
                adjusted[idx] *= 0.2

    if pneumonia_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Pneumonia"],
            partial_names=["pneumonia"],
            floor=0.52,
        )

    if uri_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Common Cold", "Viral pharyngitis", "Acute viral pharyngitis (disorder)", "Viral sinusitis (disorder)"],
            partial_names=["common cold", "viral pharyngitis", "viral sinusitis"],
            floor=0.62,
        )

    if panic_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Panic attack"],
            partial_names=["panic attack"],
            floor=0.48,
        )

    if malaria_pattern:
        _apply_probability_floor(
            adjusted,
            labels,
            exact_names=["Malaria"],
            partial_names=["malaria"],
            floor=0.52,
        )

    if not has_respiratory:
        for idx, name in enumerate(lower_labels):
            if "covid" in name:
                adjusted[idx] *= 0.08
            if "acute respiratory failure" in name or "respiratory distress" in name:
                adjusted[idx] *= 0.12

    if "chest pain" not in clean_text:
        for idx, name in enumerate(lower_labels):
            if "heart attack" in name:
                adjusted[idx] *= 0.2

    if not any(term in clean_text for term in ["leg swelling", "calf pain", "calf swelling", "deep vein", "dvt"]):
        for idx, name in enumerate(lower_labels):
            if "deep venous thrombosis" in name:
                adjusted[idx] *= 0.15

    total = float(np.sum(np.clip(adjusted, 0.0, None)))
    if total <= 0:
        return probs

    return [float(max(v, 0.0) / total) for v in adjusted]


def predict_text_probabilities(symptom_text: str, top_k: int = 5):
    model, vectorizer, labels, svd = _load_text_artifacts()
    if model is None or vectorizer is None:
        return _heuristic_prediction(symptom_text, top_k=top_k)

    clean_text = clean_symptom_text(symptom_text)
    matrix = vectorizer.transform([clean_text])
    if svd is not None:
        matrix = svd.transform(matrix)
    probs = model.predict_proba(matrix)[0]
    model_labels = labels if labels else [str(v) for v in getattr(model, "classes_", [])]
    probs = _keyword_boost_distribution(clean_text, model_labels, list(probs))

    if not model_labels:
        model_labels = [str(i) for i in range(len(probs))]

    predictions = [
        {"condition": model_labels[idx], "probability": round(float(prob), 4)}
        for idx, prob in enumerate(probs)
    ]
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    predictions = apply_prediction_safety_overrides(clean_text, predictions, top_k=top_k)
    predictions = predictions[:top_k]
    confidence = predictions[0]["probability"] if predictions else 0.0

    return {
        "predictions": predictions,
        "confidence": round(confidence, 4),
        "model_version": "text-ml-v1",
    }
