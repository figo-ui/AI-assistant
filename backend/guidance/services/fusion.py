import math
from typing import Dict, List


def _to_distribution(predictions: List[Dict[str, float]]) -> Dict[str, float]:
    dist: Dict[str, float] = {}
    for pred in predictions:
        condition = str(pred.get("condition", "")).strip()
        if not condition:
            continue
        dist[condition] = max(float(pred.get("probability", 0.0)), 0.0)
    return dist


def _normalize(dist: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    total = sum(max(dist.get(key, 0.0), 0.0) for key in keys)
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: max(dist.get(key, 0.0), 0.0) / total for key in keys}


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    p_norm = _normalize(p, keys)
    q_norm = _normalize(q, keys)
    midpoint = {key: 0.5 * (p_norm[key] + q_norm[key]) for key in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            if a[key] > 0.0 and b[key] > 0.0:
                total += a[key] * math.log(a[key] / b[key], 2)
        return total

    return 0.5 * (_kl(p_norm, midpoint) + _kl(q_norm, midpoint))


def _single_modality_result(
    predictions: List[Dict[str, float]],
    *,
    version: str,
    text_weight: float,
    image_weight: float,
) -> Dict[str, object]:
    fused = sorted(predictions, key=lambda item: item["probability"], reverse=True)
    top_prob = float(fused[0]["probability"]) if fused else 0.0
    margin = top_prob - float(fused[1]["probability"]) if len(fused) > 1 else top_prob
    return {
        "predictions": fused,
        "confidence": round(top_prob, 4),
        "confidence_band": confidence_band(top_prob, margin),
        "uncertainty": round(1.0 - top_prob, 4),
        "disagreement": 0.0,
        "margin": round(margin, 4),
        "version": version,
        "modality_weights": {
            "text": round(text_weight, 4),
            "image": round(image_weight, 4),
        },
    }


def fuse_predictions(
    text_predictions: List[Dict[str, float]],
    image_predictions: List[Dict[str, float]],
    text_confidence: float,
    image_confidence: float,
    image_quality: float = 1.0,
    text_weight: float = 0.62,
    image_weight: float = 0.38,
    top_k: int = 5,
):
    text_dist = _to_distribution(text_predictions)
    image_dist = _to_distribution(image_predictions)

    if not image_dist:
        return _single_modality_result(
            sorted(text_predictions, key=lambda item: item["probability"], reverse=True)[:top_k],
            version="fusion-v2-text-only",
            text_weight=1.0 if text_predictions else 0.0,
            image_weight=0.0,
        )
    if not text_dist:
        return _single_modality_result(
            sorted(image_predictions, key=lambda item: item["probability"], reverse=True)[:top_k],
            version="fusion-v2-image-only",
            text_weight=0.0,
            image_weight=1.0 if image_predictions else 0.0,
        )

    disagreement = _js_divergence(text_dist, image_dist)
    agreement = max(0.2, 1.0 - disagreement)
    text_reliability = max(0.0, min(1.0, text_confidence))
    image_reliability = max(0.0, min(1.0, image_confidence * image_quality))

    alpha = text_weight * max(0.2, text_reliability)
    beta = image_weight * max(0.2, image_reliability)

    if text_reliability >= image_reliability:
        beta *= agreement
    else:
        alpha *= agreement

    if alpha + beta <= 0.0:
        alpha = text_weight
        beta = image_weight

    keys = sorted(set(text_dist.keys()) | set(image_dist.keys()))
    text_norm = _normalize(text_dist, keys)
    image_norm = _normalize(image_dist, keys)
    scores: Dict[str, float] = {}
    for condition in keys:
        score = alpha * text_norm.get(condition, 0.0) + beta * image_norm.get(condition, 0.0)
        scores[condition] = score

    total = sum(scores.values())
    if total > 0:
        scores = {key: value / total for key, value in scores.items()}

    fused = [{"condition": key, "probability": round(float(value), 4)} for key, value in scores.items()]
    fused.sort(key=lambda item: item["probability"], reverse=True)
    fused = fused[:top_k]

    top_prob = float(fused[0]["probability"]) if fused else 0.0
    margin = top_prob - float(fused[1]["probability"]) if len(fused) > 1 else top_prob
    total_weight = max(alpha + beta, 1e-6)

    return {
        "predictions": fused,
        "confidence": round(top_prob, 4),
        "confidence_band": confidence_band(top_prob, margin),
        "uncertainty": round(1.0 - top_prob, 4),
        "disagreement": round(disagreement, 4),
        "margin": round(margin, 4),
        "version": "fusion-v2",
        "modality_weights": {
            "text": round(alpha / total_weight, 4),
            "image": round(beta / total_weight, 4),
        },
        "agreement": round(agreement, 4),
    }


def confidence_band(max_prob: float, margin: float) -> str:
    if max_prob >= 0.65 and margin >= 0.2:
        return "high"
    if max_prob >= 0.42 or margin >= 0.1:
        return "medium"
    return "low"
