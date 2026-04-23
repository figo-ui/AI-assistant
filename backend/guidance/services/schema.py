import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


VALID_RISK_LEVELS = {"Low", "Medium", "High"}


@dataclass
class ConditionProbability:
    condition: str
    probability: float


@dataclass
class TriageLLMResponse:
    predictions: List[ConditionProbability]
    risk_level: str
    red_flags: List[str]
    recommendation: str
    reasoning: str
    raw_text: str = ""


def response_schema_prompt() -> str:
    return json.dumps(
        {
            "predictions": [
                {"condition": "condition name", "probability": 0.61},
                {"condition": "alternative condition", "probability": 0.24},
                {"condition": "third condition", "probability": 0.15},
            ],
            "risk_level": "Low|Medium|High",
            "red_flags": ["flag 1", "flag 2"],
            "recommendation": "clear next action",
            "reasoning": "brief clinical reasoning",
        },
        indent=2,
    )


def validate_triage_response(payload: Dict, raw_text: str = "") -> TriageLLMResponse:
    predictions_payload = payload.get("predictions")
    if not isinstance(predictions_payload, list):
        raise ValueError("predictions must be a list")

    predictions: List[ConditionProbability] = []
    for item in predictions_payload[:5]:
        if not isinstance(item, dict):
            continue
        condition = str(item.get("condition", "")).strip()
        if not condition:
            continue
        try:
            probability = float(item.get("probability", 0.0))
        except Exception:
            probability = 0.0
        probability = max(0.0, min(1.0, probability))
        predictions.append(ConditionProbability(condition=condition, probability=probability))

    if predictions:
        total = sum(item.probability for item in predictions)
        if total > 0:
            predictions = [
                ConditionProbability(
                    condition=item.condition,
                    probability=round(item.probability / total, 4),
                )
                for item in predictions
            ]
        else:
            uniform = round(1.0 / len(predictions), 4)
            predictions = [
                ConditionProbability(condition=item.condition, probability=uniform)
                for item in predictions
            ]

    risk_level = str(payload.get("risk_level", "Medium")).title()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "Medium"

    red_flags_payload = payload.get("red_flags", [])
    red_flags = []
    if isinstance(red_flags_payload, list):
        red_flags = [str(item).strip() for item in red_flags_payload if str(item).strip()]

    recommendation = str(payload.get("recommendation", "")).strip()
    reasoning = str(payload.get("reasoning", "")).strip()

    return TriageLLMResponse(
        predictions=predictions,
        risk_level=risk_level,
        red_flags=red_flags,
        recommendation=recommendation,
        reasoning=reasoning,
        raw_text=raw_text,
    )


def response_to_dict(response: TriageLLMResponse) -> Dict:
    payload = asdict(response)
    payload["predictions"] = [asdict(item) for item in response.predictions]
    return payload
