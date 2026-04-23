import re
from typing import Dict, List

from .clinical_safety import build_safety_summary
from .clinical_protocol import MANDATORY_DISCLAIMER


SEVERITY_PRIOR = {
    "eczema": 0.35,
    "psoriasis": 0.45,
    "acne": 0.20,
    "fungal infection": 0.55,
    "contact dermatitis": 0.30,
    "cellulitis": 0.75,
    "melanoma": 0.95,
    "malaria": 0.75,
}

FORCED_HIGH_CONDITIONS = {
    "melanoma",
}

RED_FLAG_TERMS = [
    "chest pain",
    "trouble breathing",
    "shortness of breath",
    "high fever",
    "fainting",
    "severe bleeding",
    "rapid swelling",
    "unconscious",
    "slurred speech",
    "one side weakness",
    "one-side weakness",
    "one sided weakness",
    "unilateral weakness",
    "facial droop",
    "face droop",
    "confusion",
]

GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)

DISCLAIMER_TEXT = MANDATORY_DISCLAIMER


def detect_red_flags(symptom_text: str) -> List[str]:
    text = symptom_text.lower()
    flags = [term for term in RED_FLAG_TERMS if term in text]
    if _stroke_pattern(text):
        flags.append("possible stroke pattern")
    if _acute_coronary_pattern(text):
        flags.append("possible cardiac emergency pattern")
    if _sepsis_pattern(text):
        flags.append("possible sepsis pattern")
    if _kidney_fever_pattern(text):
        flags.append("possible kidney infection pattern")
    safety_summary = build_safety_summary(symptom_text)
    flags.extend(safety_summary["red_flags"])
    return sorted(set(flags))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _risk_level(score: float) -> str:
    if score > 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def _stroke_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    neuro_terms = [
        "slurred speech",
        "one side weakness",
        "one-side weakness",
        "one sided weakness",
        "unilateral weakness",
        "facial droop",
        "face droop",
        "difficulty speaking",
        "cannot speak",
    ]
    return any(term in text for term in neuro_terms)


def _acute_coronary_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_chest_pain = "chest pain" in text or "chest pressure" in text or "pressure in chest" in text
    has_associated = any(
        term in text
        for term in [
            "trouble breathing",
            "shortness of breath",
            "sweating",
            "sweat",
            "left arm pain",
            "jaw pain",
        ]
    )
    return has_chest_pain and has_associated


def _sepsis_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_infection = any(term in text for term in ["fever", "chills", "infection"])
    has_systemic = any(term in text for term in ["confusion", "unconscious", "low blood pressure", "very weak"])
    return has_infection and has_systemic


def _kidney_fever_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    has_fever = "fever" in text or "chills" in text
    has_kidney_zone = any(term in text for term in ["kidney", "flank", "lower back", "back pain"])
    persistent_hint = any(term in text for term in ["long time", "for days", "for weeks", "persistent"])
    urinary_hint = any(term in text for term in ["burning urination", "frequent urination", "dysuria", "urine"])
    return has_fever and has_kidney_zone and (persistent_hint or urinary_hint)


def _lower_uti_pattern(symptom_text: str) -> bool:
    text = symptom_text.lower()
    return any(
        term in text
        for term in [
            "burning urination",
            "burning micturition",
            "painful urination",
            "dysuria",
            "frequent urination",
            "frequent urine",
            "urinary frequency",
            "lower abdomen pain",
            "suprapubic",
        ]
    )


def _recommendation(
    risk_level: str,
    red_flags: List[str],
    confidence_band: str,
    symptom_text: str,
) -> str:
    if "possible stroke pattern" in red_flags or "possible cardiac emergency pattern" in red_flags or "possible sepsis pattern" in red_flags:
        return "Emergency pattern detected: seek immediate emergency care now."
    if _kidney_fever_pattern(symptom_text):
        return (
            "Possible urinary/kidney involvement pattern: arrange a same-day in-person medical evaluation "
            "for urine tests and clinical examination."
        )
    if _lower_uti_pattern(symptom_text):
        return "Likely urinary symptom pattern: arrange a same-day or next-day clinic visit for urine testing and treatment guidance."
    if red_flags or risk_level == "High":
        return "Urgent: seek immediate in-person medical care or emergency services."
    if confidence_band == "low":
        return "Low confidence result: arrange a clinical consultation for proper evaluation."
    if risk_level == "Medium":
        return "Schedule a same-day or next-day clinic visit for professional assessment."
    return "Low-risk pattern: monitor symptoms and seek care if symptoms worsen or persist."


def _risk_factors(
    *,
    top_condition: str,
    top_prob: float,
    red_flags: List[str],
    vulnerability: float,
    uncertainty: float,
    disagreement: float,
) -> List[str]:
    if GENERIC_CONDITION_RE.fullmatch(top_condition.strip()):
        top_condition_text = "non-specific symptom pattern"
    else:
        top_condition_text = top_condition

    factors: List[str] = [
        f"Primary predicted condition pattern: {top_condition_text} ({top_prob * 100:.1f}% confidence).",
    ]
    if red_flags:
        factors.append(f"Red-flag symptoms detected: {', '.join(red_flags)}.")
    if vulnerability >= 0.3:
        factors.append("Patient vulnerability context increases caution level (age/comorbid factors).")
    if uncertainty >= 0.5:
        factors.append("Model uncertainty is elevated; clinical confirmation is important.")
    if disagreement >= 0.4:
        factors.append("Text and image predictions disagree meaningfully; treat result as lower certainty.")
    return factors


def _prevention_advice(risk_level: str, top_condition: str) -> List[str]:
    advice = [
        "Stay hydrated and track symptom progression at least twice daily.",
        "Avoid self-medication without clinician guidance, especially antibiotics/steroids.",
        "Seek in-person evaluation if symptoms worsen, persist, or new severe symptoms appear.",
    ]
    condition = top_condition.lower()
    if "infection" in condition:
        advice.insert(0, "Maintain strict hygiene and avoid sharing personal items.")
    if "dermatitis" in condition or "eczema" in condition:
        advice.insert(0, "Use gentle skin care products and avoid known irritants/allergens.")
    if risk_level == "High":
        advice = ["Do not delay care: proceed to emergency or urgent care services immediately."] + advice
    return advice[:5]


def compute_risk(
    fused_predictions: List[Dict[str, float]],
    uncertainty: float,
    disagreement: float,
    symptom_text: str,
    confidence_band: str,
    vulnerability: float = 0.0,
):
    top_condition = fused_predictions[0]["condition"] if fused_predictions else "unspecified"
    top_prob = float(fused_predictions[0]["probability"]) if fused_predictions else 0.0
    severity_prior = SEVERITY_PRIOR.get(str(top_condition).lower(), 0.45)
    severity_component = _clip(severity_prior * top_prob)

    red_flags = detect_red_flags(symptom_text)
    safety_summary = build_safety_summary(symptom_text)
    redflag_component = _clip(len(red_flags) * 0.35)
    vulnerability_component = _clip(vulnerability)
    uncertainty_component = _clip(uncertainty)
    disagreement_component = _clip(disagreement)

    risk_score = _clip(
        0.50 * severity_component
        + 0.20 * redflag_component
        + 0.15 * vulnerability_component
        + 0.10 * uncertainty_component
        + 0.05 * disagreement_component
    )
    risk_level = _risk_level(risk_score)
    forced_risk = safety_summary.get("risk_level")
    if forced_risk == "High":
        risk_score = max(risk_score, 0.85)
        risk_level = "High"
    elif forced_risk == "Medium" and risk_level == "Low":
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"
    if _stroke_pattern(symptom_text) or _acute_coronary_pattern(symptom_text) or _sepsis_pattern(symptom_text):
        risk_score = max(risk_score, 0.85)
        risk_level = "High"
    if str(top_condition).lower() in FORCED_HIGH_CONDITIONS and top_prob >= 0.6:
        risk_score = max(risk_score, 0.75)
        risk_level = "High"
    if "malaria" in str(top_condition).lower() and any(term in symptom_text.lower() for term in ["fever", "chills", "headache"]):
        risk_score = max(risk_score, 0.4)
        risk_level = "Medium"
    if risk_level == "Low" and _kidney_fever_pattern(symptom_text):
        risk_score = max(risk_score, 0.38)
        risk_level = "Medium"
    if risk_level == "Low" and _lower_uti_pattern(symptom_text):
        risk_score = max(risk_score, 0.34)
        risk_level = "Medium"

    recommendation = _recommendation(
        risk_level=risk_level,
        red_flags=red_flags,
        confidence_band=confidence_band,
        symptom_text=symptom_text,
    )
    if safety_summary.get("recommendation") and (
        forced_risk == "High"
        or forced_risk == "Medium"
        or recommendation.startswith("Low confidence")
    ):
        recommendation = safety_summary["recommendation"]
    risk_factors = _risk_factors(
        top_condition=str(top_condition),
        top_prob=top_prob,
        red_flags=red_flags,
        vulnerability=vulnerability_component,
        uncertainty=uncertainty_component,
        disagreement=disagreement_component,
    )
    prevention_advice = _prevention_advice(risk_level=risk_level, top_condition=str(top_condition))

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "severity_component": round(severity_component, 4),
        "redflag_component": round(redflag_component, 4),
        "vulnerability_component": round(vulnerability_component, 4),
        "uncertainty_component": round(uncertainty_component, 4),
        "disagreement_component": round(disagreement_component, 4),
        "recommendation_text": recommendation,
        "disclaimer_text": DISCLAIMER_TEXT,
        "needs_urgent_care": bool(risk_level == "High" or red_flags),
        "red_flags": red_flags,
        "risk_factors": risk_factors,
        "prevention_advice": prevention_advice,
    }
