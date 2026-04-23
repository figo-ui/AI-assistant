from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


RISK_RANK = {"Low": 1, "Medium": 2, "High": 3}


@dataclass(frozen=True)
class SafetyPattern:
    key: str
    risk_level: str
    required_any: tuple[str, ...] = ()
    required_all_groups: tuple[tuple[str, ...], ...] = ()
    override_conditions: tuple[str, ...] = ()
    recommendation: str = ""
    red_flags: tuple[str, ...] = ()
    probability_floor: float = 0.55
    priority: int = 50


PATTERNS: tuple[SafetyPattern, ...] = (
    SafetyPattern(
        key="stroke",
        risk_level="High",
        required_any=("slurred speech", "facial droop", "face droop", "one sided weakness", "one side weakness", "unilateral weakness"),
        override_conditions=("Stroke", "Paralysis (brain hemorrhage)"),
        recommendation="Possible stroke pattern: call emergency services or go to the emergency department immediately.",
        red_flags=("possible stroke pattern",),
        probability_floor=0.72,
        priority=90,
    ),
    SafetyPattern(
        key="acute_coronary_syndrome",
        risk_level="High",
        required_all_groups=(("chest pain", "chest pressure", "pressure in chest"), ("trouble breathing", "shortness of breath", "sweating", "left arm pain", "jaw pain")),
        override_conditions=("Heart attack", "Unstable angina", "Possible NSTEMI / STEMI"),
        recommendation="Possible cardiac emergency pattern: seek emergency care immediately.",
        red_flags=("possible cardiac emergency pattern",),
        probability_floor=0.68,
        priority=70,
    ),
    SafetyPattern(
        key="pulmonary_embolism",
        risk_level="High",
        required_all_groups=(("chest pain", "trouble breathing", "shortness of breath"), ("calf pain", "leg swelling", "calf swelling")),
        override_conditions=("Pulmonary embolism", "Acute pulmonary embolism (disorder)"),
        recommendation="Possible pulmonary embolism pattern: seek emergency care immediately.",
        red_flags=("possible pulmonary embolism pattern",),
        probability_floor=0.64,
        priority=92,
    ),
    SafetyPattern(
        key="aortic_dissection",
        risk_level="High",
        required_all_groups=(("chest pain", "back pain"), ("tearing", "ripping", "sudden severe")),
        override_conditions=("Possible aortic dissection",),
        recommendation="Possible aortic dissection pattern: emergency evaluation is required immediately.",
        red_flags=("possible aortic dissection pattern",),
        probability_floor=0.70,
        priority=96,
    ),
    SafetyPattern(
        key="sepsis",
        risk_level="High",
        required_all_groups=(("fever", "chills", "infection"), ("confusion", "unconscious", "very weak", "low blood pressure")),
        override_conditions=("Sepsis caused by virus (disorder)",),
        recommendation="Possible sepsis pattern: seek emergency care immediately.",
        red_flags=("possible sepsis pattern",),
        probability_floor=0.70,
        priority=84,
    ),
    SafetyPattern(
        key="meningitis",
        risk_level="High",
        required_all_groups=(("fever", "high fever"), ("stiff neck", "neck stiffness"), ("severe headache", "headache", "light sensitivity", "photophobia", "confusion")),
        override_conditions=("Possible meningitis",),
        recommendation="Possible meningitis pattern: emergency evaluation is required immediately.",
        red_flags=("possible meningitis pattern",),
        probability_floor=0.72,
        priority=91,
    ),
    SafetyPattern(
        key="ectopic_pregnancy",
        risk_level="High",
        required_all_groups=(("pregnant", "pregnancy", "missed period"), ("lower abdomen pain", "pelvic pain", "vaginal bleeding", "fainting")),
        override_conditions=("Possible ectopic pregnancy", "Normal pregnancy"),
        recommendation="Possible ectopic pregnancy pattern: seek emergency care immediately.",
        red_flags=("possible ectopic pregnancy pattern",),
        probability_floor=0.66,
        priority=93,
    ),
    SafetyPattern(
        key="appendicitis",
        risk_level="High",
        required_all_groups=(("right lower abdomen", "right lower belly", "right lower quadrant"), ("fever", "vomiting", "loss of appetite")),
        override_conditions=("Possible appendicitis",),
        recommendation="Possible appendicitis pattern: urgent in-person evaluation is required today.",
        red_flags=("possible appendicitis pattern",),
        probability_floor=0.62,
        priority=76,
    ),
    SafetyPattern(
        key="dka",
        risk_level="High",
        required_all_groups=(("diabetes",), ("vomiting", "deep breathing", "fruity breath", "very thirsty", "confusion")),
        override_conditions=("Possible diabetic ketoacidosis", "Diabetes"),
        recommendation="Possible diabetic ketoacidosis pattern: seek emergency care immediately.",
        red_flags=("possible diabetic ketoacidosis pattern",),
        probability_floor=0.66,
        priority=88,
    ),
    SafetyPattern(
        key="anaphylaxis",
        risk_level="High",
        required_all_groups=(("allergy", "after food", "after medication", "after sting"), ("throat swelling", "lip swelling", "trouble breathing", "wheezing")),
        override_conditions=("Anaphylaxis",),
        recommendation="Possible anaphylaxis pattern: use emergency services immediately.",
        red_flags=("possible anaphylaxis pattern",),
        probability_floor=0.74,
        priority=97,
    ),
    SafetyPattern(
        key="pneumothorax",
        risk_level="High",
        required_all_groups=(("sudden chest pain", "sudden breathlessness", "sudden shortness of breath"), ("shortness of breath", "trouble breathing", "breathlessness")),
        override_conditions=("Spontaneous pneumothorax",),
        recommendation="Possible pneumothorax pattern: seek emergency evaluation immediately.",
        red_flags=("possible pneumothorax pattern",),
        probability_floor=0.62,
        priority=89,
    ),
    SafetyPattern(
        key="gi_bleed",
        risk_level="High",
        required_any=("black stool", "blood in stool", "vomiting blood", "coffee ground vomit"),
        override_conditions=("Possible gastrointestinal bleed", "Peptic ulcer diseae"),
        recommendation="Possible gastrointestinal bleeding pattern: seek urgent emergency evaluation immediately.",
        red_flags=("possible gastrointestinal bleed pattern",),
        probability_floor=0.64,
        priority=87,
    ),
    SafetyPattern(
        key="bowel_obstruction",
        risk_level="High",
        required_all_groups=(("abdominal pain", "belly pain"), ("cannot pass stool", "cannot pass gas", "vomiting", "severe bloating")),
        override_conditions=("Possible bowel obstruction",),
        recommendation="Possible bowel obstruction pattern: urgent emergency evaluation is required.",
        red_flags=("possible bowel obstruction pattern",),
        probability_floor=0.62,
        priority=82,
    ),
    SafetyPattern(
        key="epiglottitis",
        risk_level="High",
        required_all_groups=(("fever", "high fever"), ("drooling", "cannot swallow", "muffled voice", "trouble breathing")),
        override_conditions=("Epiglottitis",),
        recommendation="Possible airway emergency pattern: seek emergency care immediately.",
        red_flags=("possible airway emergency pattern",),
        probability_floor=0.70,
        priority=94,
    ),
    SafetyPattern(
        key="severe_asthma",
        risk_level="High",
        required_all_groups=(("wheezing", "asthma"), ("trouble breathing", "shortness of breath", "cannot speak full sentences")),
        override_conditions=("Bronchospasm / acute asthma exacerbation", "Bronchial Asthma"),
        recommendation="Possible severe asthma exacerbation: emergency treatment may be required immediately.",
        red_flags=("possible severe asthma pattern",),
        probability_floor=0.64,
        priority=86,
    ),
    SafetyPattern(
        key="kidney_infection",
        risk_level="Medium",
        required_all_groups=(("fever", "chills"), ("flank pain", "kidney pain", "lower back pain", "kidney")),
        override_conditions=("Urinary tract infection",),
        recommendation="Possible kidney infection pattern: arrange same-day in-person evaluation with urine testing.",
        red_flags=("possible kidney infection pattern",),
        probability_floor=0.56,
        priority=60,
    ),
    SafetyPattern(
        key="lower_uti",
        risk_level="Medium",
        required_all_groups=(("burning urination", "dysuria", "painful urination"), ("frequent urination", "frequent urine", "lower abdomen pain", "suprapubic")),
        override_conditions=("Urinary tract infection",),
        recommendation="Likely urinary tract symptom pattern: arrange same-day or next-day clinic assessment.",
        red_flags=(),
        probability_floor=0.45,
        priority=55,
    ),
    SafetyPattern(
        key="pneumonia",
        risk_level="Medium",
        required_all_groups=(("fever", "chills"), ("cough",), ("chest pain", "pain when breathing", "shortness of breath", "trouble breathing")),
        override_conditions=("Pneumonia",),
        recommendation="Possible pneumonia pattern: urgent same-day assessment is recommended.",
        red_flags=("possible pneumonia pattern",),
        probability_floor=0.52,
        priority=65,
    ),
    SafetyPattern(
        key="contact_dermatitis",
        risk_level="Low",
        required_all_groups=(("itchy", "itching", "rash"), ("chemical", "soap", "detergent", "cosmetic", "irritation")),
        override_conditions=("Allergy", "Drug Reaction"),
        recommendation="Likely irritant/allergic skin pattern: avoid the trigger and arrange routine clinical review if symptoms persist.",
        red_flags=(),
        probability_floor=0.58,
        priority=40,
    ),
    SafetyPattern(
        key="viral_uri",
        risk_level="Low",
        required_all_groups=(("sore throat", "runny nose"), ("mild cough", "cough", "low fever", "congestion")),
        override_conditions=("Common Cold", "Viral pharyngitis", "Acute viral pharyngitis (disorder)", "Viral sinusitis (disorder)"),
        recommendation="Likely mild viral upper-respiratory pattern: monitor symptoms and seek care if they worsen or persist.",
        red_flags=(),
        probability_floor=0.60,
        priority=35,
    ),
    SafetyPattern(
        key="panic_attack",
        risk_level="Low",
        required_all_groups=(("chest tightness", "fast heartbeat", "palpitations"), ("fear", "panic", "anxious")),
        override_conditions=("Panic attack",),
        recommendation="Possible panic-like pattern: seek urgent care if symptoms are new, severe, or resemble cardiac symptoms.",
        red_flags=(),
        probability_floor=0.48,
        priority=38,
    ),
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _matches(pattern: SafetyPattern, text: str) -> bool:
    if pattern.required_any and not _contains_any(text, pattern.required_any):
        return False
    for group in pattern.required_all_groups:
        if not _contains_any(text, group):
            return False
    return True


def analyze_safety_patterns(symptom_text: str) -> List[SafetyPattern]:
    text = str(symptom_text or "").strip().lower()
    return [pattern for pattern in PATTERNS if _matches(pattern, text)]


def strongest_pattern(patterns: List[SafetyPattern]) -> SafetyPattern | None:
    if not patterns:
        return None
    ranked = sorted(
        patterns,
        key=lambda item: (RISK_RANK.get(item.risk_level, 0), item.priority, item.probability_floor),
        reverse=True,
    )
    return ranked[0]


def apply_prediction_safety_overrides(symptom_text: str, predictions: List[Dict], top_k: int = 5) -> List[Dict]:
    if not predictions:
        return predictions

    patterns = analyze_safety_patterns(symptom_text)
    if not patterns:
        return predictions

    strongest = strongest_pattern(patterns)
    if strongest is None or not strongest.override_conditions:
        return predictions

    working = [
        {"condition": str(item.get("condition", "")).strip(), "probability": float(item.get("probability", 0.0))}
        for item in predictions
        if str(item.get("condition", "")).strip()
    ]
    if not working:
        return predictions

    existing = {item["condition"]: item["probability"] for item in working}
    target_conditions = list(strongest.override_conditions)
    total_target_floor = max(0.0, min(0.95, float(strongest.probability_floor)))
    share = total_target_floor / max(1, len(target_conditions))

    non_target_total = sum(prob for cond, prob in existing.items() if cond not in target_conditions)
    target_existing_total = sum(existing.get(cond, 0.0) for cond in target_conditions)
    combined_total = non_target_total + target_existing_total
    if combined_total <= 0:
        combined_total = 1.0

    scaled: Dict[str, float] = {}
    non_target_scale = max(0.0, 1.0 - total_target_floor) / max(non_target_total, 1e-12)
    for cond, prob in existing.items():
        if cond in target_conditions:
            scaled[cond] = max(prob, share)
        else:
            scaled[cond] = prob * non_target_scale

    for cond in target_conditions:
        scaled[cond] = max(scaled.get(cond, 0.0), share)

    normalized_total = sum(max(value, 0.0) for value in scaled.values())
    if normalized_total <= 0:
        return predictions

    output = [
        {"condition": cond, "probability": round(max(prob, 0.0) / normalized_total, 4)}
        for cond, prob in scaled.items()
    ]
    output.sort(key=lambda item: item["probability"], reverse=True)
    return output[:top_k]


def build_safety_summary(symptom_text: str) -> Dict:
    patterns = analyze_safety_patterns(symptom_text)
    strongest = strongest_pattern(patterns)
    return {
        "matched_patterns": [pattern.key for pattern in patterns],
        "risk_level": strongest.risk_level if strongest else None,
        "recommendation": strongest.recommendation if strongest else None,
        "red_flags": sorted({flag for pattern in patterns for flag in pattern.red_flags}),
        "override_conditions": list(strongest.override_conditions) if strongest else [],
    }
