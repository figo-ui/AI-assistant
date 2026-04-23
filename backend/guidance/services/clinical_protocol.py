from __future__ import annotations

from typing import Dict, List, Tuple


MANDATORY_DISCLAIMER = (
    "For clinical decision support only; final validation by a licensed medical professional is mandatory."
)

RESPONSE_SECTIONS = [
    "A) Provider Clinical Summary",
    "B) Longitudinal Trend Analysis",
    "C) Assessment and Differential",
    "D) Guideline-Linked Recommendations",
    "E) Safety Alerts and Contraindications",
    "F) Draft Orders/Referrals (if indicated)",
    "G) Patient-Friendly Summary",
    "H) Disclaimer",
]

HIGH_RISK_INTERACTION_PAIRS: List[Tuple[str, str, str]] = [
    ("warfarin", "ibuprofen", "Increased bleeding risk"),
    ("warfarin", "aspirin", "Increased bleeding risk"),
    ("nitrate", "sildenafil", "Severe hypotension risk"),
    ("nitroglycerin", "sildenafil", "Severe hypotension risk"),
    ("ace inhibitor", "spironolactone", "Hyperkalemia risk"),
]

GUIDELINE_HEURISTICS: List[Tuple[str, str, str]] = [
    ("chest pain", "ACC/AHA", "Chest pain triage and acute coronary syndrome evaluation"),
    ("shortness of breath", "ACC/AHA", "Heart failure evaluation and decompensation triage"),
    ("diabetes", "ADA", "Glycemic targets and antihyperglycemic treatment intensity"),
    ("copd", "GOLD", "COPD assessment and exacerbation management"),
    ("ckd", "KDIGO", "Chronic kidney disease staging and risk-based management"),
    ("fever", "IDSA", "Infectious syndromes and empiric antimicrobial stewardship"),
]


def _as_lower_text(values: List[str]) -> str:
    return " ".join(str(v).strip().lower() for v in values if str(v).strip())


def _metadata_list(metadata: Dict, key: str) -> List[str]:
    value = metadata.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _find_missing_data(metadata: Dict) -> List[str]:
    required = [
        "age",
        "medications",
        "allergies",
        "vitals_history",
        "labs_history",
    ]
    missing: List[str] = []
    for key in required:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            missing.append(key)
    return missing


def _guideline_references(case_text: str, comorbidities: List[str]) -> List[Dict[str, str]]:
    combined = f"{case_text} {_as_lower_text(comorbidities)}".lower()
    references: List[Dict[str, str]] = []
    seen = set()
    for keyword, body, topic in GUIDELINE_HEURISTICS:
        if keyword in combined and (body, topic) not in seen:
            references.append(
                {
                    "guideline": body,
                    "topic": topic,
                }
            )
            seen.add((body, topic))
    return references


def _interaction_alerts(medications: List[str], allergies: List[str]) -> List[str]:
    alerts: List[str] = []
    lower_meds = [med.lower() for med in medications]
    lower_allergies = [allergy.lower() for allergy in allergies]

    for left, right, reason in HIGH_RISK_INTERACTION_PAIRS:
        left_match = any(left in med for med in lower_meds)
        right_match = any(right in med for med in lower_meds)
        if left_match and right_match:
            alerts.append(f"Potential drug-drug interaction: {left} + {right} ({reason}).")

    for med in lower_meds:
        for allergy in lower_allergies:
            if allergy and allergy in med:
                alerts.append(
                    f"Potential drug-allergy conflict: medication '{med}' may conflict with listed allergy '{allergy}'."
                )

    return alerts


def build_clinical_report(
    symptom_text: str,
    probable_conditions: List[Dict[str, float]],
    risk_level: str,
    risk_score: float,
    confidence_band: str,
    recommendation_text: str,
    red_flags: List[str],
    metadata: Dict,
) -> Dict[str, object]:
    metadata = metadata or {}
    medications = _metadata_list(metadata, "medications")
    allergies = _metadata_list(metadata, "allergies")
    comorbidities = _metadata_list(metadata, "comorbidities")

    top_conditions = probable_conditions[:3]
    missing_data = _find_missing_data(metadata)
    guideline_refs = _guideline_references(symptom_text, comorbidities)
    safety_alerts = _interaction_alerts(medications, allergies)

    pregnancy_status = str(metadata.get("pregnancy_status", "")).strip().lower()
    if pregnancy_status in {"pregnant", "possible", "unknown"}:
        safety_alerts.append(
            "Pregnancy/lactation status requires explicit medication safety and contraindication review."
        )

    if risk_level == "High" or red_flags:
        safety_alerts.insert(0, "HIGH PRIORITY: urgent safety concern detected from risk profile/red-flag terms.")

    clinical_summary = (
        f"Current case pattern is {risk_level.lower()} risk (score {risk_score:.2f}, confidence {confidence_band}). "
        f"Top differential candidates: "
        + ", ".join(
            f"{item['condition']} ({float(item['probability']) * 100:.1f}%)" for item in top_conditions
        )
        if top_conditions
        else f"Current case pattern is {risk_level.lower()} risk (score {risk_score:.2f}, confidence {confidence_band})."
    )

    if missing_data:
        trend_line = (
            "Longitudinal trend assessment limited by missing data: "
            + ", ".join(missing_data)
            + "."
        )
    else:
        trend_line = (
            "Historical vitals/labs metadata available; compare current status against prior trajectory "
            "(improving/worsening/stable) before finalizing treatment."
        )

    assessment = {
        "primary_assessment": recommendation_text,
        "differential_diagnosis": [
            {
                "condition": item["condition"],
                "probability": float(item["probability"]),
            }
            for item in top_conditions
        ],
        "uncertainty_note": (
            "Evidence quality is constrained by confidence band and available context. "
            "Validate with clinical examination and definitive diagnostics."
        ),
    }

    guideline_recommendations = (
        guideline_refs
        if guideline_refs
        else [
            {
                "guideline": "No condition-specific trigger matched from provided inputs",
                "topic": "Use local standard-of-care pathways and complete diagnostic workup.",
            }
        ]
    )

    draft_orders: List[Dict[str, object]] = []
    if risk_level in {"Medium", "High"} or confidence_band == "low":
        urgency = "urgent" if risk_level == "High" else "routine"
        draft_orders = [
            {
                "type": "clinical_summary",
                "status": "Pending provider sign-off.",
                "content": (
                    "Patient evaluated via AI-supported triage summary with risk-stratified differential; "
                    "provider to validate history, exam, and diagnostics."
                ),
            },
            {
                "type": "referral_order",
                "status": "Pending provider sign-off.",
                "specialty": "Internal Medicine",
                "urgency": urgency,
                "indication": "Risk-stratified follow-up and diagnostic confirmation.",
            },
            {
                "type": "prescription_or_test_order",
                "status": "Pending provider sign-off.",
                "template": {
                    "drug_or_test": "[to be determined by provider]",
                    "dose_or_panel": "[to be determined by provider]",
                    "frequency": "[to be determined by provider]",
                    "duration": "[to be determined by provider]",
                },
            },
        ]

    patient_summary = (
        "Your care team reviewed your symptoms with an AI support tool. "
        "This result suggests possible causes and next steps, but it is not a final diagnosis. "
        "Please follow up with your clinician, especially if symptoms worsen."
    )

    return {
        "provider_clinical_summary": clinical_summary,
        "longitudinal_trend_analysis": trend_line,
        "assessment_and_differential": assessment,
        "guideline_linked_recommendations": guideline_recommendations,
        "safety_alerts_and_contraindications": safety_alerts,
        "draft_orders_referrals": draft_orders,
        "patient_friendly_summary": patient_summary,
        "disclaimer": MANDATORY_DISCLAIMER,
    }
