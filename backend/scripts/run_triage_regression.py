import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare_ai.settings")

import django  # noqa: E402

django.setup()

from guidance.services.fusion import fuse_predictions  # noqa: E402
from guidance.services.language_support import detect_language, normalize_text_for_models  # noqa: E402
from guidance.services.pii_redaction import redact_phi_text  # noqa: E402
from guidance.services.risk import compute_risk  # noqa: E402
from guidance.services.search_router import run_search_router  # noqa: E402
from guidance.services.text_model import predict_text_probabilities  # noqa: E402


MOCK_SEARCH_RESULTS = [
    {
        "title": "Mock 2026 Ethiopia malaria update",
        "url": "https://example.org/malaria-ethiopia-2026",
        "snippet": "Illustrative outbreak and treatment guidance result.",
        "source": "MockSearch",
    },
    {
        "title": "Mock current clinical guideline summary",
        "url": "https://example.org/current-guideline",
        "snippet": "Illustrative current-guideline source for local regression.",
        "source": "MockSearch",
    },
]


BASE_CASES = [
    {"name": "kidney_fever_persistent", "input": "i feel pain on my back down and minimum fever for long time around kidney", "expected_top3": ["Urinary tract infection"], "expected_risk": "Medium", "emergency": False},
    {"name": "lower_uti", "input": "burning urination with lower abdomen pain and frequent urine", "expected_top3": ["Urinary tract infection"], "expected_risk": "Medium", "emergency": False},
    {"name": "stroke_fast", "input": "severe headache with one side weakness and slurred speech", "expected_top3": ["Stroke", "Paralysis (brain hemorrhage)"], "expected_risk": "High", "emergency": True},
    {"name": "acs", "input": "chest pain with trouble breathing and sweating", "expected_top3": ["Heart attack", "Unstable angina", "Possible NSTEMI / STEMI"], "expected_risk": "High", "emergency": True},
    {"name": "chemical_rash", "input": "itchy red skin rash on my hand after chemical exposure", "expected_top3": ["Allergy", "Drug Reaction"], "expected_risk": "Low", "emergency": False},
    {"name": "pneumonia", "input": "fever cough chest pain when breathing", "expected_top3": ["Pneumonia"], "expected_risk": "Medium", "emergency": False},
    {"name": "viral_uri", "input": "sore throat runny nose mild cough and low fever", "expected_top3": ["Common Cold", "Viral pharyngitis", "Acute viral pharyngitis (disorder)", "Viral sinusitis (disorder)"], "expected_risk": "Low", "emergency": False},
    {"name": "panic_attack_like", "input": "sudden chest tightness with fear and fast heartbeat but no fever", "expected_top3": ["Panic attack"], "expected_risk": "Low", "emergency": False},
    {"name": "anaphylaxis", "input": "allergy after food with lip swelling throat swelling and trouble breathing", "expected_top3": ["Anaphylaxis"], "expected_risk": "High", "emergency": True},
    {"name": "meningitis", "input": "high fever severe headache stiff neck confusion and light sensitivity", "expected_top3": ["Possible meningitis"], "expected_risk": "High", "emergency": True},
    {"name": "appendicitis", "input": "right lower abdomen pain with fever vomiting and loss of appetite", "expected_top3": ["Possible appendicitis"], "expected_risk": "High", "emergency": True},
    {"name": "ectopic_pregnancy", "input": "missed period with pregnancy lower abdominal pain vaginal bleeding and fainting", "expected_top3": ["Possible ectopic pregnancy", "Normal pregnancy"], "expected_risk": "High", "emergency": True},
    {"name": "dka", "input": "diabetes with vomiting deep breathing fruity breath and confusion", "expected_top3": ["Possible diabetic ketoacidosis", "Diabetes"], "expected_risk": "High", "emergency": True},
    {"name": "gi_bleed", "input": "black stool and vomiting blood with dizziness", "expected_top3": ["Possible gastrointestinal bleed", "Peptic ulcer diseae"], "expected_risk": "High", "emergency": True},
    {"name": "bowel_obstruction", "input": "severe abdominal pain with vomiting cannot pass gas and severe bloating", "expected_top3": ["Possible bowel obstruction"], "expected_risk": "High", "emergency": True},
    {"name": "epiglottitis", "input": "high fever drooling muffled voice cannot swallow and trouble breathing", "expected_top3": ["Epiglottitis"], "expected_risk": "High", "emergency": True},
    {"name": "pulmonary_embolism", "input": "chest pain shortness of breath with calf pain and leg swelling", "expected_top3": ["Pulmonary embolism", "Acute pulmonary embolism (disorder)"], "expected_risk": "High", "emergency": True},
    {"name": "pneumothorax", "input": "sudden chest pain with sudden breathlessness and trouble breathing", "expected_top3": ["Spontaneous pneumothorax"], "expected_risk": "High", "emergency": True},
    {"name": "sepsis", "input": "fever chills infection with confusion and very weak low blood pressure", "expected_top3": ["Sepsis caused by virus (disorder)"], "expected_risk": "High", "emergency": True},
    {"name": "severe_asthma", "input": "asthma wheezing and trouble breathing cannot speak full sentences", "expected_top3": ["Bronchospasm / acute asthma exacerbation", "Bronchial Asthma"], "expected_risk": "High", "emergency": True},
    {
        "name": "melanoma_photo",
        "input": "dark irregular mole getting larger and changing color",
        "expected_top3": ["melanoma"],
        "expected_risk": "High",
        "emergency": False,
        "image_predictions": [
            {"condition": "melanoma", "probability": 0.88},
            {"condition": "melanocytic nevus", "probability": 0.08},
        ],
        "min_image_weight": 0.35,
    },
    {
        "name": "contact_dermatitis_photo",
        "input": "itchy red rash after detergent and chemical exposure on the hands",
        "expected_top3": ["contact dermatitis", "chemical burn"],
        "expected_risk": "Low",
        "emergency": False,
        "image_predictions": [
            {"condition": "contact dermatitis", "probability": 0.72},
            {"condition": "chemical burn", "probability": 0.18},
        ],
        "min_image_weight": 0.25,
    },
    {
        "name": "cellulitis_photo",
        "input": "painful warm spreading red area on leg with fever",
        "expected_top3": ["cellulitis"],
        "expected_risk": "Medium",
        "emergency": False,
        "image_predictions": [
            {"condition": "cellulitis", "probability": 0.81},
            {"condition": "contact dermatitis", "probability": 0.1},
        ],
        "min_image_weight": 0.3,
    },
    {
        "name": "psoriasis_photo",
        "input": "chronic dry thick scaly plaques on elbows and knees",
        "expected_top3": ["psoriasis"],
        "expected_risk": "Low",
        "emergency": False,
        "image_predictions": [
            {"condition": "psoriasis", "probability": 0.79},
            {"condition": "eczema", "probability": 0.11},
        ],
        "min_image_weight": 0.3,
    },
    {
        "name": "fungal_ring_photo",
        "input": "itchy circular rash with scaling that keeps spreading",
        "expected_top3": ["fungal infection"],
        "expected_risk": "Low",
        "emergency": False,
        "image_predictions": [
            {"condition": "fungal infection", "probability": 0.77},
            {"condition": "eczema", "probability": 0.13},
        ],
        "min_image_weight": 0.3,
    },
]


AMHARIC_CASES = [
    {"name": "am_heart_attack", "input": "የደረት ህመም እና የመተንፈስ ችግር አለኝ", "expected_top3": ["Heart attack", "Possible NSTEMI / STEMI"], "expected_risk": "High", "emergency": True, "language": "am"},
    {"name": "am_stroke", "input": "ከፍተኛ ትኩሳት ሳይሆን አንድ ጎን ድካም እና ንግግር ችግር አለኝ", "expected_top3": ["Stroke", "Paralysis (brain hemorrhage)"], "expected_risk": "High", "emergency": True, "language": "am"},
    {"name": "am_uti", "input": "ማቃጠል ሽንት እና ትኩሳት አለኝ", "expected_top3": ["Urinary tract infection"], "expected_risk": "Medium", "emergency": False, "language": "am"},
    {"name": "am_viral_uri", "input": "የጉሮሮ ህመም እና አፍንጫ ፍሳሽ እና ሳል አለኝ", "expected_top3": ["Common Cold", "Viral pharyngitis"], "expected_risk": "Low", "emergency": False, "language": "am"},
    {"name": "am_chemical_rash", "input": "ሽፍታ እና ማሳከክ ከኬሚካል በኋላ ተከሰተ", "expected_top3": ["Allergy", "Drug Reaction"], "expected_risk": "Low", "emergency": False, "language": "am"},
]


SEARCH_CASES = [
    {"name": "search_malaria_ethiopia", "input": "latest malaria Ethiopia 2026 high fever chills headache after mosquito bites", "expected_top3": ["Malaria"], "expected_risk": "Medium", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS},
    {"name": "search_pneumonia_guideline", "input": "current pneumonia guideline 2026 fever cough chest pain when breathing", "expected_top3": ["Pneumonia"], "expected_risk": "Medium", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS},
    {"name": "search_stroke_guideline", "input": "latest stroke guideline 2026 severe headache one side weakness and slurred speech", "expected_top3": ["Stroke", "Paralysis (brain hemorrhage)"], "expected_risk": "High", "emergency": True, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS},
    {"name": "search_meningitis_outbreak", "input": "current outbreak meningitis Ethiopia 2026 high fever severe headache stiff neck confusion", "expected_top3": ["Possible meningitis"], "expected_risk": "High", "emergency": True, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS},
    {"name": "search_dermatitis_update", "input": "new current guideline itchy red rash after chemical exposure on hands", "expected_top3": ["Allergy", "Drug Reaction"], "expected_risk": "Low", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS},
    {"name": "search_redacted_name", "input": "my name is John Doe latest malaria Ethiopia 2026 fever chills headache", "expected_top3": ["Malaria"], "expected_risk": "Medium", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS, "expected_redaction_entities": ["NAME"]},
    {"name": "search_redacted_email", "input": "email me at patient@example.com current pneumonia guideline 2026 fever cough chest pain", "expected_top3": ["Pneumonia"], "expected_risk": "Medium", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS, "expected_redaction_entities": ["EMAIL_ADDRESS"]},
    {"name": "search_redacted_phone", "input": "call 202-555-0189 latest stroke guideline 2026 severe headache one side weakness slurred speech", "expected_top3": ["Stroke", "Paralysis (brain hemorrhage)"], "expected_risk": "High", "emergency": True, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS, "expected_redaction_entities": ["PHONE_NUMBER"]},
    {"name": "search_amharic_guideline", "input": "ወባ ኢትዮጵያ latest 2026 ትኩሳት እና ብርድ ብርድ", "expected_top3": ["Malaria"], "expected_risk": "Medium", "emergency": False, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS, "language": "am"},
    {"name": "search_redacted_date", "input": "on 03/08/2026 I had latest meningitis outbreak Ethiopia high fever severe headache stiff neck confusion", "expected_top3": ["Possible meningitis"], "expected_risk": "High", "emergency": True, "force_search": True, "search_consent": True, "mock_search_results": MOCK_SEARCH_RESULTS, "expected_redaction_entities": ["DATE"]},
]


CASES = BASE_CASES + AMHARIC_CASES + SEARCH_CASES


def _contains_expected(top3: List[str], expected: List[str]) -> bool:
    lower_top3 = [item.lower() for item in top3]
    for target in expected:
        target_lower = target.lower()
        if any(target_lower in candidate or candidate in target_lower for candidate in lower_top3):
            return True
    return False


def _run_case(case: Dict) -> Tuple[Dict, bool, bool, bool, bool]:
    detected_language = detect_language(case["input"], preferred=case.get("language"))
    analysis_text = normalize_text_for_models(case["input"], detected_language)
    redaction = redact_phi_text(analysis_text)
    search_context = run_search_router(
        case["input"],
        translated_query=str(redaction.get("redacted_text", "")),
        force_search=bool(case.get("force_search", False)),
        search_consent=bool(case.get("search_consent", False)),
        mock_results=case.get("mock_search_results"),
        max_results=5,
    )

    text = predict_text_probabilities(analysis_text, top_k=5)
    image_predictions = case.get("image_predictions")
    image_output = {"predictions": [], "confidence": 0.0, "quality_score": 0.0, "model_version": "none"}
    if image_predictions:
        ordered = sorted(image_predictions, key=lambda item: item["probability"], reverse=True)
        image_output = {
            "predictions": ordered,
            "confidence": float(ordered[0]["probability"]) if ordered else 0.0,
            "quality_score": 0.95,
            "model_version": "regression-mock-image-v1",
        }

    fused = fuse_predictions(
        text_predictions=text["predictions"],
        image_predictions=image_output["predictions"],
        text_confidence=float(text["confidence"]),
        image_confidence=float(image_output["confidence"]),
        image_quality=float(image_output.get("quality_score", 1.0)),
        top_k=5,
    )
    risk = compute_risk(
        fused["predictions"],
        uncertainty=max(0.0, 1.0 - float(fused["confidence"])),
        disagreement=float(fused["disagreement"]),
        symptom_text=analysis_text,
        confidence_band=fused["confidence_band"],
    )

    top3 = [item["condition"] for item in fused["predictions"][:3]]
    hit_top3 = _contains_expected(top3, case["expected_top3"])
    hit_risk = risk["risk_level"] == case["expected_risk"]
    image_weight = float(fused.get("modality_weights", {}).get("image", 0.0))
    image_pass = image_weight >= float(case.get("min_image_weight", 0.0))
    search_required = bool(case.get("force_search", False))
    search_pass = (not search_required) or bool(search_context.get("results"))
    redaction_required = case.get("expected_redaction_entities", [])
    redaction_pass = all(item in set(redaction.get("entities", [])) for item in redaction_required)

    return (
        {
            "name": case["name"],
            "input": case["input"],
            "analysis_text": analysis_text,
            "detected_language": detected_language,
            "top3": top3,
            "text_top3": [item["condition"] for item in text["predictions"][:3]],
            "confidence": fused["confidence"],
            "risk_level": risk["risk_level"],
            "needs_urgent_care": risk["needs_urgent_care"],
            "recommendation": risk["recommendation_text"],
            "expected_top3": case["expected_top3"],
            "expected_risk": case["expected_risk"],
            "top3_pass": hit_top3,
            "risk_pass": hit_risk,
            "image_pass": image_pass,
            "search_pass": search_pass,
            "redaction_pass": redaction_pass,
            "fusion_version": fused["version"],
            "image_model_version": image_output.get("model_version"),
            "modality_weights": fused.get("modality_weights", {}),
            "search_sources": search_context.get("sources", []),
            "redaction_entities": redaction.get("entities", []),
            "external_query": redaction.get("redacted_text", ""),
        },
        hit_top3,
        hit_risk,
        image_pass,
        search_pass and redaction_pass,
    )


def main() -> int:
    results = []
    passed_top3 = 0
    passed_risk = 0
    passed_image = 0
    passed_search = 0
    image_cases = 0
    search_cases = 0
    emergency_cases = 0
    emergency_high = 0
    covid_unrelated_hits = 0
    top_probs = []

    for case in CASES:
        result, hit_top3, hit_risk, image_pass, search_pass = _run_case(case)
        top3 = result["top3"]
        top_probs.append(float(result["confidence"]))
        passed_top3 += int(hit_top3)
        passed_risk += int(hit_risk)

        if "image_predictions" in case:
            image_cases += 1
            passed_image += int(image_pass)
        if case.get("force_search", False):
            search_cases += 1
            passed_search += int(search_pass)

        if case["emergency"]:
            emergency_cases += 1
            if result["risk_level"] == "High" or result["needs_urgent_care"]:
                emergency_high += 1

        unrelated = not any(term in case["input"].lower() for term in ["cough", "runny nose", "sore throat", "loss of smell", "loss of taste"])
        if unrelated and any("covid" in item.lower() for item in top3):
            covid_unrelated_hits += 1

        results.append(result)

    summary = {
        "cases": len(CASES),
        "top3_pass_count": passed_top3,
        "risk_pass_count": passed_risk,
        "image_pass_count": passed_image,
        "search_pass_count": passed_search,
        "image_cases": image_cases,
        "search_cases": search_cases,
        "top3_pass_rate": round(passed_top3 / len(CASES), 4),
        "risk_pass_rate": round(passed_risk / len(CASES), 4),
        "image_pass_rate": round(passed_image / max(1, image_cases), 4),
        "search_pass_rate": round(passed_search / max(1, search_cases), 4),
        "mean_top_probability": round(sum(top_probs) / len(top_probs), 4),
        "emergency_cases": emergency_cases,
        "emergency_flag_high_or_urgent": emergency_high,
        "emergency_flag_rate": round(emergency_high / max(1, emergency_cases), 4),
        "covid_bias_unrelated_top3_hits": covid_unrelated_hits,
        "results": results,
    }

    out_path = ROOT / "models" / "triage_regression_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
