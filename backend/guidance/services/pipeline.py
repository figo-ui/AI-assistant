import time
import re
from pathlib import Path
from typing import Dict, List

from django.conf import settings

from ..models import CaseSubmission, FacilityResult, InferenceRecord, RiskAssessment
from .clinical_safety import apply_prediction_safety_overrides
from .clinical_protocol import RESPONSE_SECTIONS, build_clinical_report
from .facilities import lookup_nearby_facilities
from .fusion import fuse_predictions
from .image_model import predict_image_probabilities
from .label_mapping import map_prediction_list
from .language_support import detect_language, localize_analysis_result, normalize_text_for_models
from .llm_triage import llm_available, predict_with_llm
from .pii_redaction import redact_phi_text
from .risk import compute_risk
from .search_router import run_search_router
from .rag import build_rag_context, should_use_rag
from .text_model import predict_text_probabilities

GENERIC_CONDITION_RE = re.compile(r"^(condition\s+\d+|class_\d+)$", re.IGNORECASE)

MODEL_PROFILE_CONFIG = {
    "clinical fast": {
        "top_k": 3,
        "force_search": False,
        "prefer_llm": False,
        "search_max_results": 3,
    },
    "clinical balanced": {
        "top_k": 5,
        "force_search": False,
        "prefer_llm": None,
        "search_max_results": 5,
    },
    "clinical thorough": {
        "top_k": 7,
        "force_search": True,
        "prefer_llm": True,
        "search_max_results": 7,
    },
}


def _resolve_model_profile(metadata: Dict) -> tuple[str, Dict[str, object]]:
    profile = str((metadata or {}).get("model_profile", "Clinical Balanced")).strip().lower()
    if profile not in MODEL_PROFILE_CONFIG:
        profile = "clinical balanced"
    return profile, MODEL_PROFILE_CONFIG[profile]


def _is_generic_condition(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return True
    return bool(GENERIC_CONDITION_RE.fullmatch(text))


def _to_user_facing_predictions(predictions: List[Dict[str, float]]) -> List[Dict[str, float]]:
    named = [item for item in predictions if not _is_generic_condition(str(item.get("condition", "")))]
    if named:
        return named

    if not predictions:
        return []

    fallback = dict(predictions[0])
    fallback["condition"] = "Unspecified clinical pattern"
    return [fallback]


def _vulnerability_from_metadata(metadata: Dict) -> float:
    if not metadata:
        return 0.0

    score = 0.0
    age = metadata.get("age")
    if isinstance(age, (int, float)) and age >= 65:
        score += 0.4

    comorbidities = metadata.get("comorbidities", [])
    if isinstance(comorbidities, list):
        score += min(0.6, 0.2 * len(comorbidities))

    return max(0.0, min(1.0, score))


def _select_text_inference(
    symptom_text: str,
    *,
    language: str,
    search_context: Dict,
    top_k: int,
    prefer_llm: bool | None,
) -> Dict:
    llm_enabled = bool(getattr(settings, "USE_LLM_TRIAGE", False))
    fallback_enabled = bool(getattr(settings, "LLM_FALLBACK_TO_CLASSICAL", True))
    if prefer_llm is False:
        llm_enabled = False

    if llm_enabled and llm_available():
        llm_output = predict_with_llm(
            symptom_text,
            top_k=top_k,
            language=language,
            search_context=search_context,
        )
        if llm_output.get("available") and llm_output.get("predictions"):
            return {
                "predictions": llm_output["predictions"],
                "confidence": llm_output["predictions"][0]["probability"] if llm_output["predictions"] else 0.0,
                "model_version": "triage-llm-v1",
                "fallback_used": False,
            }

    classical = predict_text_probabilities(symptom_text, top_k=top_k)
    if llm_enabled and fallback_enabled:
        return {
            **classical,
            "model_version": f"{classical['model_version']}+llm-fallback",
            "fallback_used": True,
        }
    return {**classical, "fallback_used": False}


def _build_conversation_context(case: CaseSubmission, window: int = 5) -> str:
    """Return a brief summary of recent chat messages for context injection."""
    if not case.chat_session_id:
        return ""
    from ..models import ChatMessage  # local import to avoid circular
    recent = (
        ChatMessage.objects.filter(session_id=case.chat_session_id)
        .exclude(id__in=ChatMessage.objects.filter(
            session_id=case.chat_session_id,
            role=ChatMessage.Role.USER,
            content=case.symptom_text,
        ).values_list("id", flat=True)[:1])
        .order_by("-created_at")[:window]
    )
    lines = []
    for msg in reversed(list(recent)):
        role_label = "Patient" if msg.role == "user" else "Assistant"
        snippet = (msg.content or "")[:200].strip()
        if snippet:
            lines.append(f"{role_label}: {snippet}")
    return "\n".join(lines)


def run_case_analysis(case: CaseSubmission) -> Dict:
    start = time.monotonic()
    profile_name, profile_cfg = _resolve_model_profile(case.metadata)
    top_k = int(profile_cfg.get("top_k", 5))
    prefer_llm = profile_cfg.get("prefer_llm", None)
    profile_force_search = bool(profile_cfg.get("force_search", False))
    search_max_results = int(profile_cfg.get("search_max_results", 5))

    response_language = detect_language(
        case.symptom_text,
        preferred=str((case.metadata or {}).get("language", "")).strip().lower() or None,
    )
    analysis_text = normalize_text_for_models(case.symptom_text, response_language)

    # Inject conversation context window (last 5 turns) to improve continuity
    conversation_context = _build_conversation_context(case, window=5)
    if conversation_context:
        analysis_text = f"[Prior conversation context]\n{conversation_context}\n\n[Current message]\n{analysis_text}"

    redaction = redact_phi_text(analysis_text)
    search_context = run_search_router(
        case.symptom_text,
        translated_query=str(redaction.get("redacted_text", "")),
        force_search=bool((case.metadata or {}).get("force_search", False)) or profile_force_search,
        search_consent=bool((case.metadata or {}).get("search_consent_given", False)),
        mock_results=(case.metadata or {}).get("mock_search_results"),
        max_results=search_max_results,
    )
    search_context["redaction"] = {
        "entities": list(redaction.get("entities", [])),
        "presidio_used": bool(redaction.get("presidio_used", False)),
        "external_query": str(redaction.get("redacted_text", "")),
    }
    rag_context = {}
    if should_use_rag(case.symptom_text):
        rag_context = build_rag_context(
            case.symptom_text,
            top_k=max(3, min(6, top_k)),
            search_context=search_context,
        )
    text_output = _select_text_inference(
        analysis_text,
        language=response_language,
        search_context=search_context,
        top_k=top_k,
        prefer_llm=prefer_llm,
    )

    image_output = {
        "predictions": [],
        "confidence": 0.0,
        "model_version": "image-missing-v2",
        "quality_score": 0.0,
        "scope": "not_available",
        "limited_scope": True,
    }
    if case.uploaded_image:
        image_output = predict_image_probabilities(Path(case.uploaded_image.path), top_k=top_k)

    fused = fuse_predictions(
        text_predictions=text_output["predictions"],
        image_predictions=image_output["predictions"],
        text_confidence=float(text_output["confidence"]),
        image_confidence=float(image_output["confidence"]),
        image_quality=float(image_output.get("quality_score", 1.0)),
        top_k=top_k,
    )

    mapped_text_predictions = map_prediction_list(text_output["predictions"])
    mapped_image_predictions = map_prediction_list(image_output["predictions"])
    mapped_fused_predictions = map_prediction_list(fused["predictions"])
    mapped_fused_predictions = apply_prediction_safety_overrides(
        analysis_text,
        mapped_fused_predictions,
        top_k=top_k,
    )
    user_facing_fused_predictions = _to_user_facing_predictions(mapped_fused_predictions)

    risk = compute_risk(
        fused_predictions=mapped_fused_predictions,
        uncertainty=float(fused["uncertainty"]),
        disagreement=float(fused["disagreement"]),
        symptom_text=analysis_text,
        confidence_band=fused["confidence_band"],
        vulnerability=_vulnerability_from_metadata(case.metadata),
    )
    clinical_report = build_clinical_report(
        symptom_text=analysis_text,
        probable_conditions=mapped_fused_predictions,
        risk_level=risk["risk_level"],
        risk_score=float(risk["risk_score"]),
        confidence_band=fused["confidence_band"],
        recommendation_text=risk["recommendation_text"],
        red_flags=risk["red_flags"],
        metadata=case.metadata,
    )

    facilities = lookup_nearby_facilities(
        location_lat=case.location_lat,
        location_lng=case.location_lng,
        facility_type=case.facility_type_requested or "hospital",
        specialization=case.specialization_requested or "",
        radius_km=case.search_radius_km,
        limit=5,
    )

    # REQ-8: Auto-escalate to emergency facility search when risk is High
    if risk["risk_level"] == "High" and case.location_lat and case.location_lng:
        emergency_facilities = lookup_nearby_facilities(
            location_lat=case.location_lat,
            location_lng=case.location_lng,
            facility_type="emergency",
            specialization="",
            radius_km=case.search_radius_km,
            limit=3,
        )
        # Prepend emergency facilities, deduplicate by provider_name
        seen_names = {f.get("provider_name") for f in emergency_facilities}
        non_dupe = [f for f in facilities if f.get("provider_name") not in seen_names]
        facilities = emergency_facilities + non_dupe
    latency_ms = int((time.monotonic() - start) * 1000)

    InferenceRecord.objects.update_or_create(
        case=case,
        defaults={
            "text_predictions": mapped_text_predictions,
            "image_predictions": mapped_image_predictions,
            "fused_predictions": mapped_fused_predictions,
            "text_confidence": text_output["confidence"],
            "image_confidence": image_output["confidence"],
            "fusion_confidence": fused["confidence"],
            "confidence_band": fused["confidence_band"],
            "text_model_version": text_output["model_version"],
            "image_model_version": image_output["model_version"],
            "fusion_version": fused["version"],
            "uncertainty": fused["uncertainty"],
            "disagreement": fused["disagreement"],
            "latency_ms": latency_ms,
        },
    )

    RiskAssessment.objects.update_or_create(
        case=case,
        defaults=risk,
    )

    case.facilities.all().delete()
    if facilities:
        FacilityResult.objects.bulk_create(
            [FacilityResult(case=case, **facility) for facility in facilities]
        )

    case.status = "completed"
    case.save(update_fields=["status"])

    # REQ-8: Log emergency incidents for High-risk cases
    if risk["risk_level"] == "High" or risk.get("needs_urgent_care"):
        try:
            from ..models import AuditLog
            AuditLog.objects.create(
                actor=None,
                action="emergency_case_flagged",
                target_type="case",
                target_id=str(case.id),
                metadata={
                    "risk_level": risk["risk_level"],
                    "risk_score": float(risk["risk_score"]),
                    "red_flags": list(risk.get("red_flags", [])),
                    "user_id": case.user_id,
                    "needs_urgent_care": bool(risk.get("needs_urgent_care")),
                },
            )
        except Exception:
            pass  # never block the response for logging

    result = {
        "case_id": case.id,
        "created_at": case.created_at.isoformat(),
        "probable_conditions": user_facing_fused_predictions,
        "raw_probable_conditions": mapped_fused_predictions,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "confidence_band": fused["confidence_band"],
        "recommendation_text": risk["recommendation_text"],
        "prevention_advice": risk["prevention_advice"],
        "disclaimer_text": risk["disclaimer_text"],
        "risk_factors": risk["risk_factors"],
        "red_flags": risk["red_flags"],
        "needs_urgent_care": risk["needs_urgent_care"],
        "response_format_sections": RESPONSE_SECTIONS,
        "clinical_report": clinical_report,
        "modality_predictions": {
            "text": mapped_text_predictions,
            "image": mapped_image_predictions,
        },
        "fusion_details": {
            "modality_weights": fused.get("modality_weights", {}),
            "agreement": fused.get("agreement"),
            "image_quality": image_output.get("quality_score", 0.0),
            "image_scope": image_output.get("scope", "not_available"),
            "image_limited_scope": bool(image_output.get("limited_scope", True)),
            "text_fallback_used": bool(text_output.get("fallback_used", False)),
        },
        "search_context": search_context,
        "rag_context": rag_context,
        "nearby_facilities": facilities,
        "emergency_auto_triggered": risk["risk_level"] == "High" and bool(case.location_lat),
        "model_versions": {
            "text": text_output["model_version"],
            "image": image_output["model_version"],
            "fusion": fused["version"],
        },
        "model_profile": profile_name,
        "latency_ms": latency_ms,
    }
    return localize_analysis_result(result, response_language)
