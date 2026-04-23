from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple


HIPAA_IDENTIFIER_LABELS = [
    "NAME",
    "GEOGRAPHIC_LOCATION",
    "DATE",
    "PHONE_NUMBER",
    "FAX_NUMBER",
    "EMAIL_ADDRESS",
    "SSN",
    "MRN",
    "HEALTH_PLAN_BENEFICIARY_NUMBER",
    "ACCOUNT_NUMBER",
    "CERTIFICATE_OR_LICENSE_NUMBER",
    "VEHICLE_IDENTIFIER",
    "DEVICE_IDENTIFIER",
    "WEB_URL",
    "IP_ADDRESS",
    "BIOMETRIC_IDENTIFIER",
    "FULL_FACE_PHOTO_REFERENCE",
    "UNIQUE_IDENTIFIER",
]

REGEX_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("PHONE_NUMBER", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:mrn|medical record number)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("ACCOUNT_NUMBER", re.compile(r"\b(?:account|acct)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("HEALTH_PLAN_BENEFICIARY_NUMBER", re.compile(r"\b(?:member id|policy number|insurance id)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("CERTIFICATE_OR_LICENSE_NUMBER", re.compile(r"\b(?:license|certificate)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("VEHICLE_IDENTIFIER", re.compile(r"\b(?:plate|license plate|vin)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("DEVICE_IDENTIFIER", re.compile(r"\b(?:device id|serial number|imei)[:\s#-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)),
    ("WEB_URL", re.compile(r"\bhttps?://[^\s]+|\bwww\.[^\s]+\b", re.IGNORECASE)),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?\b", re.IGNORECASE)),
    ("UNIQUE_IDENTIFIER", re.compile(r"\b(?:id|identifier|ref|reference)[:\s#-]*[A-Z0-9]{6,}\b", re.IGNORECASE)),
    ("FULL_FACE_PHOTO_REFERENCE", re.compile(r"\b(?:selfie|portrait|face photo|passport photo)\b", re.IGNORECASE)),
    ("BIOMETRIC_IDENTIFIER", re.compile(r"\b(?:fingerprint|retina|iris scan|voiceprint)\b", re.IGNORECASE)),
    ("FAX_NUMBER", re.compile(r"\b(?:fax)[:\s#-]*\+?\d[\d\s.-]{5,}\b", re.IGNORECASE)),
    ("GEOGRAPHIC_LOCATION", re.compile(r"\b\d{1,5}\s+[A-Z0-9][A-Z0-9\s.-]{2,}\b", re.IGNORECASE)),
    ("NAME", re.compile(r"\b(?:my name is|patient name is|i am)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")),
]


def _replacement(label: str) -> str:
    return f"[REDACTED_{label}]"


@lru_cache(maxsize=1)
def _presidio_engines():
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except Exception:
        return None, None

    try:
        return AnalyzerEngine(), AnonymizerEngine()
    except Exception:
        return None, None


def _redact_with_presidio(text: str) -> tuple[str, List[str]]:
    analyzer, anonymizer = _presidio_engines()
    if analyzer is None or anonymizer is None or not text.strip():
        return text, []

    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "DATE_TIME",
                "LOCATION",
                "US_SSN",
                "URL",
                "IP_ADDRESS",
                "CREDIT_CARD",
                "MEDICAL_LICENSE",
            ],
        )
    except Exception:
        return text, []

    if not results:
        return text, []

    operators = {
        item.entity_type: {"type": "replace", "new_value": _replacement(item.entity_type)}
        for item in results
    }
    try:
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    except Exception:
        return text, []

    return anonymized.text, sorted({item.entity_type for item in results})


def _redact_with_regex(text: str) -> tuple[str, List[str]]:
    redacted = text
    entities: List[str] = []
    for label, pattern in REGEX_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(_replacement(label), redacted)
            entities.append(label)
    return redacted, sorted(set(entities))


def redact_phi_text(text: str) -> Dict[str, object]:
    working = str(text or "").strip()
    if not working:
        return {
            "original_text": "",
            "redacted_text": "",
            "entities": [],
            "presidio_used": False,
        }

    presidio_redacted, presidio_entities = _redact_with_presidio(working)
    regex_redacted, regex_entities = _redact_with_regex(presidio_redacted)
    return {
        "original_text": working,
        "redacted_text": regex_redacted,
        "entities": sorted(set(presidio_entities + regex_entities)),
        "presidio_used": bool(presidio_entities),
    }
