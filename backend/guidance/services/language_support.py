import re
from typing import Dict, List, Optional


AMHARIC_SCRIPT_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F]")
WHITESPACE_RE = re.compile(r"\s+")

SUPPORTED_LANGUAGES = {"en", "am"}

COMMON_STRINGS: Dict[str, Dict[str, str]] = {
    "disclaimer": {
        "en": "This is not medical advice.",
        "am": "ይህ የሕክምና ምክር አይደለም።",
    },
    "seek_professional_care": {
        "en": "Seek professional care.",
        "am": "የሙያ ሕክምና እርዳታ ይፈልጉ።",
    },
    "risk_low": {"en": "Low", "am": "ዝቅተኛ"},
    "risk_medium": {"en": "Medium", "am": "መካከለኛ"},
    "risk_high": {"en": "High", "am": "ከፍተኛ"},
    "current_sources_checked": {
        "en": "Current sources checked",
        "am": "የተመረመሩ ወቅታዊ ምንጮች",
    },
    "freshness_note": {
        "en": "This answer used current search results because the query needs up-to-date information.",
        "am": "ይህ መልስ ወቅታዊ መረጃ ስለሚፈልግ አዳዲስ የፍለጋ ውጤቶችን ተጠቅሟል።",
    },
    "monitor": {
        "en": "Monitor symptoms and seek care if they worsen or persist.",
        "am": "ምልክቶቹን ይከታተሉ፤ ከባድ ከሆኑ ወይም ከቀጠሉ ሕክምና ይፈልጉ።",
    },
    "same_day": {
        "en": "Arrange same-day or next-day clinical care.",
        "am": "በዚያው ቀን ወይም በሚቀጥለው ቀን ክሊኒካዊ እንክብካቤ ያዘጋጁ።",
    },
    "urgent": {
        "en": "Seek urgent in-person or emergency care now.",
        "am": "አሁኑኑ አስቸኳይ የቀጥታ ወይም የድንገተኛ ሕክምና ይፈልጉ።",
    },
    "assistant_open_low": {
        "en": "Thanks for sharing your symptoms.",
        "am": "ምልክቶችዎን ስለገለጹ እናመሰግናለን።",
    },
    "assistant_open_medium": {
        "en": "Thank you for explaining your symptoms clearly.",
        "am": "ምልክቶችዎን በግልፅ ሁኔታ ስለገለጹ እናመሰግናለን።",
    },
    "assistant_open_high": {
        "en": "Thank you for sharing these details quickly.",
        "am": "እነዚህን ዝርዝሮች በፍጥነት ስለካፈሉ እናመሰግናለን።",
    },
    "possible_causes": {
        "en": "Possible causes from your symptoms",
        "am": "ከምልክቶችዎ የሚገመቱ ምክንያቶች",
    },
    "risk_level_line": {
        "en": "Current risk level",
        "am": "የአሁኑ የአደጋ ደረጃ",
    },
    "recommended_next_step": {
        "en": "Recommended next step",
        "am": "የሚመከረው ቀጣይ እርምጃ",
    },
}

RECOMMENDATION_TRANSLATIONS = {
    "Emergency pattern detected: seek immediate emergency care now.": "የአስቸኳይ ሁኔታ ምልክት ተገኝቷል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible urinary/kidney involvement pattern: arrange a same-day in-person medical evaluation for urine tests and clinical examination.": "የሽንት መስመር ወይም የኩላሊት ተሳትፎ ሊኖር ይችላል፤ በዚያው ቀን ለሽንት ምርመራ እና ለክሊኒካዊ ግምገማ ቀጥታ ሕክምና ያዘጋጁ።",
    "Likely urinary symptom pattern: arrange a same-day or next-day clinic visit for urine testing and treatment guidance.": "የሽንት መስመር ምልክቶች ይመስላሉ፤ በዚያው ቀን ወይም በሚቀጥለው ቀን ለሽንት ምርመራ እና ለሕክምና መመሪያ ክሊኒክ ይጎብኙ።",
    "Urgent: seek immediate in-person medical care or emergency services.": "አስቸኳይ ነው፤ አሁኑኑ ቀጥታ ሕክምና ወይም የድንገተኛ አገልግሎት ይፈልጉ።",
    "Low confidence result: arrange a clinical consultation for proper evaluation.": "የእርግጠኝነት ደረጃው ዝቅ ነው፤ ትክክለኛ ግምገማ ለማድረግ ክሊኒካዊ ምክክር ያዘጋጁ።",
    "Schedule a same-day or next-day clinic visit for professional assessment.": "በዚያው ቀን ወይም በሚቀጥለው ቀን ለሙያዊ ግምገማ ክሊኒክ ቀጠሮ ይያዙ።",
    "Low-risk pattern: monitor symptoms and seek care if symptoms worsen or persist.": "ዝቅተኛ አደጋ ያለው ንድፍ ይመስላል፤ ምልክቶቹን ይከታተሉ እና ከባድ ከሆኑ ወይም ከቀጠሉ ሕክምና ይፈልጉ።",
    "Possible stroke pattern: call emergency services or go to the emergency department immediately.": "የስትሮክ ምልክት ሊሆን ይችላል፤ አሁኑኑ ለድንገተኛ አገልግሎት ይደውሉ ወይም ወደ ድንገተኛ ክፍል ይሂዱ።",
    "Possible cardiac emergency pattern: seek emergency care immediately.": "የልብ ድንገተኛ ሁኔታ ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible pulmonary embolism pattern: seek emergency care immediately.": "የሳንባ ደም መርጋት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible aortic dissection pattern: emergency evaluation is required immediately.": "የአኦርታ ፍንጣቂ ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ግምገማ ያስፈልጋል።",
    "Possible sepsis pattern: seek emergency care immediately.": "የሴፕሲስ ምልክት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible meningitis pattern: emergency evaluation is required immediately.": "የመኒንጃይቲስ ምልክት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ግምገማ ያስፈልጋል።",
    "Possible ectopic pregnancy pattern: seek emergency care immediately.": "ከማህፀን ውጭ እርግዝና ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible appendicitis pattern: urgent in-person evaluation is required today.": "የአፔንዲሲቲስ ምልክት ሊሆን ይችላል፤ ዛሬ አስቸኳይ ቀጥታ ግምገማ ያስፈልጋል።",
    "Possible diabetic ketoacidosis pattern: seek emergency care immediately.": "የዲያቤቲክ ኬቶአሲዶሲስ ምልክት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible anaphylaxis pattern: use emergency services immediately.": "የአናፊላክሲስ ምልክት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ አገልግሎት ይጠቀሙ።",
    "Possible pneumothorax pattern: seek emergency evaluation immediately.": "የፕኑሞቶራክስ ምልክት ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ግምገማ ይፈልጉ።",
    "Possible gastrointestinal bleeding pattern: seek urgent emergency evaluation immediately.": "የሆድ ወይም የአንጀት ደም መፍሰስ ሊኖር ይችላል፤ አሁኑኑ አስቸኳይ የድንገተኛ ግምገማ ይፈልጉ።",
    "Possible bowel obstruction pattern: urgent emergency evaluation is required.": "የአንጀት መዘጋት ሊኖር ይችላል፤ አስቸኳይ የድንገተኛ ግምገማ ያስፈልጋል።",
    "Possible airway emergency pattern: seek emergency care immediately.": "የመተንፈሻ መንገድ ድንገተኛ ሁኔታ ሊኖር ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ይፈልጉ።",
    "Possible severe asthma exacerbation: emergency treatment may be required immediately.": "ከባድ የአስማ መባባስ ሊሆን ይችላል፤ አሁኑኑ የድንገተኛ ሕክምና ሊያስፈልግ ይችላል።",
    "Possible kidney infection pattern: arrange same-day in-person evaluation with urine testing.": "የኩላሊት ኢንፌክሽን ሊሆን ይችላል፤ በዚያው ቀን ለሽንት ምርመራ ቀጥታ ግምገማ ያዘጋጁ።",
    "Likely urinary tract symptom pattern: arrange same-day or next-day clinic assessment.": "የሽንት መስመር ምልክት ይመስላል፤ በዚያው ቀን ወይም በሚቀጥለው ቀን የክሊኒክ ግምገማ ያዘጋጁ።",
    "Possible pneumonia pattern: urgent same-day assessment is recommended.": "የኒሞኒያ ምልክት ሊሆን ይችላል፤ በዚያው ቀን አስቸኳይ ግምገማ ይመከራል።",
    "Likely irritant/allergic skin pattern: avoid the trigger and arrange routine clinical review if symptoms persist.": "የቆዳ አለርጂ ወይም አስቸጋሪ ነገር ምላሽ ይመስላል፤ አነሳሹን ያስወግዱ እና ምልክቶቹ ከቀጠሉ መደበኛ ክሊኒካዊ ግምገማ ያዘጋጁ።",
    "Likely mild viral upper-respiratory pattern: monitor symptoms and seek care if they worsen or persist.": "ቀላል የቫይረስ የላይኛው መተንፈሻ ህመም ይመስላል፤ ምልክቶቹን ይከታተሉ እና ከባድ ከሆኑ ወይም ከቀጠሉ ሕክምና ይፈልጉ።",
    "Possible panic-like pattern: seek urgent care if symptoms are new, severe, or resemble cardiac symptoms.": "የፓኒክ አይነት ሁኔታ ሊሆን ይችላል፤ ምልክቶቹ አዲስ ከሆኑ፣ ከባድ ከሆኑ ወይም እንደ ልብ ህመም ከተመሰሉ አስቸኳይ ሕክምና ይፈልጉ።",
}

AMHARIC_TO_ENGLISH_MAP = {
    "የደረት ህመም": "chest pain",
    "የደረት ሕመም": "chest pain",
    "የመተንፈስ ችግር": "shortness of breath",
    "መተንፈስ ችግር": "shortness of breath",
    "ከፍተኛ ትኩሳት": "high fever",
    "ትኩሳት": "fever",
    "ራስ ህመም": "headache",
    "ራስ ሕመም": "headache",
    "አንድ ጎን ድካም": "one side weakness",
    "ንግግር ችግር": "slurred speech",
    "ማቃጠል ሽንት": "burning urination",
    "ብዙ ሽንት": "frequent urination",
    "የሆድ ህመም": "abdominal pain",
    "የጉሮሮ ህመም": "sore throat",
    "አፍንጫ ፍሳሽ": "runny nose",
    "ሳል": "cough",
    "ሽፍታ": "rash",
    "ማሳከክ": "itchy",
    "ኬሚካል": "chemical",
    "ድንገተኛ መተንፈስ": "trouble breathing",
    "አፍ እና ጉሮሮ መፍጠር": "throat swelling",
    "መዝናኛ አለመሆን": "confusion",
    "ማስወከም": "vomiting",
    "እግር መብጠት": "leg swelling",
    "የጥጃ ህመም": "calf pain",
    "ወባ": "malaria",
    "ኢትዮጵያ": "ethiopia",
}


def translate_static(key: str, language: str) -> str:
    lang = language if language in SUPPORTED_LANGUAGES else "en"
    value = COMMON_STRINGS.get(key, {})
    return str(value.get(lang) or value.get("en") or key)


def localize_risk_level(level: str, language: str) -> str:
    return translate_static(f"risk_{str(level).lower()}", language)


def detect_language(text: str, preferred: Optional[str] = None) -> str:
    preferred_normalized = str(preferred or "").strip().lower()
    if preferred_normalized in SUPPORTED_LANGUAGES:
        return preferred_normalized

    text = str(text or "").strip()
    if not text:
        return "en"
    if AMHARIC_SCRIPT_RE.search(text):
        return "am"

    try:
        from langdetect import detect

        detected = str(detect(text)).strip().lower()
        if detected.startswith("am"):
            return "am"
    except Exception:
        pass
    return "en"


def normalize_text_for_models(text: str, language: str) -> str:
    cleaned = str(text or "").strip()
    if language != "am":
        return cleaned
    translated = cleaned
    for source, target in sorted(AMHARIC_TO_ENGLISH_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(source, f" {target} ")
    translated = WHITESPACE_RE.sub(" ", translated).strip()
    return translated or cleaned


def translate_dynamic_text(text: str, language: str) -> str:
    content = str(text or "").strip()
    if not content or language != "am":
        return content
    if content in RECOMMENDATION_TRANSLATIONS:
        return RECOMMENDATION_TRANSLATIONS[content]
    translated = content
    for source, target in sorted(RECOMMENDATION_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(source, target)
    return translated


def localized_recommendation(risk_level: str, needs_urgent_care: bool, language: str) -> str:
    if needs_urgent_care or str(risk_level).lower() == "high":
        return translate_static("urgent", language)
    if str(risk_level).lower() == "medium":
        return translate_static("same_day", language)
    return translate_static("monitor", language)


def localized_prevention_advice(risk_level: str, language: str) -> List[str]:
    if language != "am":
        if str(risk_level).lower() == "high":
            return [
                "Do not delay care: proceed to emergency or urgent care services immediately.",
                "Stay hydrated and avoid self-medicating without clinician guidance.",
                "Seek in-person evaluation if symptoms worsen or new red flags appear.",
            ]
        return [
            "Stay hydrated and track symptom progression.",
            "Avoid self-medicating without clinician guidance.",
            "Seek care if symptoms worsen, persist, or new severe symptoms appear.",
        ]
    if str(risk_level).lower() == "high":
        return [
            "ሕክምናን አትዘግዩ፤ ወዲያውኑ ወደ አስቸኳይ ወይም የድንገተኛ እንክብካቤ ይሂዱ።",
            "በቂ ፈሳሽ ይውሰዱ እና ያለ ሐኪም ምክር መድሀኒት አትውሰዱ።",
            "ምልክቶች ከባድ ከሆኑ ወይም አዲስ አደገኛ ምልክቶች ከታዩ ወዲያውኑ ሕክምና ይፈልጉ።",
        ]
    return [
        "በቂ ፈሳሽ ይውሰዱ እና ምልክቶቹን ይከታተሉ።",
        "ያለ ሐኪም ምክር መድሀኒት አትውሰዱ።",
        "ምልክቶች ከባድ ከሆኑ ወይም ከቀጠሉ ሕክምና ይፈልጉ።",
    ]


def localize_analysis_result(result: Dict, language: str) -> Dict:
    localized = dict(result)
    localized["response_language"] = language
    localized["detected_language"] = language
    localized["risk_level_label"] = localize_risk_level(str(result.get("risk_level", "")), language)
    recommendation_text = str(result.get("recommendation_text", "")).strip() or localized_recommendation(
        risk_level=str(result.get("risk_level", "")),
        needs_urgent_care=bool(result.get("needs_urgent_care")),
        language=language,
    )
    localized["recommendation_text"] = translate_dynamic_text(recommendation_text, language)
    localized["disclaimer_text"] = translate_static("disclaimer", language)
    localized["prevention_advice"] = localized_prevention_advice(str(result.get("risk_level", "")), language)

    clinical_report = dict(result.get("clinical_report", {}) or {})
    if clinical_report:
        clinical_report["patient_friendly_summary"] = build_assistant_summary(result, language)
        clinical_report["disclaimer"] = translate_static("disclaimer", language)
        localized["clinical_report"] = clinical_report

    localized["localized_strings"] = {
        "disclaimer_text": translate_static("disclaimer", language),
        "seek_professional_care": translate_static("seek_professional_care", language),
        "risk_level": localize_risk_level(str(result.get("risk_level", "")), language),
        "freshness_note": translate_static("freshness_note", language),
    }
    return localized


def build_assistant_summary(result: Dict, language: str) -> str:
    probable = result.get("probable_conditions") or result.get("raw_probable_conditions") or []
    conditions = probable[:3]
    risk_level = str(result.get("risk_level", "Low"))
    localized_risk = localize_risk_level(risk_level, language)
    recommendation = str(result.get("recommendation_text", "")).strip() or localized_recommendation(
        risk_level=risk_level,
        needs_urgent_care=bool(result.get("needs_urgent_care")),
        language=language,
    )
    recommendation = translate_dynamic_text(recommendation, language)
    conditions_text = ", ".join(
        f"{item.get('condition', 'Unknown')} ({float(item.get('probability', 0.0)) * 100:.1f}%)"
        for item in conditions
    ) if conditions else ""

    search_context = result.get("search_context") or {}
    sources = search_context.get("results") or []
    source_tail = ""
    if sources:
        source_names = ", ".join(str(item.get("source", "")) for item in sources[:3] if item.get("source"))
        if language == "am":
            source_tail = f" {translate_static('current_sources_checked', language)}: {source_names}."
        else:
            source_tail = f" {translate_static('current_sources_checked', language)}: {source_names}."

    opener_key = f"assistant_open_{str(risk_level).lower()}"
    opener = translate_static(opener_key, language)
    if language == "am":
        if conditions_text:
            return (
                f"{opener} {translate_static('possible_causes', language)}: {conditions_text}. "
                f"{translate_static('risk_level_line', language)}: {localized_risk}. "
                f"{translate_static('recommended_next_step', language)}: {recommendation}. "
                f"{translate_static('disclaimer', language)} {source_tail}"
            ).strip()
        return (
            f"{opener} {translate_static('risk_level_line', language)}: {localized_risk}. "
            f"{translate_static('recommended_next_step', language)}: {recommendation}. "
            f"{translate_static('disclaimer', language)} {source_tail}"
        ).strip()

    if conditions_text:
        return (
            f"{opener} {translate_static('possible_causes', language)}: {conditions_text}. "
            f"{translate_static('risk_level_line', language)}: {localized_risk}. "
            f"{translate_static('recommended_next_step', language)}: {recommendation}. "
            f"{translate_static('disclaimer', language)}{source_tail}"
        ).strip()
    return (
        f"{opener} {translate_static('risk_level_line', language)}: {localized_risk}. "
        f"{translate_static('recommended_next_step', language)}: {recommendation}. "
        f"{translate_static('disclaimer', language)}{source_tail}"
    ).strip()
