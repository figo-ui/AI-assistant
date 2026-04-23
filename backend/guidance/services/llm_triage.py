import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from django.conf import settings

from .clinical_safety import apply_prediction_safety_overrides, build_safety_summary
from .schema import response_schema_prompt, validate_triage_response
from .search_router import build_search_prompt_context


SYSTEM_PROMPT = (
    "You are a healthcare triage assistant. "
    "Return exactly one JSON object and no prose before or after it. "
    "Do not claim a final diagnosis. "
    "Use short, direct condition names and normalized probabilities that sum to 1."
)


def _adapter_path() -> Path:
    return Path(getattr(settings, "TRIAGE_LLM_ADAPTER_PATH", os.getenv("TRIAGE_LLM_ADAPTER_PATH", ""))).resolve()


@lru_cache(maxsize=1)
def _load_llm():
    adapter_path = _adapter_path()
    if not adapter_path.exists():
        return None, None

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = getattr(
        settings,
        "TRIAGE_LLM_BASE_MODEL",
        os.getenv("TRIAGE_LLM_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.load_adapter(str(adapter_path))

    return model, tokenizer


def llm_available() -> bool:
    try:
        model, tokenizer = _load_llm()
    except Exception:
        return False
    return bool(model is not None and tokenizer is not None)


def _extract_json_payload(text: str) -> Optional[Dict]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _prompt_language(language: str) -> str:
    return "Amharic" if str(language).lower() == "am" else "English"


def predict_with_llm(
    symptom_text: str,
    top_k: int = 5,
    *,
    language: str = "en",
    search_context: Optional[Dict[str, object]] = None,
) -> Dict:
    import torch

    try:
        model, tokenizer = _load_llm()
    except Exception:
        return {"available": False, "predictions": [], "risk_level": None, "raw_text": ""}
    if model is None or tokenizer is None:
        return {"available": False, "predictions": [], "risk_level": None, "raw_text": ""}

    prompt_sections = [
        SYSTEM_PROMPT,
        f"Respond in {_prompt_language(language)}.",
        "Use this schema exactly:",
        response_schema_prompt(),
    ]
    search_prompt = build_search_prompt_context(search_context)
    if search_prompt:
        prompt_sections.append(search_prompt)

    messages = [
        {"role": "system", "content": "\n\n".join(prompt_sections)},
        {"role": "user", "content": symptom_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=384,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    payload = _extract_json_payload(decoded)
    if payload is None:
        return {
            "available": False,
            "predictions": [],
            "risk_level": None,
            "raw_text": decoded,
            "error": "Model did not return valid JSON",
        }

    structured = validate_triage_response(payload, raw_text=decoded)
    predictions = [
        {"condition": item.condition, "probability": item.probability}
        for item in structured.predictions
    ]
    predictions = apply_prediction_safety_overrides(symptom_text, predictions, top_k=top_k)

    return {
        "available": True,
        "predictions": predictions[:top_k],
        "risk_level": structured.risk_level,
        "red_flags": structured.red_flags,
        "recommendation": structured.recommendation,
        "reasoning": structured.reasoning,
        "raw_text": decoded,
        "safety": build_safety_summary(symptom_text),
        "language": language,
        "search_context": search_context or {},
    }
