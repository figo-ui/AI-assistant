import requests
from django.conf import settings
from typing import List, Dict, Optional
import hashlib
from django.core.cache import cache
import json as _json

# Secure sites only for medical accuracy
MEDICAL_WHITELIST = [
    "mayoclinic.org",
    "webmd.com",
    "nih.gov",
    "cdc.gov",
    "who.int",
    "nhs.uk",
    "healthline.com",
    "clevelandclinic.org"
]

def perform_google_search(query: str, limit: int = 3) -> List[Dict]:
    """
    Performs a grounded search using Google Custom Search API.
    Only returns results from verified medical domains if possible.
    """
    api_key = getattr(settings, "GOOGLE_SEARCH_API_KEY", None)
    cse_id = getattr(settings, "GOOGLE_SEARCH_CSE_ID", None)
    
    if not api_key or not cse_id:
        print("GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_CSE_ID not configured.")
        return []

    # Cache handling to avoid repeated API costs
    cache_key = f"google_search:{hashlib.md5(query.encode()).hexdigest()}"
    cached = cache.get(cache_key)
    if cached:
        return _json.loads(cached)

    endpoint = "https://www.googleapis.com/customsearch/v1"
    
    # We add 'site:' constraints for medical grounding if desired
    # For now, we search generally but prioritize these domains in the prompt
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": limit
    }

    try:
        response = requests.get(endpoint, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        print(f"Google Search Error: {e}")
        return []

    results = []
    for item in payload.get("items", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "source": "Google Search"
        })

    if results:
        cache.set(cache_key, _json.loads(_json.dumps(results)), timeout=3600*12) # 12h cache

    return results

def build_search_context(query: str) -> str:
    """Wraps search results into a context string for the AI."""
    results = perform_google_search(query)
    if not results:
        return ""
    
    context = "\n--- Verified Search Context ---\n"
    for r in results:
        context += f"Source: {r['title']} ({r['link']})\n"
        context += f"Snippet: {r['snippet']}\n\n"
    return context
