import html
import re
from typing import Dict, List, Optional
import requests


FRESHNESS_PATTERNS = [
    "latest",
    "current",
    "new",
    "recent",
    "today",
    "2025",
    "2026",
    "guideline",
    "outbreak",
    "ethiopia",
    "malaria",
    "cdc",
    "who",
    "pubmed",
]

SEARCH_USER_AGENT = "healthcare-assistant-search/1.0"
ANCHOR_RE = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def needs_fresh_search(text: str, force_search: bool = False) -> bool:
    if force_search:
        return True
    lowered = str(text or "").strip().lower()
    return any(term in lowered for term in FRESHNESS_PATTERNS)


def _safe_get(url: str, *, params: Optional[Dict] = None, timeout: int = 20) -> Optional[requests.Response]:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": SEARCH_USER_AGENT},
        )
        response.raise_for_status()
        return response
    except Exception:
        return None


def search_pubmed(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    search_response = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
    )
    if search_response is None:
        return []

    try:
        ids = search_response.json()["esearchresult"]["idlist"]
    except Exception:
        return []
    if not ids:
        return []

    summary_response = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    if summary_response is None:
        return []

    try:
        payload = summary_response.json()
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for item_id in ids:
        item = payload.get("result", {}).get(item_id, {})
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        results.append(
            {
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{item_id}/",
                "snippet": str(item.get("sortpubdate", "")).strip(),
                "source": "PubMed",
            }
        )
    return results[:max_results]


def search_google(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    api_key = getattr(settings, "GOOGLE_SEARCH_API_KEY", None)
    cse_id = getattr(settings, "GOOGLE_SEARCH_CSE_ID", None)
    if not api_key or not cse_id:
        return []

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": max_results}
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for item in payload.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": "Google Search",
        })
    return results


def search_duckduckgo(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    response = _safe_get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=20)
    if response is None:
        return []

    results: List[Dict[str, str]] = []
    for match in ANCHOR_RE.finditer(response.text):
        href = html.unescape(match.group("href"))
        title = TAG_RE.sub("", html.unescape(match.group("title"))).strip()
        if not href or not title:
            continue
        results.append(
            {
                "title": title,
                "url": href,
                "snippet": "",
                "source": "DuckDuckGo",
            }
        )
        if len(results) >= max_results:
            break
    return results
def _dedupe_results(results: List[Dict[str, str]], max_results: int) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in results:
        key = (str(item.get("url", "")).strip(), str(item.get("title", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break
    return deduped


def run_search_router(
    query: str,
    *,
    translated_query: Optional[str] = None,
    force_search: bool = False,
    search_consent: bool = False,
    mock_results: Optional[List[Dict[str, str]]] = None,
    max_results: int = 5,
) -> Dict[str, object]:
    freshness_needed = needs_fresh_search(query, force_search=force_search)
    if not freshness_needed:
        return {
            "freshness_needed": False,
            "query": query,
            "results": [],
            "sources": [],
            "used_mock_results": False,
            "blocked_by_policy": False,
            "consent_required": False,
        }

    effective_query = str(translated_query or query or "").strip()
    if not search_consent:
        return {
            "freshness_needed": True,
            "query": effective_query,
            "results": [],
            "sources": [],
            "used_mock_results": False,
            "blocked_by_policy": True,
            "consent_required": True,
        }

    if not effective_query:
        return {
            "freshness_needed": True,
            "query": effective_query,
            "results": [],
            "sources": [],
            "used_mock_results": False,
            "blocked_by_policy": True,
            "consent_required": False,
        }

    if mock_results:
        results = _dedupe_results(list(mock_results), max_results=max_results)
        return {
            "freshness_needed": True,
            "query": effective_query,
            "results": results,
            "sources": sorted({str(item.get("source", "")) for item in results if item.get("source")}),
            "used_mock_results": True,
            "blocked_by_policy": False,
            "consent_required": False,
        }

    results = []
    # Try Google Search first
    google_res = search_google(effective_query, max_results=max_results)
    if google_res:
        results.extend(google_res)
    
    # Supplement with PubMed for clinical depth
    results.extend(search_pubmed(effective_query, max_results=3))
    
    # Fallback to DuckDuckGo if still low on results
    if len(results) < max_results:
        results.extend(search_duckduckgo(effective_query, max_results=2))

    deduped = _dedupe_results(results, max_results=max_results)
    return {
        "freshness_needed": True,
        "query": effective_query,
        "results": deduped,
        "sources": sorted({str(item.get("source", "")) for item in deduped if item.get("source")}),
        "used_mock_results": False,
        "blocked_by_policy": False,
        "consent_required": False,
    }


def build_search_prompt_context(search_context: Optional[Dict[str, object]]) -> str:
    if not search_context or not search_context.get("results"):
        return ""
    lines = ["Use these current sources if they are relevant to the answer:"]
    for item in list(search_context["results"])[:5]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        source = str(item.get("source", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        line = f"- [{source}] {title} - {url}"
        if snippet:
            line += f" ({snippet})"
        lines.append(line)
    return "\n".join(lines)
