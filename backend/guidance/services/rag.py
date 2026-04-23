from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEDQUAD_PATH = PROJECT_ROOT / "data" / "dialogue" / "medquad_dialogue_pairs.csv"
SYNTHEA_PATH = PROJECT_ROOT / "data" / "synthea_symptom_condition.csv"

RAG_TRIGGER_TERMS = {
    "treatment",
    "treat",
    "therapy",
    "management",
    "prevention",
    "prevent",
    "guideline",
    "recommendation",
    "medication",
    "dosage",
    "next step",
}


@dataclass(frozen=True)
class RagDoc:
    text: str
    source: str
    metadata: Dict[str, str]


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    docs: List[RagDoc]
    faiss_index: Optional[object] = None


def should_use_rag(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in RAG_TRIGGER_TERMS)


def _load_medquad(max_docs: int) -> List[RagDoc]:
    if not MEDQUAD_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with MEDQUAD_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = str(row.get("user_text", "")).strip()
            answer = str(row.get("assistant_text", "")).strip()
            if not question or not answer:
                continue
            text = f"Q: {question}\nA: {answer}"
            docs.append(
                RagDoc(
                    text=text,
                    source="medquad",
                    metadata={"question": question[:240], "answer": answer[:240]},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _load_synthea(max_docs: int) -> List[RagDoc]:
    if not SYNTHEA_PATH.exists():
        return []
    docs: List[RagDoc] = []
    with SYNTHEA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symptoms = str(row.get("symptom_text", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not symptoms or not condition:
                continue
            text = f"Symptoms: {symptoms}\nPossible condition: {condition}"
            docs.append(
                RagDoc(
                    text=text,
                    source="synthea",
                    metadata={"symptoms": symptoms[:240], "condition": condition[:120]},
                )
            )
            if len(docs) >= max_docs:
                break
    return docs


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@lru_cache(maxsize=1)
def _build_index() -> RagIndex:
    max_docs = int(os.getenv("RAG_MAX_DOCS", "8000"))
    medquad_docs = _load_medquad(max_docs=max_docs // 2)
    synthea_docs = _load_synthea(max_docs=max_docs // 2)
    docs = medquad_docs + synthea_docs
    if not docs:
        return RagIndex(vectorizer=TfidfVectorizer(), matrix=np.zeros((0, 1)), docs=[])

    texts = [doc.text for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix_sparse = vectorizer.fit_transform(texts)
    matrix = matrix_sparse.astype(np.float32).toarray()

    faiss_index = None
    if faiss is not None and matrix.shape[0] > 0:
        normalized = _normalize(matrix)
        faiss_index = faiss.IndexFlatIP(normalized.shape[1])
        faiss_index.add(normalized)
        matrix = normalized

    return RagIndex(vectorizer=vectorizer, matrix=matrix, docs=docs, faiss_index=faiss_index)


def query_rag(query: str, top_k: int = 4) -> List[Dict[str, object]]:
    if not query.strip():
        return []
    index = _build_index()
    if not index.docs:
        return []

    query_vec = index.vectorizer.transform([query]).astype(np.float32).toarray()
    if index.faiss_index is not None:
        query_vec = _normalize(query_vec)
        scores, indices = index.faiss_index.search(query_vec, min(top_k, len(index.docs)))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = index.docs[int(idx)]
            hits.append({"text": doc.text, "source": doc.source, "score": float(score), "metadata": doc.metadata})
        return hits

    scores = cosine_similarity(query_vec, index.matrix)[0]
    ranked = np.argsort(scores)[::-1][: min(top_k, len(index.docs))]
    results = []
    for idx in ranked:
        doc = index.docs[int(idx)]
        results.append(
            {"text": doc.text, "source": doc.source, "score": float(scores[int(idx)]), "metadata": doc.metadata}
        )
    return results


def build_rag_context(query: str, *, top_k: int = 4, search_context: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    hits = query_rag(query, top_k=top_k)
    return {
        "enabled": bool(hits),
        "query": query,
        "items": hits,
        "sources": sorted({item["source"] for item in hits}),
        "external_search": search_context or {},
    }
