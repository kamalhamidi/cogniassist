"""
vectorstore/retriever.py — Recherche intelligente par similarité.

Combine le VectorStore avec un reranking par chevauchement de mots-clés
pour améliorer la pertinence des résultats.
"""

import logging
from typing import Optional
from langchain.schema import Document
from vectorstore.store import VectorStore

logger = logging.getLogger("cogniassist.vectorstore")

STOPWORDS_FR = frozenset([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "à", "au", "aux", "ce", "qui", "que", "par", "sur", "dans",
    "est", "sont", "avec", "pour", "pas", "ne", "se", "sa", "son",
])
STOPWORDS_EN = frozenset([
    "the", "a", "an", "of", "in", "is", "are", "and", "to",
    "for", "with", "on", "it", "this", "that", "was", "be",
])
STOPWORDS = STOPWORDS_FR | STOPWORDS_EN


class SmartRetriever:
    """Retriever intelligent avec reranking par mots-clés."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        """Initialise le retriever avec un VectorStore."""
        self.vector_store = vector_store or VectorStore()
        self.use_reranking: bool = True
        self.min_similarity_score: float = 0.3

    def retrieve(
        self, query: str, k: int = 5, filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Récupère les k documents les plus pertinents."""
        candidates = self.vector_store.similarity_search_with_score(
            query=query, k=k * 2, filter_metadata=filter_metadata,
        )
        if not candidates:
            return []

        filtered = [(d, s) for d, s in candidates if s >= self.min_similarity_score]
        if not filtered:
            return []

        if self.use_reranking:
            filtered = self._rerank(query, filtered)

        return [doc for doc, _ in filtered[:k]]

    def _rerank(
        self, query: str, candidates: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        """Reranking hybride : 0.7 × similarité + 0.3 × mots-clés."""
        query_keywords = {
            w.lower() for w in query.split()
            if w.lower() not in STOPWORDS and len(w) > 1
        }
        if not query_keywords:
            return candidates

        reranked: list[tuple[Document, float]] = []
        for doc, sim_score in candidates:
            text_lower = doc.page_content.lower()
            matches = sum(1 for kw in query_keywords if kw in text_lower)
            kw_score = matches / len(query_keywords)
            combined = 0.7 * sim_score + 0.3 * kw_score
            reranked.append((doc, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve_with_context_window(self, query: str, k: int = 3) -> str:
        """Formate les top-k chunks en contexte pour le prompt RAG."""
        documents = self.retrieve(query, k=k)
        if not documents:
            return "Aucun document pertinent trouvé."

        parts: list[str] = []
        for i, doc in enumerate(documents):
            fn = doc.metadata.get("file_name", "source inconnue")
            pg = doc.metadata.get("page_number", "—")
            parts.append(f"--- Document {i+1} (source: {fn}, page: {pg}) ---\n{doc.page_content}")

        return "\n\n".join(parts)
