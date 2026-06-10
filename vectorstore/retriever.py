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


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Retrieval — Dense (ChromaDB) + Sparse (BM25) avec RRF
# ═══════════════════════════════════════════════════════════════════════

from vectorstore.bm25_index import BM25Index


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[Document]:
    """Fusionne deux listes de résultats via Reciprocal Rank Fusion.

    Calcule un score RRF pondéré pour chaque document et déduplique
    par chunk_id dans les métadonnées.

    Args:
        dense_results: Documents issus de la recherche dense (ChromaDB).
        sparse_results: Documents issus de la recherche sparse (BM25).
        k: Constante RRF (contrôle l'importance du rang).
        dense_weight: Poids de la composante dense.
        sparse_weight: Poids de la composante sparse.

    Returns:
        Liste de Documents triés par score RRF décroissant.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Scorer les résultats dense
    for rank, doc in enumerate(dense_results):
        chunk_id = doc.metadata.get("chunk_id", f"dense_{rank}")
        score = dense_weight / (k + rank + 1)
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc

    # Scorer les résultats sparse
    for rank, doc in enumerate(sparse_results):
        chunk_id = doc.metadata.get("chunk_id", f"sparse_{rank}")
        score = sparse_weight / (k + rank + 1)
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc

    # Trier par score fusionné décroissant
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[chunk_id] for chunk_id in sorted_ids]


class HybridRetriever:
    """Retriever hybride combinant dense (SmartRetriever) et sparse (BM25)."""

    def __init__(
        self,
        smart_retriever: SmartRetriever,
        bm25_index: BM25Index,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        """Initialise le retriever hybride.

        Args:
            smart_retriever: Retriever dense (ChromaDB + reranking).
            bm25_index: Index BM25 sparse.
            dense_weight: Poids RRF pour les résultats dense.
            sparse_weight: Poids RRF pour les résultats sparse.
        """
        self.smart_retriever = smart_retriever
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """Récupère les k documents les plus pertinents via fusion hybride.

        Fetch k*3 candidats dans chaque retriever avant fusion RRF.

        Args:
            query: Requête en langage naturel.
            k: Nombre de résultats finaux souhaités.
            filter_metadata: Filtre optionnel pour le retriever dense.

        Returns:
            Liste de Documents triés par score RRF décroissant.
        """
        candidates_k = k * 3

        # Dense : SmartRetriever (ChromaDB)
        dense_results = self.smart_retriever.retrieve(
            query, k=candidates_k, filter_metadata=filter_metadata,
        )

        # Sparse : BM25
        bm25_results_raw = self.bm25_index.retrieve(query, k=candidates_k)
        sparse_results = [r["document"] for r in bm25_results_raw]

        # Fusion RRF
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        return fused[:k]

    def retrieve_with_context_window(self, query: str, k: int = 5) -> str:
        """Formate les top-k chunks hybrides en contexte pour le prompt RAG.

        Args:
            query: Requête en langage naturel.
            k: Nombre de chunks à inclure dans le contexte.

        Returns:
            Contexte formaté avec sources pour injection dans le prompt.
        """
        documents = self.retrieve(query, k=k)
        if not documents:
            return "Aucun document pertinent trouvé."

        parts: list[str] = []
        for i, doc in enumerate(documents):
            fn = doc.metadata.get("file_name", "source inconnue")
            pg = doc.metadata.get("page_number", "—")
            parts.append(
                f"--- [Extrait {i + 1} — {fn}, page {pg}] ---\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(parts)
