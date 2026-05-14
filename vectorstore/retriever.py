"""
vectorstore/retriever.py — Recherche par similarité.

Combine l'embedder et le vector store pour rechercher les documents
les plus pertinents en fonction d'une requête utilisateur.
"""

from typing import Optional
from dataclasses import dataclass

from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore


@dataclass
class RetrievedDocument:
    """Représente un document retrouvé par similarité."""

    content: str
    score: float
    metadata: dict


class DocumentRetriever:
    """
    Recherche des documents pertinents par similarité sémantique.

    Combine un Embedder pour vectoriser la requête et un VectorStore
    pour retrouver les documents les plus proches.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        max_results: Optional[int] = None,
    ) -> None:
        """
        Initialise le retriever.

        Args:
            embedder: Instance de l'Embedder. Si None, en crée un par défaut.
            store: Instance du VectorStore. Si None, en crée un par défaut.
            max_results: Nombre maximum de documents à retourner.
        """
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

        if max_results is None:
            from config import settings
            max_results = settings.MAX_RETRIEVED_DOCS

        self.max_results = max_results

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """
        Recherche les documents les plus pertinents pour une requête.

        Args:
            query: La requête utilisateur en langage naturel.
            n_results: Nombre de résultats (override max_results).
            where: Filtre optionnel sur les métadonnées.

        Returns:
            Liste de RetrievedDocument triés par pertinence.
        """
        n = n_results or self.max_results

        # Générer l'embedding de la requête
        query_embedding = self.embedder.embed_text(query)

        # Rechercher dans la base vectorielle
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=n,
            where=where,
        )

        # Construire les résultats
        retrieved: list[RetrievedDocument] = []
        if results and results.get("documents"):
            documents = results["documents"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)

            for doc, dist, meta in zip(documents, distances, metadatas):
                retrieved.append(
                    RetrievedDocument(
                        content=doc,
                        score=1.0 - dist,  # Cosine distance → similarity
                        metadata=meta,
                    )
                )

        return retrieved

    def retrieve_with_context(self, query: str, n_results: Optional[int] = None) -> str:
        """
        Recherche et formate les documents pertinents en contexte texte.

        Args:
            query: La requête utilisateur.
            n_results: Nombre de résultats.

        Returns:
            Texte formaté avec les documents pertinents.
        """
        documents = self.retrieve(query, n_results=n_results)

        if not documents:
            return "Aucun document pertinent trouvé."

        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "source inconnue")
            context_parts.append(
                f"[Document {i} — {source} (score: {doc.score:.2f})]\n{doc.content}"
            )

        return "\n\n---\n\n".join(context_parts)
