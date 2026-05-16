"""
Module vectorstore — Gestion des embeddings et de la base vectorielle.

Usage principal :
    from vectorstore import add_to_vectorstore, search_vectorstore
    stats = add_to_vectorstore(chunks)
    results = search_vectorstore("ma requête", k=5)
"""

import logging
from langchain.schema import Document
from vectorstore.embedder import EmbeddingManager
from vectorstore.store import VectorStore
from vectorstore.retriever import SmartRetriever

logger = logging.getLogger("cogniassist.vectorstore")

__all__ = [
    "EmbeddingManager", "VectorStore", "SmartRetriever",
    "add_to_vectorstore", "search_vectorstore",
]


def add_to_vectorstore(chunks: list[Document]) -> dict:
    """
    Ajoute des chunks à ChromaDB et retourne les statistiques.

    Args:
        chunks: Liste de Documents LangChain à indexer.

    Returns:
        Statistiques de la collection après ajout.
    """
    store = VectorStore()
    added = store.add_documents(chunks)
    logger.info("%d chunk(s) ajouté(s) au vectorstore.", added)
    return store.get_collection_stats()


def search_vectorstore(query: str, k: int = 5) -> list[Document]:
    """
    Recherche dans le vectorstore via le SmartRetriever.

    Args:
        query: Requête en langage naturel.
        k: Nombre de résultats.

    Returns:
        Liste de Documents triés par pertinence.
    """
    store = VectorStore()
    retriever = SmartRetriever(vector_store=store)
    return retriever.retrieve(query, k=k)
