"""
Module vectorstore — Gestion des embeddings et de la base vectorielle.

Ce module gère la génération d'embeddings avec Sentence-Transformers,
le stockage dans ChromaDB, et la recherche par similarité.
"""

from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore
from vectorstore.retriever import DocumentRetriever

__all__ = ["Embedder", "VectorStore", "DocumentRetriever"]
