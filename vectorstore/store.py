"""
vectorstore/store.py — Gestion de la base vectorielle ChromaDB.

Fournit une interface pour créer, alimenter et interroger une
collection ChromaDB persistante avec des embeddings personnalisés.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """
    Interface de gestion pour ChromaDB.

    Gère la persistance, l'ajout de documents et la recherche
    dans la base vectorielle.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "cogniassist_docs",
    ) -> None:
        """
        Initialise la connexion à ChromaDB.

        Args:
            persist_directory: Répertoire de persistance ChromaDB.
                              Si None, utilise les settings.
            collection_name: Nom de la collection ChromaDB.
        """
        if persist_directory is None:
            from config import settings
            persist_directory = str(settings.chroma_persist_path)

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Créer le répertoire si nécessaire
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialiser le client ChromaDB
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Retourne la collection ChromaDB active."""
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """
        Ajoute des documents à la base vectorielle.

        Args:
            ids: Identifiants uniques des documents.
            documents: Textes des documents.
            embeddings: Vecteurs d'embeddings correspondants.
            metadatas: Métadonnées associées à chaque document.
        """
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Recherche les documents les plus similaires à un embedding.

        Args:
            query_embedding: Vecteur d'embedding de la requête.
            n_results: Nombre de résultats à retourner.
            where: Filtre optionnel sur les métadonnées.

        Returns:
            Dictionnaire avec les documents, distances et métadonnées.
        """
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        """Retourne le nombre de documents dans la collection."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Supprime la collection actuelle."""
        self._client.delete_collection(self.collection_name)

    def reset(self) -> None:
        """Réinitialise la collection (supprime et recrée)."""
        self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
