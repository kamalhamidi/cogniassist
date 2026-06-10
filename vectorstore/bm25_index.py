"""
vectorstore/bm25_index.py — Index BM25 sparse pour le hybrid retrieval.

Fournit un index BM25Okapi persistant sur disque (pickle) permettant
la recherche lexicale complémentaire à la recherche dense (ChromaDB).
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger("cogniassist.vectorstore")


class BM25Index:
    """Index BM25 sparse avec persistance pickle."""

    def __init__(self, persist_path: str) -> None:
        """Initialise l'index BM25, charge depuis le disque si existant.

        Args:
            persist_path: Chemin du fichier pickle pour la persistance.
        """
        self.persist_path = persist_path
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []

        self._load()

    def build(self, documents: list[Document]) -> None:
        """Construit l'index BM25 complet depuis zéro et le persiste.

        Args:
            documents: Liste de Documents LangChain à indexer.
        """
        if not documents:
            self.bm25 = None
            self.documents = []
            self.tokenized_corpus = []
            self._save()
            return

        self.documents = list(documents)
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save()

        logger.info(
            "Index BM25 construit : %d documents indexés.", len(self.documents)
        )

    def add_documents(self, new_docs: list[Document]) -> None:
        """Ajoute des documents et reconstruit l'index BM25.

        Args:
            new_docs: Nouveaux Documents LangChain à ajouter.
        """
        if not new_docs:
            return

        self.documents.extend(new_docs)
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save()

        logger.info(
            "Index BM25 mis à jour : %d documents (+%d nouveaux).",
            len(self.documents), len(new_docs),
        )

    def remove_document(self, source_filename: str) -> None:
        """Supprime les chunks d'une source et reconstruit l'index.

        Args:
            source_filename: Nom du fichier source à supprimer.
        """
        original_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents
            if doc.metadata.get("file_name") != source_filename
        ]

        removed = original_count - len(self.documents)
        if removed == 0:
            logger.debug(
                "BM25 : aucun chunk trouvé pour '%s'.", source_filename
            )
            return

        # Reconstruire l'index
        if self.documents:
            self.tokenized_corpus = [
                self._tokenize(doc.page_content) for doc in self.documents
            ]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
            self.tokenized_corpus = []

        self._save()
        logger.info(
            "BM25 : %d chunk(s) supprimé(s) pour '%s'. Reste : %d.",
            removed, source_filename, len(self.documents),
        )

    def retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Recherche les k documents les plus pertinents via BM25.

        Args:
            query: Requête en langage naturel.
            k: Nombre de résultats à retourner.

        Returns:
            Liste de dicts avec clés 'document', 'score', 'index'.
            Les documents avec un score nul sont filtrés.
        """
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Associer scores, documents et indices, filtrer les scores nuls
        scored_docs: list[dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                scored_docs.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "index": idx,
                })

        # Trier par score décroissant et limiter à k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise un texte en mots minuscules.

        Args:
            text: Texte à tokeniser.

        Returns:
            Liste de tokens en minuscules.
        """
        return text.lower().split()

    def _save(self) -> None:
        """Persiste l'index BM25 et les documents sur disque (pickle)."""
        try:
            # Créer le répertoire parent si nécessaire
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "bm25": self.bm25,
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            }
            with open(self.persist_path, "wb") as f:
                pickle.dump(data, f)

            logger.debug("Index BM25 sauvegardé dans '%s'.", self.persist_path)
        except Exception as e:
            logger.error("Erreur sauvegarde BM25 : %s", str(e))

    def _load(self) -> None:
        """Charge l'index BM25 depuis le disque si le fichier existe."""
        path = Path(self.persist_path)
        if not path.exists():
            logger.debug("Aucun index BM25 trouvé sur disque.")
            return

        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)

            self.bm25 = data.get("bm25")
            self.documents = data.get("documents", [])
            self.tokenized_corpus = data.get("tokenized_corpus", [])

            logger.info(
                "Index BM25 chargé depuis '%s' : %d documents.",
                self.persist_path, len(self.documents),
            )
        except Exception as e:
            logger.warning(
                "Impossible de charger l'index BM25 : %s — "
                "l'index sera reconstruit au prochain ajout.",
                str(e),
            )
            self.bm25 = None
            self.documents = []
            self.tokenized_corpus = []
