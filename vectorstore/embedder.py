"""
vectorstore/embedder.py — Gestion des modèles d'embeddings.

Utilise Ollama comme modèle principal d'embeddings et
HuggingFace (Sentence-Transformers) comme fallback automatique
si Ollama n'est pas accessible.
"""

import logging
from typing import Optional

from langchain_ollama import OllamaEmbeddings

from config import settings

logger = logging.getLogger("cogniassist.vectorstore")


class EmbeddingManager:
    """
    Gestionnaire d'embeddings avec basculement automatique.

    Priorité : Ollama (nomic-embed-text) → HuggingFace (fallback local).
    Le basculement est transparent pour l'appelant.
    """

    def __init__(self) -> None:
        """
        Initialise le modèle Ollama principal et le fallback HuggingFace.

        Le modèle Ollama est configuré via settings.embedding_model
        et settings.ollama_base_url. Le fallback HuggingFace utilise
        paraphrase-multilingual-mpnet-base-v2 sur CPU.
        """
        # Modèle principal : Ollama
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

        # Fallback : HuggingFace (chargé à la demande)
        self._fallback_embeddings = None
        self.active_model: str = "ollama"

        logger.info(
            "EmbeddingManager initialisé — modèle : %s (Ollama @ %s)",
            settings.embedding_model, settings.ollama_base_url,
        )

    @property
    def fallback_embeddings(self):
        """Charge le modèle HuggingFace en lazy-loading."""
        if self._fallback_embeddings is None:
            logger.info("Chargement du modèle fallback HuggingFace…")
            from langchain_huggingface import HuggingFaceEmbeddings

            self._fallback_embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Modèle fallback HuggingFace chargé avec succès.")
        return self._fallback_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Génère des embeddings pour une liste de textes.

        Tente Ollama en premier. En cas d'échec de connexion,
        bascule automatiquement vers le modèle HuggingFace.

        Args:
            texts: Liste de textes à encoder.

        Returns:
            Liste de vecteurs d'embeddings (liste de floats).
        """
        if not texts:
            return []

        # Essayer Ollama en premier
        try:
            result = self.embeddings.embed_documents(texts)
            self.active_model = "ollama"
            return result
        except Exception as e:
            logger.warning(
                "Ollama indisponible pour embed_documents : %s — "
                "basculement vers HuggingFace",
                str(e),
            )
            self.active_model = "huggingface"
            return self.fallback_embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Génère un embedding pour une requête unique.

        Même logique de fallback que embed_documents.
        Utilisé au moment de la recherche par similarité.

        Args:
            query: Texte de la requête.

        Returns:
            Vecteur d'embedding (liste de floats).
        """
        # Essayer Ollama en premier
        try:
            result = self.embeddings.embed_query(query)
            self.active_model = "ollama"
            return result
        except Exception as e:
            logger.warning(
                "Ollama indisponible pour embed_query : %s — "
                "basculement vers HuggingFace",
                str(e),
            )
            self.active_model = "huggingface"
            return self.fallback_embeddings.embed_query(query)

    def get_active_model(self) -> str:
        """
        Retourne le nom du modèle d'embeddings actuellement actif.

        Utile pour l'affichage dans l'interface utilisateur.

        Returns:
            Nom du modèle actif ("ollama" ou "huggingface").
        """
        return self.active_model

    def test_connection(self) -> bool:
        """
        Teste si Ollama est accessible en encodant un texte court.

        Ne lève jamais d'exception — attrape toutes les erreurs en interne.

        Returns:
            True si Ollama répond, False sinon.
        """
        try:
            self.embeddings.embed_query("test")
            logger.info("Test de connexion Ollama : succès ✅")
            return True
        except Exception as e:
            logger.warning("Test de connexion Ollama : échec ❌ — %s", str(e))
            return False
