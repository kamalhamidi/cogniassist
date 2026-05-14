"""
vectorstore/embedder.py — Génération d'embeddings.

Utilise Sentence-Transformers pour générer des vecteurs d'embeddings
à partir de texte. Le modèle est configurable via les settings.
"""

from typing import Optional

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Génère des embeddings vectoriels à partir de texte.

    Utilise un modèle Sentence-Transformers configurable pour
    encoder des passages de texte en vecteurs denses.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialise l'embedder avec le modèle spécifié.

        Args:
            model_name: Nom du modèle Sentence-Transformers.
                        Si None, utilise le modèle défini dans les settings.
        """
        if model_name is None:
            from config import settings
            model_name = settings.EMBEDDING_MODEL

        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Charge le modèle en lazy-loading (chargé au premier appel)."""
        if self._model is None:
            print(f"🔄 Chargement du modèle d'embeddings : {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("✅ Modèle chargé avec succès.")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Génère un embedding pour un seul texte.

        Args:
            text: Le texte à encoder.

        Returns:
            Vecteur d'embedding sous forme de liste de floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Génère des embeddings pour une liste de textes.

        Args:
            texts: Liste de textes à encoder.
            batch_size: Taille des lots pour le traitement.

        Returns:
            Liste de vecteurs d'embeddings.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings du modèle."""
        return self.model.get_sentence_embedding_dimension()
