"""
vectorstore/embedder.py — Génération d'embeddings.

Utilise Ollama pour générer des vecteurs d'embeddings
à partir de texte. Le modèle est configurable via les settings.
"""

from typing import Optional

from langchain_ollama import OllamaEmbeddings


class Embedder:
    """
    Génère des embeddings vectoriels à partir de texte.

    Utilise un modèle Ollama configurable pour
    encoder des passages de texte en vecteurs denses.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialise l'embedder avec le modèle spécifié.

        Args:
            model_name: Nom du modèle d'embeddings Ollama.
                        Si None, utilise le modèle défini dans les settings.
        """
        from config import settings

        self.model_name = model_name or settings.embedding_model
        self.base_url = settings.ollama_base_url
        self._model: Optional[OllamaEmbeddings] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> OllamaEmbeddings:
        """Charge le modèle en lazy-loading (chargé au premier appel)."""
        if self._model is None:
            print(f"🔄 Chargement du modèle d'embeddings : {self.model_name}...")
            self._model = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
            )
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
        embedding = self.model.embed_query(text)
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

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

        embeddings = self.model.embed_documents(texts)
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings du modèle."""
        if self._dimension is None:
            self._dimension = len(self.embed_text("dimension probe"))
        return self._dimension
