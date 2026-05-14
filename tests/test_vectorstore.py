"""
tests/test_vectorstore.py — Tests unitaires du module vectorstore.

Tests de smoke pour vérifier que les imports fonctionnent
et que les classes s'instancient correctement.
"""

import pytest


class TestEmbedder:
    """Tests de l'embedder."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from vectorstore.embedder import Embedder
        assert Embedder is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie (sans charger le modèle)."""
        from vectorstore.embedder import Embedder
        embedder = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
        assert embedder.model_name == "paraphrase-multilingual-mpnet-base-v2"
        # Le modèle n'est pas encore chargé (lazy loading)
        assert embedder._model is None


class TestVectorStore:
    """Tests du vector store."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from vectorstore.store import VectorStore
        assert VectorStore is not None

    def test_instantiation(self, tmp_path) -> None:
        """Vérifie que la classe s'instancie avec un répertoire temporaire."""
        from vectorstore.store import VectorStore
        store = VectorStore(
            persist_directory=str(tmp_path / "test_chroma"),
            collection_name="test_collection",
        )
        assert store is not None
        assert store.count() == 0

    def test_add_and_count(self, tmp_path) -> None:
        """Vérifie l'ajout de documents et le comptage."""
        from vectorstore.store import VectorStore
        store = VectorStore(
            persist_directory=str(tmp_path / "test_chroma2"),
            collection_name="test_collection2",
        )
        store.add_documents(
            ids=["doc1"],
            documents=["Ceci est un test"],
            embeddings=[[0.1] * 384],
            metadatas=[{"source": "test"}],
        )
        assert store.count() == 1


class TestDocumentRetriever:
    """Tests du retriever."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from vectorstore.retriever import DocumentRetriever, RetrievedDocument
        assert DocumentRetriever is not None
        assert RetrievedDocument is not None

    def test_retrieved_document(self) -> None:
        """Vérifie le dataclass RetrievedDocument."""
        from vectorstore.retriever import RetrievedDocument
        doc = RetrievedDocument(
            content="Test",
            score=0.95,
            metadata={"source": "test.pdf"},
        )
        assert doc.content == "Test"
        assert doc.score == 0.95
