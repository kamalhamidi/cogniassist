"""
tests/test_vectorstore.py — Tests unitaires du module vectorstore.

Tests fonctionnels couvrant l'EmbeddingManager, le VectorStore,
et le SmartRetriever. Utilise le fallback HuggingFace si Ollama
n'est pas disponible.
"""

import pytest
from langchain.schema import Document


class TestEmbeddingManager:
    """Tests du gestionnaire d'embeddings."""

    def test_embedding_manager_init(self) -> None:
        """Vérifie que l'EmbeddingManager s'initialise sans erreur."""
        from vectorstore.embedder import EmbeddingManager
        manager = EmbeddingManager()
        assert manager is not None
        assert manager.get_active_model() != ""

    def test_embed_query_returns_vector(self) -> None:
        """Vérifie que embed_query retourne un vecteur de floats."""
        from vectorstore.embedder import EmbeddingManager
        manager = EmbeddingManager()
        result = manager.embed_query("test de recherche en français")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_returns_vectors(self) -> None:
        """Vérifie que embed_documents retourne des vecteurs."""
        from vectorstore.embedder import EmbeddingManager
        manager = EmbeddingManager()
        texts = ["Premier texte", "Deuxième texte"]
        result = manager.embed_documents(texts)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_embed_empty_list(self) -> None:
        """Vérifie le comportement avec une liste vide."""
        from vectorstore.embedder import EmbeddingManager
        manager = EmbeddingManager()
        assert manager.embed_documents([]) == []


class TestVectorStore:
    """Tests du VectorStore ChromaDB."""

    def _make_store(self, tmp_path):
        """Crée un VectorStore temporaire pour les tests."""
        from vectorstore.store import VectorStore
        return VectorStore(persist_directory=str(tmp_path / "test_chroma"))

    def _sample_chunks(self) -> list[Document]:
        """Crée des chunks de test."""
        return [
            Document(
                page_content="L'intelligence artificielle transforme le monde moderne.",
                metadata={"file_name": "test_file.txt", "chunk_id": "test_1"},
            ),
            Document(
                page_content="Le traitement du langage naturel est une branche de l'IA.",
                metadata={"file_name": "test_file.txt", "chunk_id": "test_2"},
            ),
            Document(
                page_content="Python est un langage de programmation populaire.",
                metadata={"file_name": "test_file2.txt", "chunk_id": "test_3"},
            ),
        ]

    def test_vectorstore_add_and_search(self, tmp_path) -> None:
        """Vérifie l'ajout et la recherche de documents."""
        store = self._make_store(tmp_path)
        chunks = self._sample_chunks()

        added = store.add_documents(chunks)
        assert added == 3

        results = store.similarity_search("intelligence artificielle")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)

    def test_delete_document(self, tmp_path) -> None:
        """Vérifie la suppression de chunks par nom de fichier."""
        store = self._make_store(tmp_path)
        chunks = self._sample_chunks()
        store.add_documents(chunks)

        deleted = store.delete_document("test_file.txt")
        assert deleted == 2

        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 1
        assert "test_file2.txt" in stats["unique_documents"]
        assert "test_file.txt" not in stats["unique_documents"]

    def test_similarity_score_between_0_and_1(self, tmp_path) -> None:
        """Vérifie que les scores de similarité sont entre 0 et 1."""
        store = self._make_store(tmp_path)
        store.add_documents(self._sample_chunks())

        results = store.similarity_search_with_score("test query")
        assert len(results) > 0
        for doc, score in results:
            assert 0.0 <= score <= 1.0, f"Score hors limites : {score}"

    def test_get_collection_stats(self, tmp_path) -> None:
        """Vérifie les statistiques de la collection."""
        store = self._make_store(tmp_path)
        store.add_documents(self._sample_chunks())

        stats = store.get_collection_stats()
        assert "total_chunks" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats
        assert "persist_directory" in stats
        assert "unique_documents" in stats
        assert stats["total_chunks"] == 3
        assert len(stats["unique_documents"]) == 2

    def test_duplicate_detection(self, tmp_path) -> None:
        """Vérifie que les doublons ne sont pas ajoutés."""
        store = self._make_store(tmp_path)
        chunks = self._sample_chunks()

        added_first = store.add_documents(chunks)
        added_second = store.add_documents(chunks)

        assert added_first == 3
        assert added_second == 0
        assert store.get_collection_stats()["total_chunks"] == 3

    def test_reset_collection(self, tmp_path) -> None:
        """Vérifie la réinitialisation de la collection."""
        store = self._make_store(tmp_path)
        store.add_documents(self._sample_chunks())
        assert store.get_collection_stats()["total_chunks"] == 3

        store.reset_collection()
        assert store.get_collection_stats()["total_chunks"] == 0


class TestSmartRetriever:
    """Tests du SmartRetriever."""

    def _make_retriever(self, tmp_path):
        """Crée un retriever avec store temporaire."""
        from vectorstore.store import VectorStore
        from vectorstore.retriever import SmartRetriever
        store = VectorStore(persist_directory=str(tmp_path / "test_retriever"))
        return SmartRetriever(vector_store=store), store

    def test_retriever_filters_low_scores(self, tmp_path) -> None:
        """Vérifie le filtrage des résultats avec score faible."""
        from vectorstore.retriever import SmartRetriever
        retriever, store = self._make_retriever(tmp_path)

        # Ajouter un document sans rapport avec la requête
        store.add_documents([
            Document(
                page_content="La recette de cuisine du gâteau au chocolat nécessite des oeufs et du beurre.",
                metadata={"file_name": "recette.txt", "chunk_id": "recette_1"},
            ),
        ])

        # Seuil très élevé
        retriever.min_similarity_score = 0.9
        results = retriever.retrieve("algorithmes de deep learning en Python")
        assert results == []

    def test_retrieve_with_context_window(self, tmp_path) -> None:
        """Vérifie le formatage du contexte RAG."""
        retriever, store = self._make_retriever(tmp_path)
        store.add_documents([
            Document(
                page_content="Le machine learning utilise des données pour apprendre.",
                metadata={"file_name": "ml.txt", "chunk_id": "ml_1", "page_number": 1},
            ),
        ])

        context = retriever.retrieve_with_context_window("machine learning", k=1)
        assert "Document 1" in context
        assert "ml.txt" in context
        assert "machine learning" in context.lower()
