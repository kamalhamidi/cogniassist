"""
tests/test_rag.py — Tests unitaires du module RAG.

Tests du PromptBuilder, ConversationMemory et RAGPipeline.
Utilise unittest.mock pour éviter de dépendre d'Ollama.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPromptBuilder:
    """Tests du constructeur de prompts."""

    def test_prompt_builder_rag(self) -> None:
        """Vérifie que build_rag_prompt contient la question et le contexte."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()

        prompt = builder.build_rag_prompt(
            question="Qu'est-ce que le machine learning ?",
            context="Le ML est une branche de l'IA.",
            chat_history="Utilisateur : Bonjour",
            user_profile="Étudiant Master",
        )

        assert "Qu'est-ce que le machine learning ?" in prompt
        assert "Le ML est une branche de l'IA." in prompt
        assert "Bonjour" in prompt
        assert "Étudiant Master" in prompt
        assert "CogniAssist" in prompt

    def test_prompt_builder_summary(self) -> None:
        """Vérifie que build_summary_prompt tronque le contenu à 3000 chars."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()

        long_content = "A" * 5000
        prompt = builder.build_summary_prompt(
            document_content=long_content,
            file_name="test.pdf",
        )

        # Le contenu original (5000 chars) ne doit PAS être dans le prompt
        assert "A" * 5000 not in prompt
        # Le contenu tronqué (3000 chars) DOIT être présent
        assert "A" * 3000 in prompt
        assert "tronqué" in prompt
        assert "test.pdf" in prompt

    def test_prompt_builder_knowledge_gap(self) -> None:
        """Vérifie que build_knowledge_gap_prompt formate correctement."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()

        prompt = builder.build_knowledge_gap_prompt(
            documents_summary="Documents : ML, NLP",
            user_goals="Maîtriser le deep learning",
        )

        assert "ML, NLP" in prompt
        assert "deep learning" in prompt

    def test_prompt_builder_rag_default_history(self) -> None:
        """Vérifie le prompt RAG sans historique."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()

        prompt = builder.build_rag_prompt(
            question="Test",
            context="Contexte",
        )

        assert "aucun historique" in prompt


class TestConversationMemory:
    """Tests de la mémoire de conversation."""

    def test_memory_add_messages(self) -> None:
        """Vérifie l'ajout de messages et le comptage."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()

        for i in range(3):
            memory.add_user_message(f"Question {i}")
            memory.add_assistant_message(f"Réponse {i}")

        assert memory.get_message_count() == 6

        history = memory.get_formatted_history()
        assert "Utilisateur" in history
        assert "Assistant" in history

    def test_memory_compression(self) -> None:
        """Vérifie la compression quand max_messages est dépassé."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory(max_messages=4)

        for i in range(6):
            memory.add_user_message(f"Message {i}")

        # Après compression, on garde les 6 derniers → max 6 dans messages
        # mais max_messages=4, donc compression déclenchée
        assert len(memory.messages) <= 6
        assert memory.summary != ""

    def test_memory_clear(self) -> None:
        """Vérifie le vidage complet de la mémoire."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()

        memory.add_user_message("Hello")
        memory.add_assistant_message("World")
        memory.clear()

        assert memory.get_message_count() == 0
        assert memory.summary == ""
        assert memory.get_formatted_history() == ""

    def test_memory_langchain_messages(self) -> None:
        """Vérifie la conversion en messages LangChain."""
        from rag.memory import ConversationMemory
        from langchain_core.messages import HumanMessage, AIMessage

        memory = ConversationMemory()
        memory.add_user_message("Bonjour")
        memory.add_assistant_message("Salut !")

        messages = memory.get_langchain_messages()
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_memory_empty_history(self) -> None:
        """Vérifie l'historique vide."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()
        assert memory.get_formatted_history() == ""


class TestRAGPipeline:
    """Tests du pipeline RAG (mockés pour éviter Ollama)."""

    @patch("rag.pipeline.ChatOllama")
    @patch("rag.pipeline.VectorStore")
    def test_pipeline_status_keys(self, mock_vs_cls, mock_llm_cls) -> None:
        """Vérifie les clés retournées par get_pipeline_status."""
        # Mock VectorStore
        mock_store = MagicMock()
        mock_store.get_collection_stats.return_value = {
            "total_chunks": 10,
            "embedding_model": "ollama",
            "collection_name": "test",
            "persist_directory": "/tmp",
            "unique_documents": [],
        }
        mock_store.collection = MagicMock()
        mock_store.collection.count.return_value = 10
        mock_vs_cls.return_value = mock_store

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

        status = pipeline.get_pipeline_status()

        expected_keys = {
            "is_ready", "llm_model", "ollama_url",
            "message_count", "embedding_model", "total_documents_indexed",
        }
        assert expected_keys == set(status.keys())
        assert status["is_ready"] is True

    @patch("rag.pipeline.ChatOllama")
    @patch("rag.pipeline.VectorStore")
    def test_ask_raises_if_not_ready(self, mock_vs_cls, mock_llm_cls) -> None:
        """Vérifie que ask() lève RuntimeError si le pipeline n'est pas prêt."""
        mock_store = MagicMock()
        mock_store.get_collection_stats.return_value = {"total_chunks": 0}
        mock_store.collection = MagicMock()
        mock_store.collection.count.return_value = 0
        mock_vs_cls.return_value = mock_store

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Ollama down")
        mock_llm_cls.return_value = mock_llm

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        # La vérification de readiness a échoué
        assert pipeline.is_ready is False

        with pytest.raises(RuntimeError, match="pipeline RAG n'est pas prêt"):
            pipeline.ask("test question")

    @patch("rag.pipeline.ChatOllama")
    @patch("rag.pipeline.VectorStore")
    def test_clear_memory(self, mock_vs_cls, mock_llm_cls) -> None:
        """Vérifie que clear_memory vide la mémoire."""
        mock_store = MagicMock()
        mock_store.get_collection_stats.return_value = {"total_chunks": 0}
        mock_store.collection = MagicMock()
        mock_store.collection.count.return_value = 0
        mock_vs_cls.return_value = mock_store

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.memory.add_user_message("test")
        assert pipeline.memory.get_message_count() == 1

        pipeline.clear_memory()
        assert pipeline.memory.get_message_count() == 0


class TestBatchSummarizer:
    """Tests du résumeur par lots."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from rag.summarizer import BatchSummarizer
        assert BatchSummarizer is not None

    def test_summarize_with_mock_pipeline(self) -> None:
        """Vérifie le résumé par lots avec un pipeline mocké."""
        from rag.summarizer import BatchSummarizer

        mock_pipeline = MagicMock()
        mock_pipeline.summarize_document.return_value = "Résumé test"

        summarizer = BatchSummarizer(mock_pipeline)
        results = summarizer.summarize_all_documents(["doc1.pdf", "doc2.txt"])

        assert len(results) == 2
        assert results["doc1.pdf"] == "Résumé test"
        assert results["doc2.txt"] == "Résumé test"

    def test_knowledge_report_format(self) -> None:
        """Vérifie le format du rapport de connaissances."""
        from rag.summarizer import BatchSummarizer

        mock_pipeline = MagicMock()
        mock_pipeline.summarize_document.return_value = "Contenu résumé"
        mock_pipeline.analyze_knowledge_gaps.return_value = "Lacune identifiée"

        summarizer = BatchSummarizer(mock_pipeline)
        report = summarizer.generate_knowledge_report(
            user_goals="Apprendre le NLP",
            document_names=["ml.pdf"],
        )

        assert "Rapport de connaissances" in report
        assert "ml.pdf" in report
        assert "Lacune identifiée" in report
        assert "CogniAssist" in report
