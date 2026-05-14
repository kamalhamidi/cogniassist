"""
tests/test_rag.py — Tests unitaires du module RAG.

Tests de smoke pour vérifier que les imports fonctionnent
et que les classes s'instancient correctement.
"""

import pytest


class TestPromptBuilder:
    """Tests du constructeur de prompts."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from rag.prompt_builder import PromptBuilder
        assert PromptBuilder is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        assert builder is not None
        assert builder.system_template is not None

    def test_build_system_prompt(self) -> None:
        """Vérifie la construction du prompt système."""
        from rag.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        prompt = builder.build_system_prompt(context="Contexte de test")
        assert "Contexte de test" in prompt
        assert "CogniAssist" in prompt

    def test_build_conversation_prompt(self) -> None:
        """Vérifie la construction du prompt de conversation."""
        from rag.prompt_builder import PromptBuilder
        prompt = PromptBuilder.build_conversation_prompt(question="Bonjour")
        assert "Bonjour" in prompt

    def test_build_conversation_prompt_with_history(self) -> None:
        """Vérifie la construction du prompt avec historique."""
        from rag.prompt_builder import PromptBuilder
        history = [{"question": "Q1", "answer": "A1"}]
        prompt = PromptBuilder.build_conversation_prompt(
            question="Q2",
            history=history,
        )
        assert "Q1" in prompt
        assert "A1" in prompt
        assert "Q2" in prompt


class TestConversationMemory:
    """Tests de la mémoire de conversation."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from rag.memory import ConversationMemory
        assert ConversationMemory is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()
        assert memory.size == 0

    def test_add_and_retrieve(self) -> None:
        """Vérifie l'ajout et la récupération d'échanges."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()
        memory.add_exchange(question="Bonjour", answer="Salut !")
        assert memory.size == 1
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["question"] == "Bonjour"
        assert history[0]["answer"] == "Salut !"

    def test_clear(self) -> None:
        """Vérifie le vidage de la mémoire."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory()
        memory.add_exchange(question="Q", answer="A")
        memory.clear()
        assert memory.size == 0

    def test_max_exchanges(self) -> None:
        """Vérifie la limite de la mémoire."""
        from rag.memory import ConversationMemory
        memory = ConversationMemory(max_exchanges=3)
        for i in range(5):
            memory.add_exchange(question=f"Q{i}", answer=f"A{i}")
        assert memory.size == 3


class TestRAGPipeline:
    """Tests du pipeline RAG."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from rag.pipeline import RAGPipeline
        assert RAGPipeline is not None


class TestSummarizer:
    """Tests du résumeur."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from rag.summarizer import Summarizer
        assert Summarizer is not None
