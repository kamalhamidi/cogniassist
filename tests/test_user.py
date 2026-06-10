"""
tests/test_user.py — Tests unitaires du module user.

Tests des modèles SQLAlchemy, du gestionnaire de profils,
de l'historique des interactions et du recommandeur.
Utilise une base SQLite en mémoire pour l'isolation.
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Configure une base SQLite temporaire pour chaque test."""
    import user.db as db_module

    # Réinitialiser les singletons du module db
    db_module._engine = None
    db_module._SessionFactory = None

    # Pointer la base vers un fichier temporaire
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(
        "config.settings.SQLITE_DB_PATH",
        str(test_db),
    )

    # Forcer le recalcul du chemin
    from config import BASE_DIR
    monkeypatch.setattr(
        "config.Settings.sqlite_db_path",
        property(lambda self: test_db),
    )

    # Initialiser les tables
    from user.db import init_db
    init_db()

    yield


class TestDatabase:
    """Tests de la base de données."""

    def test_init_db_creates_tables(self) -> None:
        """Vérifie que init_db crée les tables attendues."""
        from sqlalchemy import inspect
        from user.db import get_engine, init_db

        init_db()
        inspector = inspect(get_engine())
        tables = inspector.get_table_names()

        assert "user_profiles" in tables
        assert "user_preferences" in tables
        assert "document_access" in tables
        assert "interactions" in tables


class TestUserProfileManager:
    """Tests du gestionnaire de profils."""

    def test_create_default_user(self) -> None:
        """Vérifie la création automatique d'un utilisateur par défaut."""
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_user")
        profile = manager.get_profile()

        assert profile["user_id"] == "test_user"
        assert profile["name"] == "Utilisateur"
        assert profile["avatar"] == "🧠"
        assert profile["language"] == "fr"
        assert profile["expertise_level"] == "intermediate"

    def test_update_profile(self) -> None:
        """Vérifie la mise à jour du profil."""
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_user2")
        manager.update_profile(name="Kamal", email="k@test.com")

        profile = manager.get_profile()
        assert profile["name"] == "Kamal"
        assert profile["email"] == "k@test.com"

    def test_update_preferences(self) -> None:
        """Vérifie la mise à jour des préférences."""
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_user3")
        manager.update_preferences(
            expertise_level="expert",
            domain_focus=["NLP", "RAG", "Python"],
            goals="Maîtriser le deep learning",
        )

        profile = manager.get_profile()
        assert profile["expertise_level"] == "expert"
        assert "NLP" in profile["domain_focus"]
        assert "RAG" in profile["domain_focus"]
        assert "deep learning" in profile["goals"]

    def test_get_personalization_context(self) -> None:
        """Vérifie le contexte de personnalisation."""
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_ctx")
        manager.update_preferences(
            expertise_level="expert",
            language="fr",
        )

        ctx = manager.get_personalization_context()
        assert "expert" in ctx
        assert "fr" in ctx
        assert "Profil" in ctx

    def test_register_and_get_documents(self) -> None:
        """Vérifie l'enregistrement et la récupération de documents."""
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_doc_user")
        manager.register_document("rapport.pdf", "pdf", 15)

        docs = manager.get_user_documents()
        assert len(docs) == 1
        assert docs[0]["file_name"] == "rapport.pdf"
        assert docs[0]["chunk_count"] == 15

    def test_register_and_delete_document(self) -> None:
        """Vérifie la suppression d'un document."""
        from unittest.mock import patch, MagicMock
        from user.profile import UserProfileManager

        manager = UserProfileManager("test_del_user")
        manager.register_document("to_delete.txt", "txt", 5)

        assert len(manager.get_user_documents()) == 1

        # Mock le VectorStore pour éviter de toucher ChromaDB
        with patch("vectorstore.store.VectorStore") as mock_vs:
            mock_vs.return_value = MagicMock()
            manager.delete_document("to_delete.txt")

        assert len(manager.get_user_documents()) == 0


class TestInteractionHistory:
    """Tests de l'historique des interactions."""

    def test_save_and_retrieve_interaction(self) -> None:
        """Vérifie la sauvegarde et récupération d'interactions."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        # Créer l'utilisateur d'abord
        UserProfileManager("test_hist")

        history = InteractionHistory("test_hist")
        iid = history.save_interaction(
            question="test question",
            answer="test answer",
            sources=["doc.pdf"],
            chunks_used=3,
            response_time_ms=500,
        )

        assert iid > 0

        recent = history.get_recent_history(limit=5)
        assert len(recent) >= 1
        assert recent[0]["question"] == "test question"
        assert recent[0]["answer"] == "test answer"
        assert "doc.pdf" in recent[0]["sources"]

    def test_save_feedback(self) -> None:
        """Vérifie la sauvegarde du feedback."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        UserProfileManager("test_fb")
        history = InteractionHistory("test_fb")

        iid = history.save_interaction(
            question="Q", answer="A", sources=[], chunks_used=0,
        )

        history.save_feedback(iid, 1)

        recent = history.get_recent_history(limit=1)
        assert recent[0]["feedback"] == 1

    def test_get_frequent_topics(self) -> None:
        """Vérifie l'extraction de sujets fréquents."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        UserProfileManager("test_topics")
        history = InteractionHistory("test_topics")

        for _ in range(5):
            history.save_interaction(
                question="machine learning avec Python",
                answer="réponse",
                sources=[], chunks_used=0,
            )

        topics = history.get_frequent_topics()
        assert any(t in topics for t in ["machine", "learning", "python"])

    def test_search_history(self) -> None:
        """Vérifie la recherche dans l'historique."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        UserProfileManager("test_search")
        history = InteractionHistory("test_search")

        history.save_interaction(
            question="Qu'est-ce que le NLP ?",
            answer="NLP signifie...",
            sources=[], chunks_used=0,
        )

        results = history.search_history("NLP")
        assert len(results) >= 1

    def test_clear_history(self) -> None:
        """Vérifie le vidage de l'historique."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        UserProfileManager("test_clear")
        history = InteractionHistory("test_clear")

        history.save_interaction(
            question="Q", answer="A", sources=[], chunks_used=0,
        )
        history.clear_history()

        assert len(history.get_recent_history()) == 0

    def test_interaction_stats(self) -> None:
        """Vérifie les statistiques d'interaction."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory

        UserProfileManager("test_stats")
        history = InteractionHistory("test_stats")

        history.save_interaction(
            question="Q1", answer="A1", sources=["a.pdf"],
            chunks_used=2, response_time_ms=100,
        )
        history.save_interaction(
            question="Q2", answer="A2", sources=["b.pdf"],
            chunks_used=3, response_time_ms=200,
        )

        stats = history.get_interaction_stats()
        assert stats["total_interactions"] == 2
        assert stats["avg_response_time_ms"] == 150.0


class TestPersonalizedRecommender:
    """Tests du recommandeur personnalisé."""

    def test_adapt_rag_parameters_beginner(self) -> None:
        """Vérifie les paramètres RAG pour un débutant."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        manager = UserProfileManager("test_beginner")
        manager.update_preferences(expertise_level="beginner")

        rec = PersonalizedRecommender("test_beginner")
        params = rec.adapt_rag_parameters()

        assert params["k"] == 8  # 7 base + 1 (detailed default)
        assert params["temperature"] == 0.2

    def test_adapt_rag_parameters_expert(self) -> None:
        """Vérifie les paramètres RAG pour un expert."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        manager = UserProfileManager("test_expert")
        manager.update_preferences(expertise_level="expert")

        rec = PersonalizedRecommender("test_expert")
        params = rec.adapt_rag_parameters()

        assert params["k"] == 5  # 4 base + 1 (detailed default)
        assert params["temperature"] == 0.1

    def test_learning_progress_score(self) -> None:
        """Vérifie le score de progression plafonné à 100."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory
        from user.recommender import PersonalizedRecommender

        manager = UserProfileManager("test_progress")
        manager.register_document("doc1.pdf", "pdf", 10)
        manager.register_document("doc2.pdf", "pdf", 5)

        history = InteractionHistory("test_progress")
        for i in range(3):
            history.save_interaction(
                question=f"Q{i}", answer=f"A{i}",
                sources=[], chunks_used=0,
            )

        rec = PersonalizedRecommender("test_progress")
        progress = rec.get_learning_progress()

        assert progress["documents_uploaded"] == 2
        assert progress["total_questions_asked"] == 3
        assert 0 < progress["knowledge_score"] <= 100
        # Score = 2*10 + 3*2 = 26
        assert progress["knowledge_score"] == 26

    def test_suggested_questions(self) -> None:
        """Vérifie que 5 questions sont toujours suggérées."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        UserProfileManager("test_suggest")
        rec = PersonalizedRecommender("test_suggest")
        questions = rec.get_suggested_questions()

        assert len(questions) == 5
        assert all(isinstance(q, str) for q in questions)

    def test_document_recommendations(self) -> None:
        """Vérifie les recommandations de documents non consultés."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        manager = UserProfileManager("test_rec_doc")
        manager.register_document("never_read.pdf", "pdf", 10)

        rec = PersonalizedRecommender("test_rec_doc")
        recs = rec.get_document_recommendations()

        assert len(recs) >= 1
        assert recs[0]["file_name"] == "never_read.pdf"
        assert recs[0]["reason"] == "Jamais consulté"
