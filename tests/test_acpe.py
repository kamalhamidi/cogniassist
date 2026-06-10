"""
tests/test_acpe.py — Tests unitaires ACPE (Adaptive Cognitive Profiling Engine).

Tests des modèles, du moteur de connaissances, de l'évolution du profil,
du profiling progressif, de l'onboarding, et du RAG adaptatif.
Utilise une base SQLite en mémoire pour l'isolation.
"""

import json
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
    test_db = tmp_path / "test_acpe.db"
    monkeypatch.setattr(
        "config.settings.SQLITE_DB_PATH",
        str(test_db),
    )

    from config import BASE_DIR
    monkeypatch.setattr(
        "config.Settings.sqlite_db_path",
        property(lambda self: test_db),
    )

    from user.db import init_db
    init_db()

    yield


class TestACPEModels:
    """Tests des modèles de base ACPE."""

    def test_tables_created(self) -> None:
        """Vérifie que les tables ACPE sont créées."""
        from sqlalchemy import inspect
        from user.db import get_engine

        inspector = inspect(get_engine())
        tables = inspector.get_table_names()

        assert "knowledge_profiles" in tables
        assert "usage_patterns" in tables
        assert "progressive_prompts" in tables

    def test_knowledge_profile_to_dict(self) -> None:
        """Vérifie la sérialisation du KnowledgeProfile."""
        from user.acpe_models import KnowledgeProfile
        from user.db import get_session

        session = get_session()
        # Créer l'utilisateur d'abord
        from user.profile import UserProfileManager
        UserProfileManager("test_kp")

        kp = KnowledgeProfile(
            user_id="test_kp",
            domain="Python",
            mastery_score=75,
            confidence=60,
            interaction_count=10,
        )
        session.add(kp)
        session.commit()

        d = kp.to_dict()
        assert d["domain"] == "Python"
        assert d["mastery_score"] == 75
        assert d["confidence"] == 60

    def test_usage_pattern_json(self) -> None:
        """Vérifie l'encodage/décodage JSON des métriques."""
        from user.acpe_models import UsagePattern
        from user.db import get_session
        from user.profile import UserProfileManager

        UserProfileManager("test_up")
        session = get_session()

        up = UsagePattern(user_id="test_up", metric_name="test_metric")
        up.set_value({"average": 42.5, "count": 10})
        session.add(up)
        session.commit()

        val = up.get_value()
        assert val["average"] == 42.5
        assert val["count"] == 10

    def test_progressive_prompt_action_data(self) -> None:
        """Vérifie l'encodage/décodage des données d'action."""
        from user.acpe_models import ProgressivePrompt
        from user.db import get_session
        from user.profile import UserProfileManager

        UserProfileManager("test_pp")
        session = get_session()

        pp = ProgressivePrompt(
            user_id="test_pp",
            prompt_key="test_key",
            message="Test message",
            action_data=json.dumps({"type": "add_interest", "domain": "Python"}),
        )
        session.add(pp)
        session.commit()

        data = pp.get_action_data()
        assert data["type"] == "add_interest"
        assert data["domain"] == "Python"


class TestUserProfileACPE:
    """Tests des extensions ACPE du profil utilisateur."""

    def test_new_fields_defaults(self) -> None:
        """Vérifie les valeurs par défaut des champs ACPE."""
        from user.profile import UserProfileManager

        mgr = UserProfileManager("test_defaults")
        p = mgr.get_profile()

        assert p["user_type"] == "individual"
        assert p["onboarding_completed"] is False
        assert p["role"] is None
        assert p["organization_data"] == {}
        assert p["adaptive_learning_enabled"] is True

    def test_complete_onboarding_individual(self) -> None:
        """Vérifie l'onboarding individuel."""
        from user.profile import UserProfileManager

        mgr = UserProfileManager("test_onboard_ind")
        mgr.complete_onboarding({
            "user_type": "individual",
            "name": "Kamal",
            "role": "student",
            "expertise_level": "expert",
            "response_style": "concise",
            "language": "fr",
            "goals": ["Étudier", "Résumer"],
            "interests": ["Python", "ML"],
        })

        p = mgr.get_profile()
        assert p["onboarding_completed"] is True
        assert p["user_type"] == "individual"
        assert p["name"] == "Kamal"
        assert p["role"] == "student"
        assert p["expertise_level"] == "expert"
        assert p["response_style"] == "concise"
        assert "Python" in p["domain_focus"]

    def test_complete_onboarding_enterprise(self) -> None:
        """Vérifie l'onboarding entreprise."""
        from user.profile import UserProfileManager

        mgr = UserProfileManager("test_onboard_ent")
        mgr.complete_onboarding({
            "user_type": "enterprise",
            "name": "Admin Corp",
            "role": "enterprise_admin",
            "organization_data": {
                "industry": "technology",
                "org_size": "51-250",
                "confidentiality": "sensitive",
            },
        })

        p = mgr.get_profile()
        assert p["user_type"] == "enterprise"
        assert p["onboarding_completed"] is True
        assert p["organization_data"]["industry"] == "technology"
        assert p["organization_data"]["confidentiality"] == "sensitive"

    def test_personalization_context_includes_acpe(self) -> None:
        """Vérifie que le contexte RAG inclut les données ACPE."""
        from user.profile import UserProfileManager

        mgr = UserProfileManager("test_ctx_acpe")
        mgr.complete_onboarding({
            "user_type": "enterprise",
            "role": "enterprise_admin",
            "organization_data": {
                "industry": "finance",
                "confidentiality": "highly_sensitive",
            },
        })

        ctx = mgr.get_personalization_context()
        assert "enterprise" in ctx
        assert "finance" in ctx
        assert "highly_sensitive" in ctx

    def test_export_profile_data(self) -> None:
        """Vérifie l'export de données."""
        from user.profile import UserProfileManager

        mgr = UserProfileManager("test_export")
        export = mgr.export_profile_data()

        assert "profile" in export
        assert "knowledge_profiles" in export
        assert "usage_patterns" in export
        assert "exported_at" in export

    def test_reset_acpe_data(self) -> None:
        """Vérifie la réinitialisation des données ACPE."""
        from user.profile import UserProfileManager
        from user.knowledge_engine import KnowledgeProfileEngine

        mgr = UserProfileManager("test_reset")
        ke = KnowledgeProfileEngine("test_reset")

        # Créer des données
        ke.update_from_interaction(
            "Python est un langage", "Oui", [], None,
        )
        assert len(ke.get_knowledge_profile()) > 0

        # Reset
        mgr.reset_acpe_data()
        assert len(ke.get_knowledge_profile()) == 0

    def test_reset_system(self, tmp_path, monkeypatch) -> None:
        """Vérifie la réinitialisation complète du système."""
        from user.profile import UserProfileManager
        from user.history import InteractionHistory
        from user.db import reset_system, get_session
        from user.profile import UserProfile, UserPreferences
        from user.history import Interaction
        from config import settings

        # Mocker les chemins d'upload et de BM25 vers des dossiers temporaires
        test_upload_dir = tmp_path / "uploads"
        test_upload_dir.mkdir()
        monkeypatch.setattr(
            "config.settings.UPLOAD_DIR",
            str(test_upload_dir),
        )

        test_bm25_path = "test_bm25_index.pkl"
        monkeypatch.setattr(
            "config.settings.BM25_INDEX_PATH",
            test_bm25_path,
        )

        # 1. Créer des données de test
        mgr = UserProfileManager("test_user")
        mgr.complete_onboarding({
            "user_type": "individual",
            "name": "Kamal",
            "role": "student",
        })

        hist = InteractionHistory("test_user")
        hist.save_interaction(
            question="Hello",
            answer="Hi",
            sources=["doc1.pdf"],
            chunks_used=2,
            response_time_ms=100,
        )

        # S'assurer que les données existent
        session = get_session()
        assert session.query(UserProfile).filter_by(user_id="test_user").count() == 1
        assert session.query(UserPreferences).filter_by(user_id="test_user").count() == 1
        assert session.query(Interaction).filter_by(user_id="test_user").count() == 1

        # Créer un fichier d'index BM25 et de document mock
        from config import BASE_DIR
        bm25_full_path = (BASE_DIR / test_bm25_path).resolve()
        bm25_full_path.write_text("dummy index content")
        assert bm25_full_path.exists()

        upload_mock_file = test_upload_dir / "test_doc.txt"
        upload_mock_file.write_text("dummy document")
        assert upload_mock_file.exists()

        # 2. Exécuter la réinitialisation complète
        reset_system()

        # 3. Vérifier que tout a été effacé
        session = get_session()
        assert session.query(UserProfile).count() == 0
        assert session.query(UserPreferences).count() == 0
        assert session.query(Interaction).count() == 0

        # Les fichiers doivent être supprimés
        assert not bm25_full_path.exists()
        assert not upload_mock_file.exists()


class TestKnowledgeEngine:
    """Tests du moteur de profil de connaissances."""

    def test_extract_domains(self) -> None:
        """Vérifie l'extraction de domaines à partir de texte."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_extract")
        ke = KnowledgeProfileEngine("test_extract")

        domains = ke._extract_domains(
            "Le machine learning et le deep learning avec Python"
        )
        assert "Python" in domains
        assert "Machine Learning" in domains
        assert "Deep Learning" in domains

    def test_update_from_interaction(self) -> None:
        """Vérifie la mise à jour du profil après une interaction."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_kp_update")
        ke = KnowledgeProfileEngine("test_kp_update")

        domains = ke.update_from_interaction(
            question="Comment fonctionne le machine learning ?",
            answer="Le ML est une branche de l'IA...",
            sources=["ml_guide.pdf"],
            feedback=1,
        )

        assert len(domains) > 0
        profile = ke.get_knowledge_profile()
        assert len(profile) > 0
        assert profile[0]["mastery_score"] > 0

    def test_mastery_increases_with_positive_feedback(self) -> None:
        """Vérifie que le score augmente avec du feedback positif."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_mastery_up")
        ke = KnowledgeProfileEngine("test_mastery_up")

        # Plusieurs interactions positives sur Python
        for _ in range(5):
            ke.update_from_interaction(
                "Python programming", "Python est...", [], feedback=1,
            )

        profile = ke.get_knowledge_profile()
        python_kp = next((k for k in profile if k["domain"] == "Python"), None)
        assert python_kp is not None
        assert python_kp["mastery_score"] > 50  # Au-dessus de la base

    def test_confidence_grows_with_interactions(self) -> None:
        """Vérifie que la confiance augmente logarithmiquement."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_confidence")
        ke = KnowledgeProfileEngine("test_confidence")

        for _ in range(10):
            ke.update_from_interaction(
                "SQL query database", "SQL est...", [], None,
            )

        profile = ke.get_knowledge_profile()
        sql_kp = next(
            (k for k in profile if k["domain"] == "SQL & Bases de données"),
            None,
        )
        assert sql_kp is not None
        assert sql_kp["confidence"] > 0
        assert sql_kp["interaction_count"] == 10

    def test_top_and_weak_domains(self) -> None:
        """Vérifie les fonctions top/weak domains."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_domains")
        ke = KnowledgeProfileEngine("test_domains")

        # Beaucoup d'interactions positives sur Python
        for _ in range(5):
            ke.update_from_interaction(
                "Python programming", "...", [], feedback=1,
            )

        # Interactions négatives sur SQL
        for _ in range(5):
            ke.update_from_interaction(
                "SQL database query", "...", [], feedback=-1,
            )

        top = ke.get_top_domains(limit=3)
        weak = ke.get_weak_domains(limit=3, threshold=50)

        assert "Python" in top
        assert "SQL & Bases de données" in weak

    def test_empty_text_returns_no_domains(self) -> None:
        """Vérifie qu'un texte vide ne retourne aucun domaine."""
        from user.knowledge_engine import KnowledgeProfileEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_empty")
        ke = KnowledgeProfileEngine("test_empty")
        assert ke._extract_domains("") == []
        assert ke._extract_domains("bonjour tout le monde") == []


class TestProfileEvolution:
    """Tests du moteur d'évolution du profil."""

    def test_analyze_interaction(self) -> None:
        """Vérifie l'analyse d'interaction complète."""
        from user.profile_evolution import ProfileEvolutionEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_evo")
        evo = ProfileEvolutionEngine("test_evo")

        # Pas d'erreur
        evo.analyze_interaction(
            question="Machine learning avec Python",
            answer="Le ML utilise des algorithmes...",
            sources=["ml.pdf"],
            feedback=1,
            response_time_ms=500,
        )

        summary = evo.get_evolution_summary()
        assert "knowledge_profile" in summary
        assert "insights" in summary

    def test_usage_patterns_tracked(self) -> None:
        """Vérifie que les patterns d'utilisation sont suivis."""
        from user.profile_evolution import ProfileEvolutionEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_patterns")
        evo = ProfileEvolutionEngine("test_patterns")

        for i in range(3):
            evo.analyze_interaction(
                question=f"Question {i}",
                answer="Réponse courte",
                sources=[],
                response_time_ms=100,
            )

        patterns = evo._get_all_metrics()
        assert "total_questions" in patterns
        assert patterns["total_questions"] == 3

    def test_generate_insights(self) -> None:
        """Vérifie la génération d'insights."""
        from user.profile_evolution import ProfileEvolutionEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_insights")
        evo = ProfileEvolutionEngine("test_insights")

        # Créer suffisamment d'interactions pour des insights
        for _ in range(5):
            evo.analyze_interaction(
                "Python machine learning",
                "...",
                [], None, 100,
            )

        insights = evo.generate_insights()
        # Au minimum, l'insight d'activité devrait exister
        assert isinstance(insights, list)


class TestProgressiveProfiling:
    """Tests du moteur de profiling progressif."""

    def test_no_prompt_before_minimum(self) -> None:
        """Vérifie qu'aucune suggestion n'apparaît trop tôt."""
        from user.progressive import ProgressiveProfilingEngine
        from user.profile import UserProfileManager

        UserProfileManager("test_prog_early")
        prog = ProgressiveProfilingEngine("test_prog_early")

        result = prog.check_for_prompts()
        assert result is None

    def test_accept_prompt_adds_interest(self) -> None:
        """Vérifie que accepter ajoute un domaine aux intérêts."""
        from user.progressive import ProgressiveProfilingEngine
        from user.acpe_models import ProgressivePrompt
        from user.profile import UserProfileManager
        from user.db import get_session

        mgr = UserProfileManager("test_accept")
        prog = ProgressiveProfilingEngine("test_accept")

        # Créer manuellement une suggestion
        session = get_session()
        pp = ProgressivePrompt(
            user_id="test_accept",
            prompt_key="add_interest_python",
            message="Ajouter Python ?",
            action_data=json.dumps({"type": "add_interest", "domain": "Python"}),
            status="pending",
        )
        session.add(pp)
        session.commit()

        # Accepter
        prog.accept_prompt(pp.id)

        # Vérifier que Python est dans les intérêts
        profile = mgr.get_profile()
        assert "Python" in profile.get("domain_focus", [])

    def test_decline_prompt(self) -> None:
        """Vérifie que décliner marque la suggestion comme refusée."""
        from user.progressive import ProgressiveProfilingEngine
        from user.acpe_models import ProgressivePrompt
        from user.profile import UserProfileManager
        from user.db import get_session

        UserProfileManager("test_decline")
        prog = ProgressiveProfilingEngine("test_decline")

        session = get_session()
        pp = ProgressivePrompt(
            user_id="test_decline",
            prompt_key="test_decline_key",
            message="Test decline",
            action_data=json.dumps({"type": "add_interest", "domain": "SQL"}),
            status="pending",
        )
        session.add(pp)
        session.commit()

        prog.decline_prompt(pp.id)

        # Utiliser la session du progressive engine pour lire le résultat
        refreshed = (
            prog.session.query(ProgressivePrompt)
            .filter_by(id=pp.id)
            .first()
        )
        assert refreshed.status == "declined"

    def test_change_style_action(self) -> None:
        """Vérifie l'action de changement de style."""
        from user.progressive import ProgressiveProfilingEngine
        from user.acpe_models import ProgressivePrompt
        from user.profile import UserProfileManager
        from user.db import get_session

        mgr = UserProfileManager("test_style")
        prog = ProgressiveProfilingEngine("test_style")

        session = get_session()
        pp = ProgressivePrompt(
            user_id="test_style",
            prompt_key="change_style_concise",
            message="Passer en concis ?",
            action_data=json.dumps({"type": "change_style", "style": "concise"}),
            status="pending",
        )
        session.add(pp)
        session.commit()

        prog.accept_prompt(pp.id)

        profile = mgr.get_profile()
        assert profile["response_style"] == "concise"


class TestAdaptiveRAG:
    """Tests des paramètres RAG adaptatifs ACPE."""

    def test_beginner_individual(self) -> None:
        """Vérifie les paramètres pour un débutant individuel."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        mgr = UserProfileManager("test_rag_beg")
        mgr.update_preferences(expertise_level="beginner")

        rec = PersonalizedRecommender("test_rag_beg")
        params = rec.adapt_rag_parameters()

        assert params["k"] == 8  # 7 base + 1 (detailed default)
        assert params["temperature"] == 0.2
        assert params["user_type"] == "individual"

    def test_expert_individual(self) -> None:
        """Vérifie les paramètres pour un expert individuel."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        mgr = UserProfileManager("test_rag_exp")
        mgr.update_preferences(expertise_level="expert")

        rec = PersonalizedRecommender("test_rag_exp")
        params = rec.adapt_rag_parameters()

        assert params["k"] == 5  # 4 base + 1 (detailed default)
        assert params["temperature"] == 0.1

    def test_enterprise_user(self) -> None:
        """Vérifie les paramètres pour un utilisateur entreprise."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        mgr = UserProfileManager("test_rag_ent")
        mgr.update_profile(user_type="enterprise")

        rec = PersonalizedRecommender("test_rag_ent")
        params = rec.adapt_rag_parameters()

        assert params["k"] == 7  # 6 enterprise + 1 (detailed default)
        assert params["temperature"] == 0.15
        assert params["user_type"] == "enterprise"

    def test_concise_style_adjustments(self) -> None:
        """Vérifie les ajustements pour le style concis."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        mgr = UserProfileManager("test_rag_concise")
        mgr.update_preferences(
            expertise_level="intermediate",
            response_style="concise",
        )

        rec = PersonalizedRecommender("test_rag_concise")
        params = rec.adapt_rag_parameters()

        # La température devrait être réduite
        assert params["temperature"] < 0.3
        assert params["response_style"] == "concise"

    def test_detailed_style_increases_k(self) -> None:
        """Vérifie que le style détaillé augmente k."""
        from user.profile import UserProfileManager
        from user.recommender import PersonalizedRecommender

        mgr = UserProfileManager("test_rag_detailed")
        mgr.update_preferences(
            expertise_level="intermediate",
            response_style="detailed",
        )

        rec = PersonalizedRecommender("test_rag_detailed")
        params = rec.adapt_rag_parameters()

        assert params["k"] >= 5  # Au moins le k de base


class TestPromptPersonalization:
    """Tests de la personnalisation des prompts."""

    def test_prompt_template_has_adaptive_instructions(self) -> None:
        """Vérifie que le template RAG contient les instructions adaptatives."""
        from rag.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        template = pb.rag_template.template

        assert "beginner" in template
        assert "expert" in template
        assert "enterprise" in template
        assert "concise" in template
        assert "step_by_step" in template

    def test_prompt_format_with_enterprise_profile(self) -> None:
        """Vérifie le formatage du prompt avec un profil entreprise."""
        from rag.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        prompt = pb.build_rag_prompt(
            question="Quelle est la politique de sécurité ?",
            context="Document de politique...",
            user_profile="Type d'utilisateur : enterprise\nSecteur : finance",
        )

        assert "enterprise" in prompt
        assert "finance" in prompt
        assert "politique de sécurité" in prompt
