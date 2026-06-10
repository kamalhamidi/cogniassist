"""
tests/test_kmb.py — Tests unitaires pour la fonctionnalité "Know Me Better".

Vérifie l'initialisation du modèle, la mise à jour des champs avec encodage JSON,
la complétion dynamique des questionnaires, et le statut de visibilité.
"""

import json
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Configure une base SQLite temporaire isolée pour chaque test."""
    import user.db as db_module

    # Réinitialiser les singletons du module db
    db_module._engine = None
    db_module._SessionFactory = None

    # Pointer la base vers un fichier temporaire
    test_db = tmp_path / "test_kmb.db"
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


class TestKnowMeBetter:
    """Tests unitaires du gestionnaire KMBManager."""

    def test_kmb_initial_defaults(self) -> None:
        """Vérifie que KMB est initialisé avec des valeurs vides par défaut."""
        from user import get_kmb_manager

        kmb_mgr = get_kmb_manager("test_kmb_user")
        data = kmb_mgr.get_kmb_data()

        assert data["completion_percentage"] == 0
        assert data["gender"] == "Prefer not to say"
        assert data["country"] == ""
        assert data["languages"] == []
        assert data["hobbies"] == []
        assert data["kmb_completed"] is False

    def test_kmb_update_string_fields(self) -> None:
        """Vérifie la mise à jour des champs de type chaîne de caractères."""
        from user import get_kmb_manager

        kmb_mgr = get_kmb_manager("test_kmb_user2")
        pct = kmb_mgr.update_kmb_field("country", "France")
        
        # Le pourcentage doit avoir augmenté
        assert pct > 0
        
        data = kmb_mgr.get_kmb_data()
        assert data["country"] == "France"
        assert data["completion_percentage"] == pct

    def test_kmb_update_list_fields(self) -> None:
        """Vérifie la mise à jour et la sérialisation des listes d'options (JSON)."""
        from user import get_kmb_manager

        kmb_mgr = get_kmb_manager("test_kmb_user3")
        hobbies_list = ["Reading", "Gaming", "Technology"]
        
        pct = kmb_mgr.update_kmb_field("hobbies", hobbies_list)
        assert pct > 0

        data = kmb_mgr.get_kmb_data()
        assert data["hobbies"] == hobbies_list
        assert isinstance(data["hobbies"], list)

    def test_kmb_onboarding_seen_flags(self) -> None:
        """Vérifie la lecture et l'écriture des flags de visibilité de l'onboarding KMB."""
        from user import get_kmb_manager

        kmb_mgr = get_kmb_manager("test_kmb_user4")
        assert kmb_mgr.has_seen_kmb_onboarding() is False

        kmb_mgr.mark_kmb_onboarding_seen()
        assert kmb_mgr.has_seen_kmb_onboarding() is True

    def test_kmb_completion_percentage_calculation(self) -> None:
        """Vérifie que la complétion est calculée précisément sur les 20 champs."""
        from user import get_kmb_manager

        kmb_mgr = get_kmb_manager("test_kmb_user5")
        
        # Aucun champ rempli
        assert kmb_mgr.get_kmb_data()["completion_percentage"] == 0

        # Remplir 1 champ (String)
        kmb_mgr.update_kmb_field("country", "Canada")
        # Remplir 1 champ (List/JSON)
        kmb_mgr.update_kmb_field("languages", ["English", "French"])
        # Remplir 1 champ (Gender - Prefer not to say est compté comme une réponse explicite)
        kmb_mgr.update_kmb_field("gender", "Prefer not to say") # déjà par défaut, mais vérifions
        
        data = kmb_mgr.get_kmb_data()
        # 3 champs sur 20 = 15%
        assert data["completion_percentage"] == 15

        # Remplir tous les autres champs (17 restants)
        fields_to_fill = {
            "date_of_birth": "1995-05-15",
            "gender": "Male", # changer de Prefer not to say
            "hobbies": ["Sports"],
            "interests": "AI and web dev",
            "favorite_topics": "RAG architecture",
            "content_preferences": ["YouTube", "Articles"],
            "followed_communities": "GitHub",
            "current_skills": "Python",
            "future_skills": "Rust",
            "yearly_goals": "Master RAG",
            "learning_style": ["Practical exercises"],
            "learning_frequency": "Daily",
            "occupation": "Student",
            "industry": "Computer Science",
            "challenges": "Time management",
            "motivations": "Curiosity",
            "communication_style": "Short and direct",
            "additional_information": "None"
        }

        for field, val in fields_to_fill.items():
            kmb_mgr.update_kmb_field(field, val)

        data = kmb_mgr.get_kmb_data()
        assert data["completion_percentage"] == 100
        assert data["kmb_completed"] is True
