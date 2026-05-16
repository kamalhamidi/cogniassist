"""
Module user — Gestion des profils et personnalisation.

Fournit les gestionnaires de profils, d'historique et de
recommandations. Initialise la base de données au chargement.

Usage :
    from user import get_user_manager, get_interaction_history
    manager = get_user_manager("default")
    profile = manager.get_profile()
"""

import logging

from user.db import init_db
from user.profile import UserProfileManager
from user.history import InteractionHistory
from user.recommender import PersonalizedRecommender

logger = logging.getLogger("cogniassist.user")

__all__ = [
    "UserProfileManager", "InteractionHistory", "PersonalizedRecommender",
    "get_user_manager", "get_interaction_history", "get_recommender",
]

# Initialiser les tables au chargement du module
init_db()


def get_user_manager(user_id: str = "default") -> UserProfileManager:
    """
    Retourne un UserProfileManager pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance de UserProfileManager.
    """
    return UserProfileManager(user_id)


def get_interaction_history(user_id: str = "default") -> InteractionHistory:
    """
    Retourne un InteractionHistory pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance d'InteractionHistory.
    """
    return InteractionHistory(user_id)


def get_recommender(user_id: str = "default") -> PersonalizedRecommender:
    """
    Retourne un PersonalizedRecommender pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance de PersonalizedRecommender.
    """
    return PersonalizedRecommender(user_id)
