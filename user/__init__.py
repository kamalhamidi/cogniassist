"""
Module user — Gestion des profils et personnalisation.

Fournit les gestionnaires de profils, d'historique et de
recommandations. Inclut les services ACPE (Adaptive Cognitive
Profiling Engine). Initialise la base de données au chargement.

Usage :
    from user import get_user_manager, get_interaction_history
    manager = get_user_manager("default")
    profile = manager.get_profile()
"""

import logging

from user.db import init_db, reset_system
from user.profile import UserProfileManager
from user.history import InteractionHistory
from user.recommender import PersonalizedRecommender
# ACPE models — importés pour que create_all() les détecte
from user.acpe_models import KnowledgeProfile, UsagePattern, ProgressivePrompt
from user.knowledge_engine import KnowledgeProfileEngine
from user.profile_evolution import ProfileEvolutionEngine
from user.progressive import ProgressiveProfilingEngine

logger = logging.getLogger("cogniassist.user")

__all__ = [
    "UserProfileManager", "InteractionHistory", "PersonalizedRecommender",
    "KnowledgeProfileEngine", "ProfileEvolutionEngine", "ProgressiveProfilingEngine",
    "KnowledgeProfile", "UsagePattern", "ProgressivePrompt",
    "get_user_manager", "get_interaction_history", "get_recommender",
    "get_knowledge_engine", "get_evolution_engine", "get_progressive_engine",
    "reset_system",
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


def get_knowledge_engine(user_id: str = "default") -> KnowledgeProfileEngine:
    """
    Retourne un KnowledgeProfileEngine pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance de KnowledgeProfileEngine.
    """
    return KnowledgeProfileEngine(user_id)


def get_evolution_engine(user_id: str = "default") -> ProfileEvolutionEngine:
    """
    Retourne un ProfileEvolutionEngine pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance de ProfileEvolutionEngine.
    """
    return ProfileEvolutionEngine(user_id)


def get_progressive_engine(user_id: str = "default") -> ProgressiveProfilingEngine:
    """
    Retourne un ProgressiveProfilingEngine pour l'utilisateur donné.

    Args:
        user_id: Identifiant de l'utilisateur.

    Returns:
        Instance de ProgressiveProfilingEngine.
    """
    return ProgressiveProfilingEngine(user_id)
