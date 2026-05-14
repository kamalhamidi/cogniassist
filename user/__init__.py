"""
Module user — Gestion des profils et personnalisation.

Ce module gère les profils utilisateurs, l'historique des interactions,
les recommandations personnalisées et la connexion à la base SQLite.
"""

from user.profile import UserProfile
from user.history import InteractionHistory
from user.recommender import Recommender
from user.db import DatabaseManager

__all__ = ["UserProfile", "InteractionHistory", "Recommender", "DatabaseManager"]
