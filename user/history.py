"""
user/history.py — Historique des interactions utilisateur.

Stocke et récupère l'historique des interactions entre
l'utilisateur et le système pour l'analyse et la personnalisation.
"""

from typing import Optional
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Integer, Float
from pydantic import BaseModel

from user.db import Base


# === Modèle SQLAlchemy (table) ===

class InteractionModel(Base):
    """Modèle SQLAlchemy pour la table des interactions."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources_count = Column(Integer, default=0)
    model_used = Column(String(50), default="")
    response_quality = Column(Float, nullable=True)  # Note de qualité (0-1)
    feedback = Column(String(20), nullable=True)  # "positive", "negative", None
    created_at = Column(DateTime, default=datetime.utcnow)


# === Schéma Pydantic ===

class Interaction(BaseModel):
    """Schéma Pydantic pour valider une interaction."""

    user_id: str
    question: str
    answer: str
    sources_count: int = 0
    model_used: str = ""
    response_quality: Optional[float] = None
    feedback: Optional[str] = None


class InteractionHistory:
    """
    Gère l'historique des interactions utilisateur.

    Permet de sauvegarder, récupérer et analyser les interactions
    passées pour améliorer la personnalisation.
    """

    def __init__(self, db_manager) -> None:
        """
        Args:
            db_manager: Instance de DatabaseManager.
        """
        self.db = db_manager

    def save_interaction(self, interaction: Interaction) -> InteractionModel:
        """
        Sauvegarde une interaction dans la base de données.

        Args:
            interaction: Données de l'interaction.

        Returns:
            Le modèle créé.
        """
        session = self.db.get_session()
        try:
            db_interaction = InteractionModel(
                user_id=interaction.user_id,
                question=interaction.question,
                answer=interaction.answer,
                sources_count=interaction.sources_count,
                model_used=interaction.model_used,
                response_quality=interaction.response_quality,
                feedback=interaction.feedback,
            )
            session.add(db_interaction)
            session.commit()
            session.refresh(db_interaction)
            return db_interaction
        finally:
            session.close()

    def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[InteractionModel]:
        """
        Récupère l'historique des interactions d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur.
            limit: Nombre maximum d'interactions à retourner.

        Returns:
            Liste des interactions, les plus récentes en premier.
        """
        session = self.db.get_session()
        try:
            return (
                session.query(InteractionModel)
                .filter_by(user_id=user_id)
                .order_by(InteractionModel.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_stats(self, user_id: str) -> dict:
        """
        Calcule des statistiques sur les interactions d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur.

        Returns:
            Dictionnaire de statistiques.
        """
        session = self.db.get_session()
        try:
            interactions = (
                session.query(InteractionModel)
                .filter_by(user_id=user_id)
                .all()
            )

            total = len(interactions)
            positive = sum(1 for i in interactions if i.feedback == "positive")
            negative = sum(1 for i in interactions if i.feedback == "negative")

            return {
                "total_interactions": total,
                "positive_feedback": positive,
                "negative_feedback": negative,
                "satisfaction_rate": positive / total if total > 0 else 0.0,
            }
        finally:
            session.close()

    def add_feedback(self, interaction_id: int, feedback: str) -> bool:
        """
        Ajoute un feedback à une interaction existante.

        Args:
            interaction_id: ID de l'interaction.
            feedback: "positive" ou "negative".

        Returns:
            True si le feedback a été ajouté.
        """
        session = self.db.get_session()
        try:
            interaction = session.query(InteractionModel).filter_by(id=interaction_id).first()
            if interaction:
                interaction.feedback = feedback
                session.commit()
                return True
            return False
        finally:
            session.close()
