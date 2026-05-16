"""
user/history.py — Historique des interactions utilisateur.

Stocke chaque question-réponse avec sources, temps de réponse et
feedback. Fournit des statistiques et l'extraction de sujets fréquents.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey

from user.db import Base, get_session

logger = logging.getLogger("cogniassist.user")

# Mots vides à ignorer pour l'extraction de sujets
STOPWORDS = frozenset([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "à", "au", "aux", "ce", "qui", "que", "par", "sur", "dans",
    "est", "sont", "avec", "pour", "pas", "ne", "se", "sa", "son",
    "the", "a", "an", "of", "in", "is", "are", "and", "to",
    "for", "with", "on", "it", "this", "that", "was", "be",
    "quels", "quel", "quelle", "quelles", "comment", "pourquoi",
    "peut", "peux", "mes", "mon", "ma", "nos", "ton", "ta",
    "tu", "je", "il", "elle", "nous", "vous", "ils", "elles",
])


# ═══════════════════════════════════════════════════════════
# Modèle SQLAlchemy
# ═══════════════════════════════════════════════════════════

class Interaction(Base):
    """Table des interactions question-réponse."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
        index=True,
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text, default="[]")
    chunks_used = Column(Integer, default=0)
    response_time_ms = Column(Integer, default=0)
    feedback = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ═══════════════════════════════════════════════════════════
# Gestionnaire d'historique
# ═══════════════════════════════════════════════════════════

class InteractionHistory:
    """
    Gère l'historique des interactions pour un utilisateur.

    Sauvegarde les échanges, le feedback, et fournit des statistiques
    et l'extraction de sujets fréquents.
    """

    def __init__(self, user_id: str = "default") -> None:
        """
        Initialise le gestionnaire d'historique.

        Args:
            user_id: Identifiant de l'utilisateur.
        """
        self.user_id = user_id
        self.session = get_session()

    def save_interaction(
        self,
        question: str,
        answer: str,
        sources: list[str],
        chunks_used: int,
        response_time_ms: int = 0,
    ) -> int:
        """
        Sauvegarde une interaction dans la base de données.

        Args:
            question: Question posée par l'utilisateur.
            answer: Réponse générée par le LLM.
            sources: Liste des noms de fichiers sources utilisés.
            chunks_used: Nombre de chunks récupérés.
            response_time_ms: Temps de réponse en millisecondes.

        Returns:
            ID de l'interaction créée.
        """
        try:
            interaction = Interaction(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=json.dumps(sources, ensure_ascii=False),
                chunks_used=chunks_used,
                response_time_ms=response_time_ms,
            )
            self.session.add(interaction)
            self.session.commit()
            self.session.refresh(interaction)
            logger.debug("Interaction #%d sauvegardée.", interaction.id)
            return interaction.id
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur sauvegarde interaction : %s", e)
            return -1

    def save_feedback(self, interaction_id: int, feedback: int) -> None:
        """
        Enregistre le feedback utilisateur pour une interaction.

        Args:
            interaction_id: ID de l'interaction.
            feedback: 1 (positif) ou -1 (négatif).
        """
        try:
            interaction = (
                self.session.query(Interaction)
                .filter_by(id=interaction_id)
                .first()
            )
            if interaction is None:
                logger.warning("Interaction #%d introuvable pour feedback.", interaction_id)
                return
            interaction.feedback = feedback
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur sauvegarde feedback : %s", e)

    def get_recent_history(self, limit: int = 20) -> list[dict]:
        """
        Retourne les dernières interactions de cet utilisateur.

        Args:
            limit: Nombre maximum d'interactions à retourner.

        Returns:
            Liste de dictionnaires ordonnés par date décroissante.
        """
        interactions = (
            self.session.query(Interaction)
            .filter_by(user_id=self.user_id)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_dict(i) for i in interactions]

    def get_interaction_stats(self) -> dict:
        """
        Retourne les statistiques globales des interactions.

        Returns:
            Dictionnaire avec total, temps moyen, feedback, et compteurs.
        """
        all_interactions = (
            self.session.query(Interaction)
            .filter_by(user_id=self.user_id)
            .all()
        )

        total = len(all_interactions)
        if total == 0:
            return {
                "total_interactions": 0,
                "avg_response_time_ms": 0.0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "most_used_sources": [],
                "interactions_today": 0,
                "interactions_week": 0,
            }

        # Temps de réponse moyen
        times = [i.response_time_ms for i in all_interactions if i.response_time_ms]
        avg_time = round(sum(times) / len(times), 1) if times else 0.0

        # Feedback
        positive = sum(1 for i in all_interactions if i.feedback == 1)
        negative = sum(1 for i in all_interactions if i.feedback == -1)

        # Sources les plus utilisées
        source_counter: Counter = Counter()
        for i in all_interactions:
            try:
                srcs = json.loads(i.sources) if i.sources else []
                source_counter.update(srcs)
            except json.JSONDecodeError:
                pass
        most_used = [s for s, _ in source_counter.most_common(3)]

        # Compteurs temporels
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)

        today_count = sum(
            1 for i in all_interactions
            if i.created_at and i.created_at >= today_start
        )
        week_count = sum(
            1 for i in all_interactions
            if i.created_at and i.created_at >= week_start
        )

        return {
            "total_interactions": total,
            "avg_response_time_ms": avg_time,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "most_used_sources": most_used,
            "interactions_today": today_count,
            "interactions_week": week_count,
        }

    def get_frequent_topics(self, limit: int = 5) -> list[str]:
        """
        Extrait les sujets fréquents des questions passées.

        Tokenise les 50 dernières questions, supprime les mots vides,
        et retourne les mots les plus fréquents.

        Args:
            limit: Nombre de sujets à retourner.

        Returns:
            Liste des mots les plus fréquents.
        """
        interactions = (
            self.session.query(Interaction)
            .filter_by(user_id=self.user_id)
            .order_by(Interaction.created_at.desc())
            .limit(50)
            .all()
        )

        word_counter: Counter = Counter()
        for i in interactions:
            words = i.question.lower().split()
            for word in words:
                # Nettoyer la ponctuation
                clean = word.strip("?!.,;:'\"()[]{}«»")
                if clean and clean not in STOPWORDS and len(clean) > 2:
                    word_counter[clean] += 1

        return [word for word, _ in word_counter.most_common(limit)]

    def search_history(self, query: str) -> list[dict]:
        """
        Recherche dans l'historique par correspondance partielle.

        Args:
            query: Terme de recherche.

        Returns:
            Liste d'interactions correspondantes (max 10).
        """
        interactions = (
            self.session.query(Interaction)
            .filter(
                Interaction.user_id == self.user_id,
                Interaction.question.ilike(f"%{query}%"),
            )
            .order_by(Interaction.created_at.desc())
            .limit(10)
            .all()
        )
        return [self._to_dict(i) for i in interactions]

    def clear_history(self) -> None:
        """Supprime tout l'historique de cet utilisateur."""
        try:
            self.session.query(Interaction).filter_by(
                user_id=self.user_id
            ).delete()
            self.session.commit()
            logger.warning("Historique vidé pour l'utilisateur '%s'.", self.user_id)
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur clear history : %s", e)

    @staticmethod
    def _to_dict(interaction: Interaction) -> dict:
        """Convertit une Interaction en dictionnaire."""
        try:
            sources = json.loads(interaction.sources) if interaction.sources else []
        except json.JSONDecodeError:
            sources = []

        return {
            "id": interaction.id,
            "question": interaction.question,
            "answer": interaction.answer,
            "sources": sources,
            "chunks_used": interaction.chunks_used or 0,
            "response_time_ms": interaction.response_time_ms or 0,
            "feedback": interaction.feedback,
            "created_at": interaction.created_at.isoformat() if interaction.created_at else "",
        }
