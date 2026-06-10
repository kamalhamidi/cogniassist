"""
user/profile_evolution.py — Moteur d'évolution du profil ACPE.

Analyse les interactions pour détecter les changements de comportement,
mettre à jour les patterns d'utilisation, et générer des insights
lisibles sur l'activité de l'utilisateur.
"""

import json
import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from user.db import get_session
from user.acpe_models import UsagePattern
from user.history import InteractionHistory
from user.knowledge_engine import KnowledgeProfileEngine

logger = logging.getLogger("cogniassist.user")


class ProfileEvolutionEngine:
    """Moteur d'évolution du profil basé sur l'analyse comportementale.

    Détecte les tendances d'utilisation, génère des insights,
    et maintient des métriques comportementales à jour.
    """

    def __init__(self, user_id: str) -> None:
        """Initialise le moteur d'évolution pour un utilisateur."""
        self.user_id = user_id
        self.session = get_session()
        self.history = InteractionHistory(user_id)
        self.knowledge_engine = KnowledgeProfileEngine(user_id)

    def analyze_interaction(
        self,
        question: str,
        answer: str,
        sources: list[str],
        feedback: Optional[int] = None,
        response_time_ms: int = 0,
    ) -> None:
        """Point d'entrée principal : analyse une interaction et met à jour le profil.

        Args:
            question: Question posée.
            answer: Réponse générée.
            sources: Fichiers sources utilisés.
            feedback: 1 (positif), -1 (négatif), ou None.
            response_time_ms: Temps de réponse en ms.
        """
        # 1. Mettre à jour le profil de connaissances
        self.knowledge_engine.update_from_interaction(
            question=question,
            answer=answer,
            sources=sources,
            feedback=feedback,
        )

        # 2. Mettre à jour les patterns d'utilisation
        self._update_usage_patterns(answer, response_time_ms)

        # 3. Détecter les changements d'intérêts
        self._detect_interest_shifts()

    def _update_usage_patterns(
        self, answer: str, response_time_ms: int,
    ) -> None:
        """Met à jour les métriques d'utilisation comportementales.

        Tracks: nombre total de questions, longueur de réponse moyenne,
        temps de réponse moyen, heure d'activité préférée.
        """
        now = datetime.utcnow()

        # Total questions
        self._increment_metric("total_questions", 1)

        # Longueur de réponse (pour détecter la préférence courte/longue)
        answer_len = len(answer.split())
        self._update_running_average("avg_answer_length", answer_len)

        # Temps de réponse moyen
        if response_time_ms > 0:
            self._update_running_average("avg_response_time_ms", response_time_ms)

        # Heure d'activité
        hour = now.hour
        self._update_frequency_map("active_hours", str(hour))

    def _detect_interest_shifts(self) -> None:
        """Détecte les nouveaux centres d'intérêt émergents.

        Compare les domaines récents (dernières 20 interactions)
        avec les domaines historiques pour détecter les nouveautés.
        """
        try:
            recent = self.history.get_recent_history(limit=20)
            if len(recent) < 5:
                return

            # Extraire les domaines des questions récentes
            recent_domains: Counter = Counter()
            for interaction in recent:
                domains = self.knowledge_engine._extract_domains(
                    interaction["question"]
                )
                recent_domains.update(domains)

            # Stocker les domaines émergents
            if recent_domains:
                top_recent = [d for d, _ in recent_domains.most_common(5)]
                self._set_metric("emerging_domains", top_recent)
        except Exception as e:
            logger.debug("Erreur detect interest shifts : %s", e)

    def generate_insights(self) -> list[str]:
        """Génère des insights lisibles sur le comportement de l'utilisateur.

        Returns:
            Liste de phrases d'insight en français.
        """
        insights: list[str] = []

        # Insight 1 : Domaines fréquents
        top_domains = self.knowledge_engine.get_top_domains(limit=3)
        if top_domains:
            domains_str = ", ".join(top_domains[:2])
            insights.append(
                f"📊 Vous posez fréquemment des questions sur {domains_str}."
            )

        # Insight 2 : Domaines à renforcer
        weak = self.knowledge_engine.get_weak_domains(limit=2)
        if weak:
            insights.append(
                f"📈 Axes de progression suggérés : {', '.join(weak)}."
            )

        # Insight 3 : Préférence de longueur de réponse
        avg_len = self._get_metric("avg_answer_length")
        if avg_len is not None:
            if isinstance(avg_len, dict):
                avg_val = avg_len.get("average", 0)
            else:
                avg_val = float(avg_len)
            if avg_val < 50:
                insights.append(
                    "✂️ Vos réponses sont généralement courtes — "
                    "vous semblez préférer des réponses concises."
                )
            elif avg_val > 150:
                insights.append(
                    "📝 Vos réponses sont détaillées — "
                    "vous semblez apprécier les explications approfondies."
                )

        # Insight 4 : Domaines émergents
        emerging = self._get_metric("emerging_domains")
        if emerging and isinstance(emerging, list) and len(emerging) > 0:
            new_domain = emerging[0]
            if new_domain not in top_domains:
                insights.append(
                    f"🆕 Nouveau centre d'intérêt détecté : {new_domain}."
                )

        # Insight 5 : Activité
        total = self._get_metric("total_questions")
        if total:
            total_val = total if isinstance(total, (int, float)) else 0
            if total_val >= 50:
                insights.append(
                    f"🏆 Vous avez posé {int(total_val)} questions — "
                    f"utilisateur assidu !"
                )
            elif total_val >= 10:
                insights.append(
                    f"💡 {int(total_val)} questions posées — "
                    f"continuez votre progression !"
                )

        return insights

    def get_evolution_summary(self) -> dict:
        """Retourne un résumé complet de l'évolution du profil.

        Returns:
            Dictionnaire avec knowledge, patterns, insights et domains.
        """
        return {
            "knowledge_profile": self.knowledge_engine.get_knowledge_profile(),
            "top_domains": self.knowledge_engine.get_top_domains(limit=5),
            "weak_domains": self.knowledge_engine.get_weak_domains(limit=3),
            "insights": self.generate_insights(),
            "usage_patterns": self._get_all_metrics(),
        }

    # ═══════════════════════════════════════════════════════════
    # Helpers pour les métriques
    # ═══════════════════════════════════════════════════════════

    def _set_metric(self, name: str, value: any) -> None:
        """Définit ou met à jour une métrique."""
        try:
            pattern = (
                self.session.query(UsagePattern)
                .filter_by(user_id=self.user_id, metric_name=name)
                .first()
            )
            if pattern is None:
                pattern = UsagePattern(
                    user_id=self.user_id,
                    metric_name=name,
                )
                self.session.add(pattern)

            pattern.set_value(value)
            pattern.updated_at = datetime.utcnow()
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.debug("Erreur set metric '%s' : %s", name, e)

    def _get_metric(self, name: str) -> any:
        """Récupère la valeur d'une métrique."""
        try:
            pattern = (
                self.session.query(UsagePattern)
                .filter_by(user_id=self.user_id, metric_name=name)
                .first()
            )
            return pattern.get_value() if pattern else None
        except Exception:
            return None

    def _get_all_metrics(self) -> dict:
        """Récupère toutes les métriques de l'utilisateur."""
        try:
            patterns = (
                self.session.query(UsagePattern)
                .filter_by(user_id=self.user_id)
                .all()
            )
            return {p.metric_name: p.get_value() for p in patterns}
        except Exception:
            return {}

    def _increment_metric(self, name: str, increment: int = 1) -> None:
        """Incrémente un compteur métrique."""
        current = self._get_metric(name)
        if current is None or not isinstance(current, (int, float)):
            current = 0
        self._set_metric(name, current + increment)

    def _update_running_average(self, name: str, new_value: float) -> None:
        """Met à jour une moyenne glissante."""
        current = self._get_metric(name)
        if current is None or not isinstance(current, dict):
            current = {"average": 0.0, "count": 0}

        count = current.get("count", 0) + 1
        old_avg = current.get("average", 0.0)
        new_avg = old_avg + (new_value - old_avg) / count

        self._set_metric(name, {"average": round(new_avg, 1), "count": count})

    def _update_frequency_map(self, name: str, key: str) -> None:
        """Incrémente un compteur dans une map de fréquences."""
        current = self._get_metric(name)
        if current is None or not isinstance(current, dict):
            current = {}

        current[key] = current.get(key, 0) + 1
        self._set_metric(name, current)
