"""
user/progressive.py — Moteur de profiling progressif ACPE.

Génère des suggestions contextuelles basées sur le comportement
de l'utilisateur, avec un mécanisme de cooldown pour limiter
les interruptions.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from user.db import get_session
from user.acpe_models import ProgressivePrompt
from user.history import InteractionHistory
from user.knowledge_engine import KnowledgeProfileEngine
from user.profile import UserProfileManager

logger = logging.getLogger("cogniassist.user")

# Nombre minimum d'interactions entre deux suggestions
COOLDOWN_INTERACTIONS = 5
# Nombre minimum d'interactions avant la première suggestion
MIN_INTERACTIONS_BEFORE_FIRST = 10


class ProgressiveProfilingEngine:
    """Moteur de profiling progressif avec cooldown intelligent.

    Génère des suggestions légères basées sur le comportement
    et respecte un cooldown pour ne pas interrompre l'utilisateur.
    """

    def __init__(self, user_id: str) -> None:
        """Initialise le moteur de profiling progressif."""
        self.user_id = user_id
        self.session = get_session()
        self.history = InteractionHistory(user_id)
        self.knowledge_engine = KnowledgeProfileEngine(user_id)
        self.profile_manager = UserProfileManager(user_id)

    def check_for_prompts(self) -> Optional[dict]:
        """Vérifie s'il y a une suggestion en attente à afficher.

        Respecte le cooldown et ne retourne qu'une suggestion à la fois.

        Returns:
            Dict de la suggestion, ou None si rien à afficher.
        """
        # Vérifier si l'apprentissage adaptatif est activé
        profile = self.profile_manager.get_profile()
        if not profile.get("adaptive_learning_enabled", True):
            return None

        # Vérifier le cooldown
        if not self._is_cooldown_ok():
            return None

        # Chercher une suggestion en attente
        pending = (
            self.session.query(ProgressivePrompt)
            .filter_by(user_id=self.user_id, status="pending")
            .order_by(ProgressivePrompt.created_at.asc())
            .first()
        )

        if pending:
            return pending.to_dict()

        # Sinon, essayer de générer de nouvelles suggestions
        self._generate_prompts()

        # Re-vérifier
        pending = (
            self.session.query(ProgressivePrompt)
            .filter_by(user_id=self.user_id, status="pending")
            .order_by(ProgressivePrompt.created_at.asc())
            .first()
        )
        return pending.to_dict() if pending else None

    def accept_prompt(self, prompt_id: int) -> None:
        """Accepte une suggestion et applique l'action associée.

        Args:
            prompt_id: ID de la suggestion à accepter.
        """
        try:
            prompt = (
                self.session.query(ProgressivePrompt)
                .filter_by(id=prompt_id, user_id=self.user_id)
                .first()
            )
            if not prompt:
                return

            prompt.status = "accepted"
            prompt.resolved_at = datetime.utcnow()

            # Appliquer l'action
            action_data = prompt.get_action_data()
            action_type = action_data.get("type")

            if action_type == "add_interest":
                domain = action_data.get("domain", "")
                if domain:
                    profile = self.profile_manager.get_profile()
                    current_domains = profile.get("domain_focus", [])
                    if domain not in current_domains:
                        current_domains.append(domain)
                        self.profile_manager.update_preferences(
                            domain_focus=current_domains,
                        )

            elif action_type == "change_style":
                new_style = action_data.get("style", "")
                if new_style:
                    self.profile_manager.update_preferences(
                        response_style=new_style,
                    )

            elif action_type == "update_expertise":
                new_level = action_data.get("level", "")
                if new_level:
                    self.profile_manager.update_preferences(
                        expertise_level=new_level,
                    )

            self.session.commit()
            logger.info(
                "Suggestion #%d acceptée pour '%s' (action=%s).",
                prompt_id, self.user_id, action_type,
            )
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur accept prompt : %s", e)

    def decline_prompt(self, prompt_id: int) -> None:
        """Décline une suggestion.

        Args:
            prompt_id: ID de la suggestion à décliner.
        """
        try:
            prompt = (
                self.session.query(ProgressivePrompt)
                .filter_by(id=prompt_id, user_id=self.user_id)
                .first()
            )
            if prompt:
                prompt.status = "declined"
                prompt.resolved_at = datetime.utcnow()
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur decline prompt : %s", e)

    def _generate_prompts(self) -> None:
        """Analyse le comportement et génère des suggestions pertinentes."""
        try:
            profile = self.profile_manager.get_profile()
            stats = self.history.get_interaction_stats()
            total = stats.get("total_interactions", 0)

            if total < MIN_INTERACTIONS_BEFORE_FIRST:
                return

            # Règle 1 : Nouveau domaine fréquent non dans les intérêts
            self._check_new_domain_interest(profile)

            # Règle 2 : Préférence de style de réponse
            self._check_response_style_preference(profile, stats)

        except Exception as e:
            logger.debug("Erreur generate prompts : %s", e)

    def _check_new_domain_interest(self, profile: dict) -> None:
        """Vérifie si un domaine fréquent n'est pas dans les intérêts."""
        top_domains = self.knowledge_engine.get_top_domains(limit=3)
        current_interests = profile.get("domain_focus", [])

        for domain in top_domains:
            # Vérifier si ce domaine n'est pas déjà dans les intérêts
            if domain.lower() not in [i.lower() for i in current_interests]:
                # Vérifier qu'on n'a pas déjà posé cette suggestion
                prompt_key = f"add_interest_{domain.lower().replace(' ', '_')}"
                existing = (
                    self.session.query(ProgressivePrompt)
                    .filter_by(
                        user_id=self.user_id,
                        prompt_key=prompt_key,
                    )
                    .first()
                )
                if existing:
                    continue

                # Obtenir le pourcentage
                knowledge = self.knowledge_engine.get_knowledge_profile()
                domain_info = next(
                    (k for k in knowledge if k["domain"] == domain), None
                )
                count = domain_info["interaction_count"] if domain_info else 0

                self._create_prompt(
                    prompt_key=prompt_key,
                    message=(
                        f"📊 Nous avons remarqué que vous posez souvent des questions "
                        f"sur **{domain}** ({count} interactions). "
                        f"Ajouter ce domaine à vos centres d'intérêt ?"
                    ),
                    action_data={
                        "type": "add_interest",
                        "domain": domain,
                    },
                )
                break  # Une seule suggestion à la fois

    def _check_response_style_preference(
        self, profile: dict, stats: dict,
    ) -> None:
        """Vérifie si le style de réponse correspond au comportement."""
        from user.acpe_models import UsagePattern

        try:
            avg_pattern = (
                self.session.query(UsagePattern)
                .filter_by(
                    user_id=self.user_id,
                    metric_name="avg_answer_length",
                )
                .first()
            )
            if not avg_pattern:
                return

            avg_data = avg_pattern.get_value()
            if not isinstance(avg_data, dict):
                return

            avg_len = avg_data.get("average", 0)
            count = avg_data.get("count", 0)
            current_style = profile.get("response_style", "detailed")

            if count < 15:
                return

            # Détecter un décalage entre le style actuel et le comportement
            suggested_style = None
            if avg_len < 40 and current_style in ("detailed", "academic"):
                suggested_style = "concise"
                message = (
                    "✂️ Vos interactions montrent une préférence pour des "
                    "réponses **courtes et directes**. "
                    "Changer votre style de réponse en « Concis » ?"
                )
            elif avg_len > 200 and current_style in ("concise", "simple"):
                suggested_style = "detailed"
                message = (
                    "📝 Vos interactions montrent une préférence pour des "
                    "réponses **détaillées**. "
                    "Changer votre style de réponse en « Détaillé » ?"
                )

            if suggested_style:
                prompt_key = f"change_style_{suggested_style}"
                existing = (
                    self.session.query(ProgressivePrompt)
                    .filter_by(
                        user_id=self.user_id,
                        prompt_key=prompt_key,
                    )
                    .first()
                )
                if not existing:
                    self._create_prompt(
                        prompt_key=prompt_key,
                        message=message,
                        action_data={
                            "type": "change_style",
                            "style": suggested_style,
                        },
                    )

        except Exception as e:
            logger.debug("Erreur check response style : %s", e)

    def _create_prompt(
        self, prompt_key: str, message: str, action_data: dict,
    ) -> None:
        """Crée une suggestion dans la base."""
        try:
            prompt = ProgressivePrompt(
                user_id=self.user_id,
                prompt_key=prompt_key,
                message=message,
                action_data=json.dumps(action_data, ensure_ascii=False),
                status="pending",
                created_at=datetime.utcnow(),
            )
            self.session.add(prompt)
            self.session.commit()
            logger.debug(
                "Suggestion '%s' créée pour '%s'.", prompt_key, self.user_id,
            )
        except Exception as e:
            self.session.rollback()
            logger.debug("Erreur create prompt : %s", e)

    def _is_cooldown_ok(self) -> bool:
        """Vérifie si le cooldown entre suggestions est respecté.

        Returns:
            True si on peut afficher une suggestion.
        """
        try:
            stats = self.history.get_interaction_stats()
            total = stats.get("total_interactions", 0)

            if total < MIN_INTERACTIONS_BEFORE_FIRST:
                return False

            # Trouver la dernière suggestion résolue
            last_resolved = (
                self.session.query(ProgressivePrompt)
                .filter_by(user_id=self.user_id)
                .filter(ProgressivePrompt.status.in_(["accepted", "declined"]))
                .order_by(ProgressivePrompt.resolved_at.desc())
                .first()
            )

            if last_resolved and last_resolved.resolved_at:
                # Compter les interactions depuis la dernière résolution
                from user.history import Interaction
                interactions_since = (
                    self.session.query(Interaction)
                    .filter(
                        Interaction.user_id == self.user_id,
                        Interaction.created_at > last_resolved.resolved_at,
                    )
                    .count()
                )
                return interactions_since >= COOLDOWN_INTERACTIONS

            return True
        except Exception:
            return False
