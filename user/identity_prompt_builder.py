"""
user/identity_prompt_builder.py — Construction du prompt système d'identité.

Assemble un prompt système « second cerveau » à partir de :
- style_profile          : comment l'utilisateur écrit ;
- croyances pertinentes  : ce que l'utilisateur pense de ce sujet ;
- profil de connaissances: ce que l'utilisateur maîtrise.

Étend le PromptBuilder existant pour le mode identité.
"""

import logging

from config import settings
from user.db import get_session

logger = logging.getLogger("cogniassist.user")

_DEFAULT_STYLE_FRAGMENT = (
    "Écris de façon claire et naturelle, avec un ton analytique équilibré. "
    "Adapte la longueur au besoin et reste fidèle à une voix personnelle."
)


class IdentityPromptBuilder:
    """
    Assemble le prompt système conscient de l'identité (mode second cerveau).
    """

    def __init__(
        self,
        style_analyzer,
        belief_extractor,
        profile_manager,
        session=None,
    ) -> None:
        """
        Args:
            style_analyzer: Instance StyleAnalyzer.
            belief_extractor: Instance BeliefExtractor.
            profile_manager: UserProfileManager existant.
            session: Session SQLAlchemy (optionnelle).
        """
        self.style_analyzer = style_analyzer
        self.belief_extractor = belief_extractor
        self.profile_manager = profile_manager
        self.session = session or get_session()

    # ─────────────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────────────

    def build_system_prompt(self, query: str) -> str:
        """
        Méthode principale, appelée au moment de la requête.

        Args:
            query: La question de l'utilisateur.

        Returns:
            Le prompt système complet à injecter dans l'appel LLM.
        """
        style = self._get_style_fragment()
        beliefs = self._get_relevant_beliefs(query)
        expertise = self._get_expertise_context()
        return self._assemble(style, beliefs, expertise)

    # ─────────────────────────────────────────────────────────────────
    # Composants
    # ─────────────────────────────────────────────────────────────────

    def _get_style_fragment(self) -> str:
        """Retourne le fragment de style, ou un fragment générique par défaut."""
        try:
            profile = self.style_analyzer.get_profile(self.session)
            if profile and profile.get("style_prompt_fragment"):
                return profile["style_prompt_fragment"]
        except Exception as e:
            logger.debug("Lecture fragment de style échouée : %s", e)
        return _DEFAULT_STYLE_FRAGMENT

    def _get_relevant_beliefs(self, query: str) -> str:
        """Formate les croyances pertinentes en bloc lisible (vide si aucune)."""
        try:
            beliefs = self.belief_extractor.get_beliefs_for_topic(query)
        except Exception as e:
            logger.debug("Lecture croyances pertinentes échouée : %s", e)
            beliefs = []

        if not beliefs:
            return ""

        lines: list[str] = []
        for b in beliefs:
            conf = {"high": "forte", "medium": "modérée", "low": "faible"}.get(
                b.get("confidence", "medium"), "modérée"
            )
            date_str = f" (écrit le {b['date_written']})" if b.get("date_written") else ""
            lines.append(
                f"- Sur « {b['topic']} » : {b['position']} "
                f"[conviction {conf}{date_str}]"
            )
        return "\n".join(lines)

    def _get_expertise_context(self) -> str:
        """Résumé court des domaines d'expertise via KnowledgeProfileEngine."""
        try:
            from user.knowledge_engine import KnowledgeProfileEngine
            user_id = getattr(self.profile_manager, "user_id", "default")
            engine = KnowledgeProfileEngine(user_id)
            top = engine.get_top_domains(limit=5)
            if top:
                return "Domaines de prédilection : " + ", ".join(top) + "."
        except Exception as e:
            logger.debug("Lecture expertise échouée : %s", e)
        return "Aucun domaine d'expertise dominant identifié pour l'instant."

    def _assemble(self, style: str, beliefs: str, expertise: str) -> str:
        """Assemble le prompt système final."""
        try:
            profile = self.profile_manager.get_profile()
            user_name = profile.get("name", "l'utilisateur")
        except Exception:
            user_name = "l'utilisateur"

        beliefs_block = beliefs or "Aucune position connue sur ce sujet spécifique."

        return f"""Tu es le second cerveau de {user_name}. Réponds exactement comme il/elle le ferait, en te basant uniquement sur ce que tu sais réellement de lui/elle.

=== STYLE D'ÉCRITURE ===
{style}

=== SES POSITIONS SUR CE SUJET ===
{beliefs_block}

=== SES DOMAINES D'EXPERTISE ===
{expertise}

=== RÈGLES ABSOLUES ===
1. N'invente jamais une opinion non présente dans ses écrits.
2. Si tu n'as pas de position connue, dis-le explicitement.
3. Marque toute affirmation incertaine avec [non confirmé].
4. Réponds dans la langue de la question."""

    # ─────────────────────────────────────────────────────────────────
    # Disponibilité
    # ─────────────────────────────────────────────────────────────────

    def is_identity_mode_ready(self) -> bool:
        """
        True seulement si assez de données existent pour le mode identité :
        un style_profile existe ET au moins MIN_BELIEFS_FOR_IDENTITY_MODE
        croyances actives/confirmées sont présentes.
        """
        try:
            profile = self.style_analyzer.get_profile(self.session)
            if not profile:
                return False

            from user.identity_models import BeliefStore
            count = (
                self.session.query(BeliefStore)
                .filter(BeliefStore.status.in_(["active", "user_confirmed", "conflicted"]))
                .count()
            )
            return count >= settings.MIN_BELIEFS_FOR_IDENTITY_MODE
        except Exception as e:
            logger.debug("Vérification disponibilité mode identité échouée : %s", e)
            return False
