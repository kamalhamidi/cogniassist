"""
rag/prompt_builder.py — Construction des prompts.

Construit les prompts système et utilisateur pour le pipeline RAG.
Intègre le contexte documentaire, l'historique de conversation
et les préférences utilisateur.
"""

from typing import Optional


class PromptBuilder:
    """
    Constructeur de prompts pour le pipeline RAG.

    Assemble les différentes parties du prompt :
    - Instructions système
    - Contexte documentaire (documents retrouvés)
    - Historique de conversation
    - Contexte utilisateur (profil, préférences)
    """

    DEFAULT_SYSTEM_TEMPLATE: str = """Tu es CogniAssist, un assistant cognitif intelligent et personnalisé.
Tu aides les utilisateurs à comprendre et exploiter leurs documents.

RÈGLES :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
- Si l'information n'est pas dans le contexte, dis-le clairement.
- Cite les sources pertinentes dans ta réponse.
- Réponds dans la langue de la question.
- Sois précis, structuré et utile.

{user_context_section}

CONTEXTE DOCUMENTAIRE :
{context}
"""

    def __init__(self, system_template: Optional[str] = None) -> None:
        """
        Initialise le constructeur de prompts.

        Args:
            system_template: Template personnalisé pour le prompt système.
        """
        self.system_template = system_template or self.DEFAULT_SYSTEM_TEMPLATE

    def build_system_prompt(
        self,
        context: str,
        user_context: Optional[str] = None,
    ) -> str:
        """
        Construit le prompt système avec le contexte documentaire.

        Args:
            context: Le contexte documentaire retrouvé.
            user_context: Informations sur le profil utilisateur.

        Returns:
            Le prompt système formaté.
        """
        user_context_section = ""
        if user_context:
            user_context_section = f"\nPROFIL UTILISATEUR :\n{user_context}\n"

        return self.system_template.format(
            context=context,
            user_context_section=user_context_section,
        )

    @staticmethod
    def build_conversation_prompt(
        question: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Construit le prompt de conversation avec l'historique.

        Args:
            question: La question actuelle de l'utilisateur.
            history: Historique des échanges précédents.

        Returns:
            Le prompt de conversation formaté.
        """
        parts: list[str] = []

        if history:
            parts.append("HISTORIQUE DE CONVERSATION :")
            for exchange in history[-5:]:  # Limiter aux 5 derniers échanges
                parts.append(f"Utilisateur : {exchange.get('question', '')}")
                parts.append(f"Assistant : {exchange.get('answer', '')}")
            parts.append("")

        parts.append(f"Question actuelle : {question}")

        return "\n".join(parts)

    @staticmethod
    def build_summary_prompt(text: str) -> str:
        """
        Construit un prompt pour résumer un texte.

        Args:
            text: Le texte à résumer.

        Returns:
            Le prompt de résumé.
        """
        return (
            "Résume le texte suivant de manière concise et structurée. "
            "Identifie les points clés et organise-les en une liste.\n\n"
            f"TEXTE :\n{text}"
        )
