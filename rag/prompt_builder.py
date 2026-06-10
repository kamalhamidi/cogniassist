"""
rag/prompt_builder.py — Construction des prompts pour le pipeline RAG.

Définit les templates de prompts utilisés pour la génération de réponses,
le résumé de documents et l'analyse des lacunes de connaissances.
Utilise LangChain PromptTemplate pour un formatage structuré.
"""

import logging

from langchain.prompts import PromptTemplate

logger = logging.getLogger("cogniassist.rag")


class PromptBuilder:
    """
    Constructeur de prompts pour le pipeline RAG CogniAssist.

    Templates disponibles :
    - rag_template : prompt principal de question-réponse
    - summary_template : résumé structuré d'un document
    - knowledge_gap_template : analyse des lacunes de connaissances
    """

    def __init__(self) -> None:
        """Initialise les templates de prompts."""

        self.rag_template = PromptTemplate(
            input_variables=["context", "question", "user_profile", "chat_history"],
            template="""Tu es CogniAssist, un assistant cognitif intelligent et personnalisé.
Tu aides l'utilisateur à exploiter ses propres documents et connaissances personnelles.

Profil utilisateur :
{user_profile}

Historique de la conversation :
{chat_history}

Contexte extrait de tes documents personnels :
{context}

Question de l'utilisateur :
{question}

Instructions :
- Réponds uniquement en te basant sur le contexte fourni
- Si le contexte ne contient pas la réponse, dis-le clairement
- Adapte le niveau de détail au profil utilisateur :
  • Si le niveau est "beginner" : explique simplement, évite le jargon technique, donne des exemples concrets
  • Si le niveau est "intermediate" : équilibre clarté et précision technique
  • Si le niveau est "expert" : sois technique, concis et direct, évite les explications basiques
- Respecte le style de réponse indiqué dans le profil :
  • "concise" : réponses courtes et directes
  • "detailed" : explications approfondies
  • "step_by_step" : décompose en étapes numérotées
  • "educational" : pédagogique avec exemples
- Si le type d'utilisateur est "enterprise" : utilise un registre professionnel, cite précisément les sources
- Réponds dans la langue préférée de l'utilisateur (indiquée dans le profil)
- Si tu cites un document, mentionne son nom

Réponse :""",
        )

        self.summary_template = PromptTemplate(
            input_variables=["document_content", "file_name"],
            template="""Tu es CogniAssist. Génère un résumé structuré du document suivant.

Nom du fichier : {file_name}

Contenu du document :
{document_content}

Génère un résumé avec exactement cette structure :
## Résumé de {file_name}

**Points clés :**
- (liste des 3 à 5 points les plus importants)

**Thèmes principaux :**
- (liste des thèmes abordés)

**Conclusion :**
(une phrase résumant l'essentiel)""",
        )

        self.knowledge_gap_template = PromptTemplate(
            input_variables=["documents_summary", "user_goals"],
            template="""Tu es CogniAssist. Analyse les documents de l'utilisateur
et identifie ses lacunes de connaissances.

Résumé des documents disponibles :
{documents_summary}

Objectifs de l'utilisateur :
{user_goals}

Identifie :
1. Ce que l'utilisateur maîtrise bien (basé sur ses documents)
2. Les lacunes importantes par rapport à ses objectifs
3. Les ressources ou sujets qu'il devrait explorer

Réponds de manière constructive et encourageante.""",
        )

        logger.debug("PromptBuilder initialisé avec 3 templates.")

    def build_rag_prompt(
        self,
        question: str,
        context: str,
        chat_history: str = "",
        user_profile: str = "Étudiant en Master Data Science",
    ) -> str:
        """
        Formate le prompt RAG principal avec toutes les variables.

        Args:
            question: La question de l'utilisateur.
            context: Le contexte documentaire retrouvé.
            chat_history: L'historique de conversation formaté.
            user_profile: Description du profil utilisateur.

        Returns:
            Le prompt complet formaté prêt pour le LLM.
        """
        return self.rag_template.format(
            question=question,
            context=context,
            chat_history=chat_history or "(aucun historique)",
            user_profile=user_profile,
        )

    def build_summary_prompt(
        self,
        document_content: str,
        file_name: str,
    ) -> str:
        """
        Formate le prompt de résumé de document.

        Tronque le contenu à 3000 caractères si nécessaire
        pour éviter un dépassement de la fenêtre de contexte.

        Args:
            document_content: Le contenu textuel du document.
            file_name: Le nom du fichier source.

        Returns:
            Le prompt de résumé formaté.
        """
        max_content_length = 3000
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "\n\n[... contenu tronqué ...]"
            logger.debug(
                "Contenu tronqué à %d caractères pour le résumé de '%s'.",
                max_content_length, file_name,
            )

        return self.summary_template.format(
            document_content=document_content,
            file_name=file_name,
        )

    def build_knowledge_gap_prompt(
        self,
        documents_summary: str,
        user_goals: str,
    ) -> str:
        """
        Formate le prompt d'analyse des lacunes de connaissances.

        Args:
            documents_summary: Résumé des documents disponibles.
            user_goals: Objectifs déclarés de l'utilisateur.

        Returns:
            Le prompt d'analyse formaté.
        """
        return self.knowledge_gap_template.format(
            documents_summary=documents_summary,
            user_goals=user_goals,
        )
