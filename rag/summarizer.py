"""
rag/summarizer.py — Résumé automatique de documents.

Utilise le LLM pour générer des résumés concis de documents
ou de longues réponses. Supporte le résumé par chunks pour
les textes dépassant la fenêtre de contexte.
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from rag.prompt_builder import PromptBuilder


class Summarizer:
    """
    Génère des résumés automatiques de textes longs.

    Utilise un LLM pour produire des résumés concis et structurés.
    Supporte le résumé par morceaux (map-reduce) pour les textes
    dépassant la limite de contexte.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_chunk_size: int = 3000,
    ) -> None:
        """
        Initialise le résumeur.

        Args:
            model_name: Nom du modèle OpenAI.
            temperature: Température de génération (basse pour plus de fidélité).
            max_chunk_size: Taille maximale d'un chunk pour le résumé.
        """
        from config import settings

        model = model_name or settings.OPENAI_MODEL
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.max_chunk_size = max_chunk_size
        self.prompt_builder = PromptBuilder()

    def summarize(self, text: str) -> str:
        """
        Génère un résumé d'un texte.

        Args:
            text: Le texte à résumer.

        Returns:
            Le résumé généré.
        """
        if not text or not text.strip():
            return "Aucun texte à résumer."

        # Si le texte est court, résumé direct
        if len(text) <= self.max_chunk_size:
            return self._summarize_single(text)

        # Sinon, résumé par chunks (map-reduce)
        return self._summarize_long(text)

    def _summarize_single(self, text: str) -> str:
        """Résume un texte court en un seul appel LLM."""
        prompt = self.prompt_builder.build_summary_prompt(text)
        messages = [
            SystemMessage(content="Tu es un assistant expert en synthèse de documents."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _summarize_long(self, text: str) -> str:
        """Résume un texte long par chunks (stratégie map-reduce)."""
        # Découper en chunks
        chunks = [
            text[i:i + self.max_chunk_size]
            for i in range(0, len(text), self.max_chunk_size)
        ]

        # Phase Map : résumer chaque chunk
        chunk_summaries: list[str] = []
        for chunk in chunks:
            summary = self._summarize_single(chunk)
            chunk_summaries.append(summary)

        # Phase Reduce : résumer les résumés
        combined = "\n\n".join(chunk_summaries)
        if len(combined) <= self.max_chunk_size:
            return self._summarize_single(combined)

        # Récursion si encore trop long
        return self._summarize_long(combined)
