"""
Module rag — Pipeline RAG avec Ollama.

Fournit le pipeline complet de Retrieval-Augmented Generation :
prompt building, mémoire de conversation, génération de réponses
et résumé de documents.

Usage principal :
    from rag import get_pipeline
    pipeline = get_pipeline()
    result = pipeline.ask("Ma question")
"""

import logging

from rag.prompt_builder import PromptBuilder
from rag.memory import ConversationMemory
from rag.pipeline import RAGPipeline
from rag.summarizer import BatchSummarizer

logger = logging.getLogger("cogniassist.rag")

__all__ = [
    "RAGPipeline", "PromptBuilder", "ConversationMemory",
    "BatchSummarizer", "get_pipeline",
]

# Singleton — une seule instance du pipeline partagée
_pipeline_instance: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """
    Retourne l'instance singleton du RAGPipeline.

    Crée le pipeline au premier appel, puis retourne la même instance
    pour les appels suivants. Évite de recharger le modèle LLM
    à chaque re-render Streamlit.

    Returns:
        Instance unique de RAGPipeline.
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        logger.info("Création de l'instance RAGPipeline (singleton)…")
        _pipeline_instance = RAGPipeline()

    return _pipeline_instance
