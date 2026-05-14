"""
Module rag — Pipeline RAG principal.

Ce module orchestre le pipeline Retrieval-Augmented Generation :
construction de prompts, gestion de la mémoire de conversation,
et résumé automatique.
"""

from rag.pipeline import RAGPipeline
from rag.prompt_builder import PromptBuilder
from rag.memory import ConversationMemory
from rag.summarizer import Summarizer

__all__ = ["RAGPipeline", "PromptBuilder", "ConversationMemory", "Summarizer"]
