"""
Module evaluation — Évaluation des performances du pipeline RAG.

Ce module fournit des outils d'évaluation avec RAGAS,
des datasets de test et la génération de rapports.
"""

from evaluation.ragas_eval import RAGASEvaluator
from evaluation.report import ReportGenerator

__all__ = ["RAGASEvaluator", "ReportGenerator"]
