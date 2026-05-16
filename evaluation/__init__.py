"""
Module evaluation — Évaluation des performances du pipeline RAG.

Fournit un évaluateur avec 4 métriques (faithfulness, answer_relevancy,
context_precision, context_recall), un reporter pour les rapports,
et une fonction utilitaire pour évaluation rapide.

Usage :
    from evaluation import run_quick_evaluation
    from evaluation.report import EvaluationReporter

    results = run_quick_evaluation(pipeline, n=3)
    reporter = EvaluationReporter(results)
    print(reporter.get_summary_metrics())
"""

from evaluation.ragas_eval import RAGEvaluator, run_quick_evaluation
from evaluation.report import EvaluationReporter

__all__ = ["RAGEvaluator", "EvaluationReporter", "run_quick_evaluation"]
