"""
tests/test_evaluation.py — Tests unitaires du module d'évaluation.

Tests du RAGEvaluator, de l'EvaluationReporter, et de run_quick_evaluation.
Utilise des mocks pour éviter de dépendre d'Ollama.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestRAGEvaluator:
    """Tests de l'évaluateur RAG."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from evaluation.ragas_eval import RAGEvaluator, run_quick_evaluation
        assert RAGEvaluator is not None
        assert run_quick_evaluation is not None

    def test_load_test_dataset(self) -> None:
        """Vérifie le chargement du dataset de test."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator()
        dataset = evaluator.load_test_dataset()

        assert isinstance(dataset, list)
        assert len(dataset) >= 3
        assert "question" in dataset[0]
        assert "ground_truth" in dataset[0]
        assert "context" in dataset[0]

    def test_load_missing_dataset(self, tmp_path) -> None:
        """Vérifie le comportement avec un dataset manquant."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator(str(tmp_path / "inexistant.json"))
        dataset = evaluator.load_test_dataset()
        assert dataset == []

    def test_evaluate_single(self) -> None:
        """Vérifie l'évaluation d'une seule paire Q/R."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator()

        scores = evaluator.evaluate_single(
            question="Qu'est-ce que le machine learning ?",
            answer="Le machine learning est une branche de l'intelligence artificielle.",
            context="Le machine learning est un domaine de l'intelligence artificielle qui utilise des algorithmes.",
            ground_truth="Le machine learning est une branche de l'IA.",
        )

        assert isinstance(scores, dict)
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores
        assert "context_recall" in scores

        # Tous les scores entre 0 et 1
        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{metric} hors limites : {score}"

    def test_evaluate_single_high_overlap(self) -> None:
        """Vérifie que des textes identiques donnent des scores élevés."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator()

        text = "CogniAssist est un assistant intelligent basé sur le RAG"
        scores = evaluator.evaluate_single(
            question=text, answer=text, context=text, ground_truth=text,
        )

        assert scores["faithfulness"] > 0.5
        assert scores["context_recall"] > 0.5

    def test_evaluate_single_no_overlap(self) -> None:
        """Vérifie que des textes sans rapport donnent des scores bas."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator()

        scores = evaluator.evaluate_single(
            question="Quelle est la météo ?",
            answer="Il fait beau aujourd'hui.",
            context="Python est un langage de programmation.",
            ground_truth="Il pleut.",
        )

        assert scores["faithfulness"] < 0.5

    def test_evaluate_with_mock_pipeline(self) -> None:
        """Vérifie l'évaluation complète avec un pipeline mocké."""
        from evaluation.ragas_eval import RAGEvaluator

        mock_pipeline = MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "CogniAssist est un assistant RAG intelligent.",
            "sources": [],
            "chunks_used": 3,
        }

        evaluator = RAGEvaluator()
        test_cases = [
            {
                "question": "Qu'est-ce que CogniAssist ?",
                "ground_truth": "CogniAssist est un assistant RAG.",
                "context": "CogniAssist est un système RAG pour les documents.",
            },
            {
                "question": "Quels formats sont supportés ?",
                "ground_truth": "PDF, DOCX, TXT.",
                "context": "Le module d'ingestion supporte PDF, DOCX et TXT.",
            },
        ]

        results = evaluator.evaluate(mock_pipeline, test_cases=test_cases)

        assert results["total_questions"] == 2
        assert len(results["results"]) == 2
        assert "avg_scores" in results
        assert "evaluation_time_s" in results
        assert results["evaluation_time_s"] >= 0

        # Vérifie les métriques moyennes
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            assert metric in results["avg_scores"]
            assert 0.0 <= results["avg_scores"][metric] <= 1.0

    def test_evaluate_empty_dataset(self) -> None:
        """Vérifie le comportement avec un dataset vide."""
        from evaluation.ragas_eval import RAGEvaluator
        evaluator = RAGEvaluator()

        results = evaluator.evaluate(MagicMock(), test_cases=[])

        assert results["total_questions"] == 0
        assert results["results"] == []
        assert results["avg_scores"] == {}

    def test_evaluate_with_n_limit(self) -> None:
        """Vérifie la limitation du nombre de questions."""
        from evaluation.ragas_eval import RAGEvaluator

        mock_pipeline = MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "Réponse test.",
            "sources": [],
        }

        evaluator = RAGEvaluator()
        results = evaluator.evaluate(mock_pipeline, n=2)

        assert results["total_questions"] <= 2

    def test_categorize_question(self) -> None:
        """Vérifie la catégorisation des questions."""
        from evaluation.ragas_eval import RAGEvaluator

        assert RAGEvaluator._categorize_question("Qu'est-ce que le RAG ?") == "définition"
        assert RAGEvaluator._categorize_question("Comment fonctionne le pipeline ?") == "processus"
        assert RAGEvaluator._categorize_question("Pourquoi utiliser ChromaDB ?") == "explication"
        assert RAGEvaluator._categorize_question("Quels formats supportés ?") == "factuel"
        assert RAGEvaluator._categorize_question("Bonjour") == "général"

    def test_run_quick_evaluation(self) -> None:
        """Vérifie la fonction utilitaire run_quick_evaluation."""
        from evaluation import run_quick_evaluation

        mock_pipeline = MagicMock()
        mock_pipeline.ask.return_value = {
            "answer": "Réponse test.",
            "sources": [],
        }

        results = run_quick_evaluation(mock_pipeline, n=2)

        assert "total_questions" in results
        assert "avg_scores" in results
        assert results["total_questions"] <= 2


class TestEvaluationReporter:
    """Tests du reporter d'évaluation."""

    @pytest.fixture
    def sample_results(self) -> dict:
        """Résultats d'évaluation de test."""
        return {
            "total_questions": 3,
            "evaluation_time_s": 1.5,
            "avg_scores": {
                "faithfulness": 0.75,
                "answer_relevancy": 0.60,
                "context_precision": 0.80,
                "context_recall": 0.65,
            },
            "results": [
                {
                    "question": "Qu'est-ce que CogniAssist ?",
                    "answer": "CogniAssist est un assistant RAG.",
                    "ground_truth": "CogniAssist est un assistant RAG intelligent.",
                    "context": "CogniAssist utilise le RAG.",
                    "scores": {
                        "faithfulness": 0.8,
                        "answer_relevancy": 0.7,
                        "context_precision": 0.9,
                        "context_recall": 0.6,
                    },
                    "category": "définition",
                },
                {
                    "question": "Comment fonctionne le pipeline ?",
                    "answer": "Le pipeline utilise un retriever et un LLM.",
                    "ground_truth": "Le pipeline RAG récupère et génère.",
                    "context": "Le pipeline comprend retriever et LLM.",
                    "scores": {
                        "faithfulness": 0.7,
                        "answer_relevancy": 0.5,
                        "context_precision": 0.7,
                        "context_recall": 0.7,
                    },
                    "category": "processus",
                },
                {
                    "question": "Quels formats supportés ?",
                    "answer": "PDF, DOCX, TXT.",
                    "ground_truth": "PDF, DOCX, TXT, MD.",
                    "context": "Supporte PDF, DOCX et TXT.",
                    "scores": {
                        "faithfulness": 0.75,
                        "answer_relevancy": 0.6,
                        "context_precision": 0.8,
                        "context_recall": 0.65,
                    },
                    "category": "factuel",
                },
            ],
        }

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from evaluation.report import EvaluationReporter
        assert EvaluationReporter is not None

    def test_get_summary_metrics(self, sample_results) -> None:
        """Vérifie le résumé des métriques."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results)
        summary = reporter.get_summary_metrics()

        assert "overall_score" in summary
        assert 0.0 <= summary["overall_score"] <= 1.0
        assert "best_metric" in summary
        assert "worst_metric" in summary
        assert "best_category" in summary
        assert "worst_category" in summary
        assert summary["total_questions"] == 3

    def test_get_scores_dataframe(self, sample_results) -> None:
        """Vérifie le DataFrame des scores."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results)
        df = reporter.get_scores_dataframe()

        assert len(df) == 3
        assert "Question" in df.columns
        assert "Score moyen" in df.columns

    def test_get_scores_dataframe_empty(self) -> None:
        """Vérifie le DataFrame vide."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter({"results": [], "avg_scores": {}})
        df = reporter.get_scores_dataframe()
        assert len(df) == 0

    def test_get_metrics_dataframe(self, sample_results) -> None:
        """Vérifie le DataFrame des métriques."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results)
        df = reporter.get_metrics_dataframe()

        assert len(df) == 4
        assert "Métrique" in df.columns
        assert "Statut" in df.columns

    def test_generate_markdown_report(self, sample_results) -> None:
        """Vérifie la génération du rapport Markdown."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results)
        report = reporter.generate_markdown_report()

        assert "Rapport d'évaluation" in report
        assert "Score global" in report
        assert "Fidélité" in report
        assert "CogniAssist" in report

    def test_export_report(self, sample_results, tmp_path) -> None:
        """Vérifie l'export en fichier."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results, output_dir=str(tmp_path))
        filepath = reporter.export_report("test_report.md")

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Rapport d'évaluation" in content

    def test_category_analysis(self, sample_results) -> None:
        """Vérifie l'analyse par catégorie."""
        from evaluation.report import EvaluationReporter
        reporter = EvaluationReporter(sample_results)
        summary = reporter.get_summary_metrics()

        assert "category_scores" in summary
        assert len(summary["category_scores"]) >= 1
        for cat, score in summary["category_scores"].items():
            assert 0.0 <= score <= 1.0
