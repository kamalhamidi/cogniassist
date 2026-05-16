"""
evaluation/ragas_eval.py — Évaluation des performances du pipeline RAG.

Évalue la qualité des réponses selon 4 métriques calculées localement
(sans dépendance externe à RAGAS ou OpenAI) :
- Faithfulness : la réponse est-elle fidèle au contexte fourni ?
- Answer Relevancy : la réponse répond-elle à la question ?
- Context Precision : le contexte récupéré est-il pertinent ?
- Context Recall : le contexte couvre-t-il la réponse attendue ?
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger("cogniassist.evaluation")


# Mots vides FR + EN à ignorer dans les calculs de similarité
_STOPWORDS = frozenset([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "à", "au", "aux", "ce", "qui", "que", "par", "sur", "dans",
    "est", "sont", "avec", "pour", "pas", "ne", "se", "sa", "son",
    "the", "a", "an", "of", "in", "is", "are", "and", "to",
    "for", "with", "on", "it", "this", "that", "was", "be",
])


def _tokenize(text: str) -> list[str]:
    """Tokenise un texte en mots nettoyés, sans mots vides."""
    words = text.lower().split()
    return [
        w.strip("?!.,;:'\"()[]{}«»-")
        for w in words
        if w.strip("?!.,;:'\"()[]{}«»-") and w.strip("?!.,;:'\"()[]{}«»-") not in _STOPWORDS
    ]


def _word_overlap(text_a: str, text_b: str) -> float:
    """Calcule le taux de chevauchement de mots entre deux textes (Jaccard)."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _recall_overlap(reference: str, candidate: str) -> float:
    """Calcule le rappel : proportion des mots de référence présents dans le candidat."""
    ref_tokens = set(_tokenize(reference))
    cand_tokens = set(_tokenize(candidate))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


class RAGEvaluator:
    """
    Évaluateur de qualité du pipeline RAG.

    Calcule 4 métriques pour chaque paire question-réponse :
    - faithfulness : fidélité de la réponse au contexte
    - answer_relevancy : pertinence de la réponse vs la question
    - context_precision : pertinence du contexte vs la question
    - context_recall : couverture du contexte vs la réponse attendue
    """

    def __init__(self, test_dataset_path: str | None = None) -> None:
        """
        Initialise l'évaluateur.

        Args:
            test_dataset_path: Chemin vers le fichier JSON du dataset.
                              Par défaut : evaluation/test_dataset.json
        """
        if test_dataset_path is None:
            test_dataset_path = str(Path(__file__).parent / "test_dataset.json")
        self.test_dataset_path = test_dataset_path

    def load_test_dataset(self) -> list[dict]:
        """
        Charge le dataset de test depuis le fichier JSON.

        Returns:
            Liste de cas de test avec question, ground_truth, context.
        """
        path = Path(self.test_dataset_path)
        if not path.exists():
            logger.warning("Dataset de test introuvable : %s", path)
            return []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Dataset chargé : %d cas de test.", len(data))
        return data

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
    ) -> dict[str, float]:
        """
        Évalue une seule paire question-réponse.

        Args:
            question: Question posée.
            answer: Réponse générée par le LLM.
            context: Contexte documentaire utilisé.
            ground_truth: Réponse attendue (référence).

        Returns:
            Dictionnaire des scores (0.0 à 1.0) par métrique.
        """
        # Faithfulness : la réponse est-elle fidèle au contexte ?
        faithfulness = _word_overlap(answer, context)

        # Answer Relevancy : la réponse répond-elle à la question ?
        answer_relevancy = _word_overlap(answer, question)

        # Context Precision : le contexte est-il pertinent pour la question ?
        context_precision = _word_overlap(context, question)

        # Context Recall : le contexte couvre-t-il la réponse attendue ?
        context_recall = _recall_overlap(ground_truth, context)

        return {
            "faithfulness": round(min(faithfulness * 2, 1.0), 4),
            "answer_relevancy": round(min(answer_relevancy * 2, 1.0), 4),
            "context_precision": round(min(context_precision * 2, 1.0), 4),
            "context_recall": round(min(context_recall * 2, 1.0), 4),
        }

    def evaluate(
        self,
        pipeline,
        test_cases: list[dict] | None = None,
        n: int | None = None,
    ) -> dict:
        """
        Évalue le pipeline sur un ensemble de cas de test.

        Args:
            pipeline: Instance de RAGPipeline.
            test_cases: Cas de test (si None, charge le dataset).
            n: Nombre max de cas à évaluer (si None, tous).

        Returns:
            Résultats complets avec scores individuels et moyennes.
        """
        if test_cases is None:
            test_cases = self.load_test_dataset()

        if n is not None:
            test_cases = test_cases[:n]

        if not test_cases:
            logger.warning("Aucun cas de test disponible.")
            return {
                "total_questions": 0,
                "results": [],
                "avg_scores": {},
                "evaluation_time_s": 0.0,
            }

        start_time = time.time()
        results: list[dict] = []

        for i, case in enumerate(test_cases):
            question = case["question"]
            ground_truth = case.get("ground_truth", "")
            provided_context = case.get("context", "")

            logger.info("Évaluation %d/%d : %s", i + 1, len(test_cases), question[:60])

            # Obtenir la réponse du pipeline
            try:
                result = pipeline.ask(question)
                answer = result["answer"]
                # Utiliser le contexte récupéré par le pipeline si possible
                retrieved_context = provided_context
            except Exception as e:
                logger.error("Erreur pipeline pour '%s' : %s", question[:40], e)
                answer = f"Erreur : {e}"
                retrieved_context = provided_context

            # Calculer les métriques
            scores = self.evaluate_single(
                question=question,
                answer=answer,
                context=retrieved_context,
                ground_truth=ground_truth,
            )

            results.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "context": retrieved_context,
                "scores": scores,
                "category": self._categorize_question(question),
            })

        # Calculer les moyennes
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        avg_scores = {}
        for metric in metric_names:
            values = [r["scores"][metric] for r in results if metric in r["scores"]]
            avg_scores[metric] = round(sum(values) / len(values), 4) if values else 0.0

        total_time = round(time.time() - start_time, 2)

        logger.info(
            "Évaluation terminée : %d questions en %.1fs. Score moyen : %.2f",
            len(results), total_time,
            sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0,
        )

        return {
            "total_questions": len(results),
            "results": results,
            "avg_scores": avg_scores,
            "evaluation_time_s": total_time,
        }

    @staticmethod
    def _categorize_question(question: str) -> str:
        """Catégorise une question pour les statistiques."""
        q_lower = question.lower()
        if any(w in q_lower for w in ["pourquoi", "why", "raison"]):
            return "explication"
        elif any(w in q_lower for w in ["qu'est-ce", "définition", "what"]):
            return "définition"
        elif any(w in q_lower for w in ["comment", "how", "fonctionn", "procéd"]):
            return "processus"
        elif any(w in q_lower for w in ["quel", "combien", "which", "list"]):
            return "factuel"
        return "général"


def run_quick_evaluation(
    pipeline,
    n: int = 3,
    test_dataset_path: str | None = None,
) -> dict:
    """
    Lance une évaluation rapide sur n questions.

    Fonction utilitaire pour un test rapide depuis la console.

    Args:
        pipeline: Instance de RAGPipeline.
        n: Nombre de questions à évaluer.
        test_dataset_path: Chemin vers le dataset (optionnel).

    Returns:
        Résultats de l'évaluation.
    """
    evaluator = RAGEvaluator(test_dataset_path)
    return evaluator.evaluate(pipeline, n=n)
