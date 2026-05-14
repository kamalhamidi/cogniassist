"""
evaluation/ragas_eval.py — Évaluation RAGAS du pipeline RAG.

Utilise le framework RAGAS pour évaluer la qualité des réponses
générées par le pipeline RAG selon plusieurs métriques :
- Faithfulness (fidélité au contexte)
- Answer relevancy (pertinence de la réponse)
- Context precision (précision du contexte)
- Context recall (rappel du contexte)
"""

from typing import Optional
from pathlib import Path
import json


class RAGASEvaluator:
    """
    Évaluateur RAGAS pour mesurer la qualité du pipeline RAG.

    Charge un dataset de test et exécute les métriques RAGAS
    pour produire un score de qualité global.
    """

    def __init__(self, test_dataset_path: Optional[str] = None) -> None:
        """
        Initialise l'évaluateur.

        Args:
            test_dataset_path: Chemin vers le fichier JSON du dataset de test.
        """
        if test_dataset_path is None:
            test_dataset_path = str(
                Path(__file__).parent / "test_dataset.json"
            )
        self.test_dataset_path = test_dataset_path

    def load_test_dataset(self) -> list[dict]:
        """
        Charge le dataset de test depuis le fichier JSON.

        Returns:
            Liste de cas de test (question, answer, context, ground_truth).
        """
        path = Path(self.test_dataset_path)
        if not path.exists():
            print(f"⚠️  Dataset de test introuvable : {path}")
            return []

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, pipeline=None) -> dict:
        """
        Exécute l'évaluation RAGAS sur le pipeline.

        Args:
            pipeline: Instance du RAGPipeline à évaluer.

        Returns:
            Dictionnaire des scores par métrique.
        """
        # TODO: Implémenter l'évaluation complète avec RAGAS
        # from ragas import evaluate
        # from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        pass

    def evaluate_single(self, question: str, answer: str, context: str, ground_truth: str) -> dict:
        """
        Évalue une seule paire question/réponse.

        Args:
            question: La question posée.
            answer: La réponse générée.
            context: Le contexte utilisé.
            ground_truth: La réponse attendue.

        Returns:
            Scores pour cette paire.
        """
        # TODO: Implémenter l'évaluation unitaire
        pass
