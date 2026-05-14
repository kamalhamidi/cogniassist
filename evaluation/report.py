"""
evaluation/report.py — Génération de rapports d'évaluation.

Produit des rapports visuels et textuels à partir des résultats
d'évaluation RAGAS, avec graphiques Plotly et export en fichier.
"""

from typing import Optional
from pathlib import Path

import pandas as pd


class ReportGenerator:
    """
    Génère des rapports d'évaluation du pipeline RAG.

    Produit des rapports sous forme de tableaux, graphiques
    et fichiers exportables.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initialise le générateur de rapports.

        Args:
            output_dir: Répertoire de sortie pour les rapports.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(self, evaluation_results: dict) -> pd.DataFrame:
        """
        Génère un tableau récapitulatif des métriques.

        Args:
            evaluation_results: Résultats de l'évaluation RAGAS.

        Returns:
            DataFrame avec les métriques et scores.
        """
        # TODO: Implémenter la génération de résumé
        pass

    def generate_chart(self, evaluation_results: dict) -> None:
        """
        Génère un graphique radar des métriques.

        Args:
            evaluation_results: Résultats de l'évaluation.
        """
        # TODO: Implémenter avec Plotly
        pass

    def export_report(self, evaluation_results: dict, filename: str = "rapport_evaluation.html") -> Path:
        """
        Exporte un rapport complet en HTML.

        Args:
            evaluation_results: Résultats de l'évaluation.
            filename: Nom du fichier de sortie.

        Returns:
            Chemin vers le fichier généré.
        """
        # TODO: Implémenter l'export HTML
        pass
