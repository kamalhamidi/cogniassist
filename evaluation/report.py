"""
evaluation/report.py — Génération de rapports d'évaluation.

Produit des rapports structurés à partir des résultats d'évaluation :
métriques résumées, graphiques, export HTML et analyse par catégorie.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import Counter

import pandas as pd

logger = logging.getLogger("cogniassist.evaluation")


class EvaluationReporter:
    """
    Génère des rapports d'évaluation à partir des résultats.

    Fournit des résumés métriques, des DataFrames pour Streamlit,
    et des exports en Markdown/HTML.
    """

    def __init__(self, evaluation_results: dict, output_dir: str | None = None) -> None:
        """
        Initialise le reporter avec les résultats d'évaluation.

        Args:
            evaluation_results: Résultats retournés par RAGEvaluator.evaluate().
            output_dir: Répertoire de sortie pour les exports.
        """
        self.results = evaluation_results
        self.output_dir = Path(output_dir) if output_dir else Path("./data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_summary_metrics(self) -> dict:
        """
        Retourne un résumé des métriques principales.

        Returns:
            Dictionnaire avec overall_score, best/worst category,
            total questions et temps d'évaluation.
        """
        avg = self.results.get("avg_scores", {})
        results_list = self.results.get("results", [])

        # Score global = moyenne des 4 métriques
        overall = sum(avg.values()) / len(avg) if avg else 0.0

        # Meilleure et pire métrique
        best_metric = max(avg, key=avg.get) if avg else "—"
        worst_metric = min(avg, key=avg.get) if avg else "—"

        # Scores par catégorie
        category_scores: dict[str, list[float]] = {}
        for r in results_list:
            cat = r.get("category", "général")
            cat_avg = sum(r["scores"].values()) / len(r["scores"]) if r["scores"] else 0.0
            category_scores.setdefault(cat, []).append(cat_avg)

        cat_avg_scores = {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in category_scores.items()
        }
        best_category = max(cat_avg_scores, key=cat_avg_scores.get) if cat_avg_scores else "—"
        worst_category = min(cat_avg_scores, key=cat_avg_scores.get) if cat_avg_scores else "—"

        return {
            "overall_score": round(overall, 4),
            "avg_scores": avg,
            "best_metric": best_metric,
            "worst_metric": worst_metric,
            "best_category": best_category,
            "worst_category": worst_category,
            "category_scores": cat_avg_scores,
            "total_questions": self.results.get("total_questions", 0),
            "evaluation_time_s": self.results.get("evaluation_time_s", 0.0),
        }

    def get_scores_dataframe(self) -> pd.DataFrame:
        """
        Retourne les scores sous forme de DataFrame pour affichage.

        Returns:
            DataFrame avec une ligne par question et les 4 métriques.
        """
        results_list = self.results.get("results", [])
        if not results_list:
            return pd.DataFrame()

        rows = []
        for r in results_list:
            row = {
                "Question": r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"],
                "Catégorie": r.get("category", "général"),
            }
            row.update({
                k.replace("_", " ").title(): f"{v:.0%}"
                for k, v in r["scores"].items()
            })
            # Score moyen par question
            avg = sum(r["scores"].values()) / len(r["scores"]) if r["scores"] else 0.0
            row["Score moyen"] = f"{avg:.0%}"
            rows.append(row)

        return pd.DataFrame(rows)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Retourne les métriques agrégées sous forme de DataFrame.

        Returns:
            DataFrame avec une ligne par métrique.
        """
        avg = self.results.get("avg_scores", {})
        if not avg:
            return pd.DataFrame()

        labels = {
            "faithfulness": "Fidélité au contexte",
            "answer_relevancy": "Pertinence de la réponse",
            "context_precision": "Précision du contexte",
            "context_recall": "Rappel du contexte",
        }

        rows = [
            {
                "Métrique": labels.get(k, k),
                "Score": f"{v:.0%}",
                "Valeur": v,
                "Statut": "✅" if v >= 0.6 else ("⚠️" if v >= 0.4 else "❌"),
            }
            for k, v in avg.items()
        ]

        return pd.DataFrame(rows)

    def generate_markdown_report(self) -> str:
        """
        Génère un rapport complet en Markdown.

        Returns:
            Rapport formaté en Markdown.
        """
        summary = self.get_summary_metrics()
        avg = summary["avg_scores"]
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")

        lines = [
            "# 📊 Rapport d'évaluation CogniAssist",
            "",
            f"**Date :** {timestamp}",
            f"**Questions évaluées :** {summary['total_questions']}",
            f"**Temps d'évaluation :** {summary['evaluation_time_s']:.1f}s",
            "",
            "---",
            "",
            "## 🎯 Score global",
            "",
            f"**{summary['overall_score']:.0%}**",
            "",
            "## 📈 Métriques détaillées",
            "",
            "| Métrique | Score |",
            "|---|---|",
        ]

        labels = {
            "faithfulness": "Fidélité au contexte",
            "answer_relevancy": "Pertinence de la réponse",
            "context_precision": "Précision du contexte",
            "context_recall": "Rappel du contexte",
        }

        for k, v in avg.items():
            status = "✅" if v >= 0.6 else ("⚠️" if v >= 0.4 else "❌")
            lines.append(f"| {labels.get(k, k)} | {v:.0%} {status} |")

        lines.extend([
            "",
            "## 🏷️ Par catégorie",
            "",
            f"- **Meilleure :** {summary['best_category']}",
            f"- **À améliorer :** {summary['worst_category']}",
            "",
        ])

        # Détail par question
        results_list = self.results.get("results", [])
        if results_list:
            lines.extend([
                "## 📋 Détail par question",
                "",
            ])
            for i, r in enumerate(results_list, 1):
                q_avg = sum(r["scores"].values()) / len(r["scores"]) if r["scores"] else 0.0
                lines.append(f"### {i}. {r['question']}")
                lines.append(f"- **Catégorie :** {r.get('category', '—')}")
                lines.append(f"- **Score moyen :** {q_avg:.0%}")
                lines.append(f"- **Réponse :** {r['answer'][:200]}...")
                lines.append("")

        lines.extend([
            "---",
            f"*Généré par CogniAssist le {timestamp}*",
        ])

        return "\n".join(lines)

    def export_report(self, filename: str = "rapport_evaluation.md") -> Path:
        """
        Exporte le rapport en fichier Markdown.

        Args:
            filename: Nom du fichier de sortie.

        Returns:
            Chemin vers le fichier généré.
        """
        report = self.generate_markdown_report()
        filepath = self.output_dir / filename
        filepath.write_text(report, encoding="utf-8")
        logger.info("Rapport exporté : %s", filepath)
        return filepath
