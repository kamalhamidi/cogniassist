"""
rag/summarizer.py — Résumé automatique par lots et rapport de connaissances.

Utilise le RAGPipeline pour générer des résumés de plusieurs documents
et produire un rapport complet d'analyse des connaissances.
"""

import logging
from datetime import datetime

logger = logging.getLogger("cogniassist.rag")


class BatchSummarizer:
    """
    Résumeur par lots de documents.

    Génère des résumés pour plusieurs documents et produit
    des rapports structurés d'analyse des connaissances.
    """

    def __init__(self, pipeline) -> None:
        """
        Initialise le résumeur avec une référence au pipeline RAG.

        Args:
            pipeline: Instance de RAGPipeline.
        """
        self.pipeline = pipeline

    def summarize_all_documents(
        self,
        document_names: list[str],
    ) -> dict[str, str]:
        """
        Génère des résumés pour une liste de documents.

        Traite chaque document individuellement. Si un document échoue,
        l'erreur est capturée et stockée dans le résultat.

        Args:
            document_names: Liste des noms de fichiers à résumer.

        Returns:
            Dictionnaire {nom_fichier: résumé_ou_erreur}.
        """
        results: dict[str, str] = {}
        total = len(document_names)

        for i, file_name in enumerate(document_names, start=1):
            logger.info("Résumé %d/%d : %s", i, total, file_name)

            try:
                summary = self.pipeline.summarize_document(file_name)
                results[file_name] = summary
            except Exception as e:
                error_msg = f"Erreur lors du résumé de '{file_name}' : {str(e)}"
                logger.error(error_msg)
                results[file_name] = error_msg

        logger.info("Résumé terminé : %d/%d documents traités.", len(results), total)
        return results

    def generate_knowledge_report(
        self,
        user_goals: str,
        document_names: list[str],
    ) -> str:
        """
        Génère un rapport complet de connaissances.

        Étapes :
        1. Résumer tous les documents (500 premiers caractères chacun)
        2. Analyser les lacunes via le pipeline
        3. Formater le rapport final

        Args:
            user_goals: Objectifs déclarés de l'utilisateur.
            document_names: Liste des noms de fichiers indexés.

        Returns:
            Rapport formaté en Markdown.
        """
        # 1. Résumer les documents
        summaries = self.summarize_all_documents(document_names)

        # 2. Analyse des lacunes
        try:
            gap_analysis = self.pipeline.analyze_knowledge_gaps(user_goals)
        except Exception as e:
            gap_analysis = f"Erreur lors de l'analyse : {str(e)}"

        # 3. Formater le rapport
        doc_list = "\n".join(f"- {name}" for name in document_names)

        summary_sections: list[str] = []
        for name, summary in summaries.items():
            preview = summary[:500]
            if len(summary) > 500:
                preview += "..."
            summary_sections.append(f"### {name}\n{preview}")

        summaries_text = "\n\n".join(summary_sections)
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")

        report = f"""# Rapport de connaissances CogniAssist

## Documents analysés
{doc_list}

## Résumés
{summaries_text}

## Analyse des lacunes
{gap_analysis}

---
Généré par CogniAssist le {timestamp}"""

        return report
