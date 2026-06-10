"""
user/recommender.py — Recommandations personnalisées.

Analyse le profil et l'historique de l'utilisateur pour suggérer
des questions, recommander des documents, et adapter les paramètres RAG.
"""

import logging
from datetime import datetime, timedelta

from user.profile import UserProfileManager
from user.history import InteractionHistory

logger = logging.getLogger("cogniassist.user")

# Questions génériques de secours
DEFAULT_QUESTIONS = [
    "Quels sont les points clés de mes documents ?",
    "Peux-tu résumer mon document le plus récent ?",
    "Quels sujets devrais-je approfondir selon mes objectifs ?",
    "Quelles sont les notions essentielles dans mes documents ?",
    "Fais-moi un récapitulatif de ce que j'ai appris.",
]


class PersonalizedRecommender:
    """
    Recommandations personnalisées basées sur le profil et l'historique.

    Suggère des questions, recommande des documents à relire,
    et adapte les paramètres du pipeline RAG au niveau de l'utilisateur.
    """

    def __init__(self, user_id: str = "default") -> None:
        """
        Initialise le recommandeur.

        Args:
            user_id: Identifiant de l'utilisateur.
        """
        self.user_id = user_id
        self.profile_manager = UserProfileManager(user_id)
        self.history = InteractionHistory(user_id)

    def get_suggested_questions(self) -> list[str]:
        """
        Suggère 5 questions pertinentes pour l'utilisateur.

        Combine le profil, les sujets fréquents et les documents
        uploadés pour générer des suggestions contextuelles.

        Returns:
            Liste de 5 questions suggérées.
        """
        profile = self.profile_manager.get_profile()
        topics = self.history.get_frequent_topics(limit=3)
        documents = self.profile_manager.get_user_documents()

        suggestions: list[str] = []

        # Questions basées sur les domaines d'intérêt
        for domain in profile.get("domain_focus", [])[:2]:
            suggestions.append(
                f"Que disent mes documents sur {domain} ?"
            )

        # Questions basées sur les documents uploadés
        if documents:
            recent_doc = documents[0]["file_name"]
            suggestions.append(
                f"Peux-tu résumer le document '{recent_doc}' ?"
            )

        # Questions basées sur les sujets fréquents
        for topic in topics[:2]:
            suggestions.append(
                f"Explique-moi davantage le concept de {topic}."
            )

        # Compléter avec les questions génériques
        for q in DEFAULT_QUESTIONS:
            if len(suggestions) >= 5:
                break
            if q not in suggestions:
                suggestions.append(q)

        return suggestions[:5]

    def get_document_recommendations(self) -> list[dict]:
        """
        Recommande des documents à relire.

        Identifie les documents jamais consultés ou non consultés
        depuis plus de 7 jours.

        Returns:
            Liste de recommandations avec raison.
        """
        documents = self.profile_manager.get_user_documents()
        recommendations: list[dict] = []
        cutoff = datetime.utcnow() - timedelta(days=7)

        for doc in documents:
            if doc["access_count"] == 0:
                recommendations.append({
                    "file_name": doc["file_name"],
                    "reason": "Jamais consulté",
                    "upload_date": doc["upload_date"],
                })
            elif doc["last_accessed"]:
                try:
                    last = datetime.fromisoformat(doc["last_accessed"])
                    if last < cutoff:
                        recommendations.append({
                            "file_name": doc["file_name"],
                            "reason": "Non consulté depuis 7 jours",
                            "upload_date": doc["upload_date"],
                        })
                except (ValueError, TypeError):
                    pass

        return recommendations

    def adapt_rag_parameters(self) -> dict:
        """
        Adapte les paramètres RAG au profil complet de l'utilisateur.

        Prend en compte le niveau d'expertise, le type d'utilisateur
        (individual/enterprise) et le style de réponse préféré.

        Returns:
            Dictionnaire avec k, user_profile, temperature, response_style
            et user_type.
        """
        profile = self.profile_manager.get_profile()
        level = profile.get("expertise_level", "intermediate")
        user_type = profile.get("user_type", "individual")
        response_style = profile.get("response_style", "detailed")

        # Paramètres de base par niveau d'expertise
        base_params = {
            "beginner": {"k": 7, "temperature": 0.2},
            "intermediate": {"k": 5, "temperature": 0.3},
            "expert": {"k": 4, "temperature": 0.1},
        }

        level_params = base_params.get(level, base_params["intermediate"])

        # Ajustements selon le type d'utilisateur
        if user_type == "enterprise":
            level_params["k"] = 6
            level_params["temperature"] = 0.15

        # Ajustements selon le style de réponse
        if response_style == "concise":
            level_params["temperature"] = max(0.1, level_params["temperature"] - 0.05)
        elif response_style in ("detailed", "educational"):
            level_params["k"] = min(8, level_params["k"] + 1)

        return {
            "k": level_params["k"],
            "user_profile": self.profile_manager.get_personalization_context(),
            "temperature": level_params["temperature"],
            "response_style": response_style,
            "user_type": user_type,
        }

    def get_learning_progress(self) -> dict:
        """
        Calcule un rapport de progression d'apprentissage.

        Returns:
            Dictionnaire avec métriques de progression et score 0-100.
        """
        documents = self.profile_manager.get_user_documents()
        stats = self.history.get_interaction_stats()
        topics = self.history.get_frequent_topics(limit=10)

        docs_count = len(documents)
        docs_with_summary = sum(1 for d in documents if d["has_summary"])
        total_questions = stats["total_interactions"]
        positive = stats["positive_feedback"]
        negative = stats["negative_feedback"]
        total_feedback = positive + negative

        # Taux de feedback positif
        feedback_rate = round((positive / total_feedback) * 100, 1) if total_feedback > 0 else 0.0

        # Score de connaissance : docs * 10 + questions * 2, plafonné à 100
        raw_score = docs_count * 10 + total_questions * 2
        knowledge_score = min(raw_score, 100)

        return {
            "documents_uploaded": docs_count,
            "documents_with_summary": docs_with_summary,
            "total_questions_asked": total_questions,
            "topics_explored": topics,
            "positive_feedback_rate": feedback_rate,
            "knowledge_score": knowledge_score,
        }
