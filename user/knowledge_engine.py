"""
user/knowledge_engine.py — Moteur de profil de connaissances ACPE.

Estime la familiarité de l'utilisateur avec différents domaines
en analysant les interactions, les feedbacks et les documents uploadés.
Les scores évoluent au fil du temps via une moyenne mobile exponentielle.
"""

import logging
import math
from collections import Counter
from datetime import datetime
from typing import Optional

from user.db import uses_db_session
from user.acpe_models import KnowledgeProfile

logger = logging.getLogger("cogniassist.user")

# Domaines reconnus automatiquement (mots-clés → domaine)
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Python": ["python", "django", "flask", "fastapi", "pip", "virtualenv", "pytest"],
    "Machine Learning": [
        "machine", "learning", "ml", "classification", "regression",
        "clustering", "supervisé", "supervised", "unsupervised",
        "algorithme", "modèle", "entraînement", "training",
    ],
    "Deep Learning": [
        "deep", "neurone", "neural", "cnn", "rnn", "lstm", "transformer",
        "convolutif", "couche", "layer", "backpropagation", "epoch",
    ],
    "NLP": [
        "nlp", "langage", "naturel", "tokenisation", "embedding",
        "bert", "gpt", "transformers", "texte", "corpus", "sentiment",
    ],
    "RAG": [
        "rag", "retrieval", "augmented", "generation", "vectoriel",
        "chromadb", "embedding", "chunk", "retriever", "langchain",
    ],
    "Data Science": [
        "data", "science", "analyse", "statistique", "visualisation",
        "pandas", "numpy", "matplotlib", "plotly", "dataset",
    ],
    "SQL & Bases de données": [
        "sql", "base", "données", "sqlite", "postgresql", "mysql",
        "requête", "query", "table", "jointure", "index",
    ],
    "Cybersécurité": [
        "sécurité", "security", "cybersécurité", "chiffrement",
        "encryption", "firewall", "vulnérabilité", "authentification",
    ],
    "DevOps & Cloud": [
        "docker", "kubernetes", "cloud", "aws", "azure", "gcp",
        "ci/cd", "pipeline", "déploiement", "deployment", "container",
    ],
    "Web Development": [
        "web", "html", "css", "javascript", "react", "api", "rest",
        "frontend", "backend", "serveur", "http", "endpoint",
    ],
}

# Mots vides à ignorer lors de l'extraction de domaines
STOPWORDS = frozenset([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "à", "au", "aux", "ce", "qui", "que", "par", "sur", "dans",
    "est", "sont", "avec", "pour", "pas", "ne", "se", "sa", "son",
    "quels", "quel", "quelle", "comment", "pourquoi", "quoi",
    "the", "a", "an", "of", "in", "is", "are", "and", "to", "for",
])


class KnowledgeProfileEngine:
    """Moteur de profil de connaissances basé sur les interactions.

    Estime et met à jour la maîtrise de l'utilisateur par domaine
    en utilisant une moyenne mobile exponentielle (EMA).
    """

    # Facteur de lissage EMA : plus haut = plus réactif aux changements
    ALPHA = 0.3
    # Seuil minimum d'interactions pour qu'un domaine soit considéré fiable
    MIN_INTERACTIONS = 3

    def __init__(self, user_id: str) -> None:
        """Initialise le moteur pour un utilisateur donné."""
        self.user_id = user_id

    def update_from_interaction(
        self,
        question: str,
        answer: str,
        sources: list[str],
        feedback: Optional[int] = None,
    ) -> list[str]:
        """Met à jour le profil de connaissances après une interaction.

        Args:
            question: Question posée par l'utilisateur.
            answer: Réponse générée.
            sources: Noms de fichiers sources utilisés.
            feedback: 1 (positif), -1 (négatif), ou None.

        Returns:
            Liste des domaines détectés dans cette interaction.
        """
        # Extraire les domaines de la question et de la réponse
        text = f"{question} {answer}"
        domains = self._extract_domains(text)

        if not domains:
            return []

        for domain in domains:
            # Signal positif par défaut (l'utilisateur explore ce domaine)
            signal = 60.0  # Légèrement au-dessus de la base
            weight = 1.0

            if feedback == 1:
                signal = 80.0
                weight = 1.5
            elif feedback == -1:
                signal = 40.0
                weight = 0.5

            self._update_mastery(domain, signal, weight)
            self._update_confidence(domain)

        return domains

    def update_from_document(self, file_name: str) -> list[str]:
        """Met à jour le profil après l'upload d'un document.

        Args:
            file_name: Nom du fichier uploadé.

        Returns:
            Liste des domaines détectés dans le nom du fichier.
        """
        domains = self._extract_domains(file_name)
        for domain in domains:
            self._update_mastery(domain, signal=70.0, weight=2.0)
            self._update_confidence(domain)
        return domains

    @uses_db_session
    def get_knowledge_profile(self, session) -> list[dict]:
        """Retourne tous les profils de connaissances triés par score.

        Returns:
            Liste de dicts avec domain, mastery_score, confidence, etc.
        """
        profiles = (
            session.query(KnowledgeProfile)
            .filter_by(user_id=self.user_id)
            .order_by(KnowledgeProfile.mastery_score.desc())
            .all()
        )
        return [kp.to_dict() for kp in profiles]

    @uses_db_session
    def get_top_domains(self, session, limit: int = 5) -> list[str]:
        """Retourne les domaines les mieux maîtrisés.

        Args:
            limit: Nombre de domaines à retourner.

        Returns:
            Liste de noms de domaines.
        """
        profiles = (
            session.query(KnowledgeProfile)
            .filter_by(user_id=self.user_id)
            .filter(KnowledgeProfile.interaction_count >= self.MIN_INTERACTIONS)
            .order_by(KnowledgeProfile.mastery_score.desc())
            .limit(limit)
            .all()
        )
        return [kp.domain for kp in profiles]

    @uses_db_session
    def get_weak_domains(self, session, limit: int = 5, threshold: int = 40) -> list[str]:
        """Retourne les domaines à renforcer (score sous le seuil).

        Args:
            limit: Nombre de domaines à retourner.
            threshold: Score en dessous duquel un domaine est considéré faible.

        Returns:
            Liste de noms de domaines faibles.
        """
        profiles = (
            session.query(KnowledgeProfile)
            .filter_by(user_id=self.user_id)
            .filter(
                KnowledgeProfile.mastery_score < threshold,
                KnowledgeProfile.interaction_count >= self.MIN_INTERACTIONS,
            )
            .order_by(KnowledgeProfile.mastery_score.asc())
            .limit(limit)
            .all()
        )
        return [kp.domain for kp in profiles]

    def _extract_domains(self, text: str) -> list[str]:
        """Extrait les domaines d'un texte via correspondance de mots-clés.

        Args:
            text: Texte à analyser (question, réponse ou nom de fichier).

        Returns:
            Liste de noms de domaines détectés (dédupliquée).
        """
        words = set(text.lower().split())
        # Nettoyer la ponctuation
        cleaned = set()
        for w in words:
            clean = w.strip("?!.,;:'\"()[]{}«»/\\-_")
            if clean and clean not in STOPWORDS and len(clean) > 1:
                cleaned.add(clean)

        detected: list[str] = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = cleaned & set(keywords)
            if len(matches) >= 1:
                detected.append(domain)

        return detected

    @uses_db_session
    def _update_mastery(
        self, session, domain: str, signal: float, weight: float = 1.0,
    ) -> None:
        """Met à jour le score de maîtrise via EMA pondérée.

        new_score = α * weight * signal + (1 - α * weight) * old_score

        Args:
            domain: Nom du domaine.
            signal: Valeur du signal (0-100).
            weight: Poids du signal.
        """
        try:
            kp = (
                session.query(KnowledgeProfile)
                .filter_by(user_id=self.user_id, domain=domain)
                .first()
            )

            if kp is None:
                kp = KnowledgeProfile(
                    user_id=self.user_id,
                    domain=domain,
                    mastery_score=50,
                    confidence=0,
                    interaction_count=0,
                )
                session.add(kp)

            # EMA
            alpha = min(self.ALPHA * weight, 0.8)  # Plafonner à 0.8
            new_score = alpha * signal + (1 - alpha) * kp.mastery_score
            kp.mastery_score = max(0, min(100, int(round(new_score))))
            kp.interaction_count += 1
            kp.updated_at = datetime.utcnow()

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Erreur update mastery (%s) : %s", domain, e)

    @uses_db_session
    def _update_confidence(self, session, domain: str) -> None:
        """Met à jour le niveau de confiance (logarithmique).

        confidence = min(100, 20 * log2(interaction_count + 1))

        Args:
            domain: Nom du domaine.
        """
        try:
            kp = (
                session.query(KnowledgeProfile)
                .filter_by(user_id=self.user_id, domain=domain)
                .first()
            )
            if kp and kp.interaction_count > 0:
                kp.confidence = min(
                    100,
                    int(round(20 * math.log2(kp.interaction_count + 1))),
                )
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Erreur update confidence (%s) : %s", domain, e)
