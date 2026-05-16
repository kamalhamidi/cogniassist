"""
ingestion/cleaner.py — Nettoyage et prétraitement du texte.

Applique des étapes de nettoyage successives sur les Documents
LangChain : normalisation Unicode NFKC, suppression des caractères
de contrôle, normalisation des espaces, et filtrage des documents trop courts.
"""

import logging
import re
import unicodedata

from langchain.schema import Document

logger = logging.getLogger("cogniassist.ingestion")

# Seuil minimum de caractères pour qu'un document soit conservé
MIN_DOCUMENT_LENGTH = 50


class TextCleaner:
    """
    Nettoie et normalise le texte des Documents LangChain.

    Pipeline de nettoyage appliqué dans l'ordre :
    1. Normalisation Unicode NFKC (ligatures, guillemets…)
    2. Suppression des caractères de contrôle non imprimables
    3. Suppression des espaces superflus
    4. Filtrage des documents trop courts (< 50 caractères)
    """

    def clean(self, documents: list[Document]) -> list[Document]:
        """
        Applique toutes les étapes de nettoyage à chaque Document.

        Le page_content de chaque Document est nettoyé en place.
        Les métadonnées ne sont jamais modifiées.

        Args:
            documents: Liste de Documents LangChain à nettoyer.

        Returns:
            Liste de Documents nettoyés (les documents trop courts sont exclus).
        """
        if not documents:
            return []

        logger.info("Nettoyage de %d document(s)…", len(documents))

        cleaned: list[Document] = []
        for doc in documents:
            text = doc.page_content

            # Appliquer les étapes de nettoyage
            text = self._normalize_unicode(text)
            text = self._remove_special_characters(text)
            text = self._remove_extra_whitespace(text)

            # Créer un nouveau Document avec le texte nettoyé
            # (les métadonnées sont copiées telles quelles)
            cleaned.append(
                Document(page_content=text, metadata=doc.metadata.copy())
            )

        # Filtrer les documents trop courts
        result = self._remove_short_documents(cleaned)

        logger.info(
            "Nettoyage terminé : %d → %d document(s) conservé(s)",
            len(documents), len(result),
        )
        return result

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """
        Supprime les espaces et sauts de ligne superflus.

        - Remplace les espaces multiples par un seul espace.
        - Réduit 3+ sauts de ligne consécutifs à 2 maximum.
        - Supprime les espaces en début/fin de texte.

        Args:
            text: Texte à nettoyer.

        Returns:
            Texte avec espaces normalisés.
        """
        # Remplacer les espaces multiples (hors newlines) par un seul espace
        text = re.sub(r"[^\S\n]+", " ", text)
        # Réduire les sauts de ligne excessifs (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Supprimer les espaces en début/fin
        return text.strip()

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """
        Supprime les caractères de contrôle non imprimables.

        Conserve \\n (newline) et \\t (tabulation) qui sont utiles
        pour la structure du texte. Conserve tous les caractères
        accentués (français, translittération arabe) et la ponctuation.

        Args:
            text: Texte à nettoyer.

        Returns:
            Texte sans caractères de contrôle.
        """
        return "".join(
            char for char in text
            if char in ("\n", "\t")
            or not unicodedata.category(char).startswith("C")
        )

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """
        Applique la normalisation Unicode NFKC.

        Corrige les artefacts d'encodage comme les ligatures
        (ﬁ → fi, ﬂ → fl) et les guillemets typographiques
        inhabituels.

        Args:
            text: Texte à normaliser.

        Returns:
            Texte normalisé en NFKC.
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_short_documents(documents: list[Document]) -> list[Document]:
        """
        Filtre les Documents dont le contenu est trop court.

        Un document avec moins de 50 caractères après nettoyage
        est considéré inutile pour le RAG et sera exclu.

        Args:
            documents: Liste de Documents nettoyés.

        Returns:
            Liste filtrée (sans les documents trop courts).
        """
        result: list[Document] = []

        for doc in documents:
            if len(doc.page_content) < MIN_DOCUMENT_LENGTH:
                source = doc.metadata.get("source", doc.metadata.get("file_name", "inconnu"))
                logger.warning(
                    "Document filtré (trop court : %d car.) — source : %s",
                    len(doc.page_content), source,
                )
                continue
            result.append(doc)

        return result
