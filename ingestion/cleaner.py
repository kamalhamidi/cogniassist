"""
ingestion/cleaner.py — Nettoyage et prétraitement du texte.

Fournit des utilitaires pour nettoyer le texte brut extrait des
documents avant le découpage en chunks : suppression d'espaces
superflus, de caractères spéciaux, normalisation Unicode, etc.
"""

import re
import unicodedata


class TextCleaner:
    """
    Nettoie et normalise le texte brut extrait de documents.

    Opérations disponibles :
    - Normalisation Unicode (NFC)
    - Suppression des espaces multiples
    - Suppression des sauts de ligne excessifs
    - Suppression des caractères de contrôle
    - Nettoyage des en-têtes/pieds de page récurrents
    """

    def clean(self, text: str) -> str:
        """
        Applique toutes les étapes de nettoyage au texte.

        Args:
            text: Texte brut à nettoyer.

        Returns:
            Texte nettoyé et normalisé.
        """
        if not text:
            return ""

        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_line_breaks(text)
        text = text.strip()

        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalise le texte en forme NFC (composition canonique)."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Supprime les caractères de contrôle (sauf newline et tab)."""
        return "".join(
            char for char in text
            if char in ("\n", "\t", "\r") or not unicodedata.category(char).startswith("C")
        )

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Remplace les espaces multiples par un seul espace."""
        return re.sub(r"[^\S\n]+", " ", text)

    @staticmethod
    def normalize_line_breaks(text: str) -> str:
        """Réduit les sauts de ligne excessifs (max 2 consécutifs)."""
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def remove_headers_footers(text: str, patterns: list[str] | None = None) -> str:
        """
        Supprime les en-têtes et pieds de page récurrents.

        Args:
            text: Texte à nettoyer.
            patterns: Liste de patterns regex à supprimer.

        Returns:
            Texte sans les motifs spécifiés.
        """
        if not patterns:
            return text

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        return text

    @staticmethod
    def remove_urls(text: str) -> str:
        """Supprime les URLs du texte."""
        return re.sub(r"https?://\S+", "", text)

    @staticmethod
    def remove_email_addresses(text: str) -> str:
        """Supprime les adresses email du texte."""
        return re.sub(r"\S+@\S+\.\S+", "", text)
