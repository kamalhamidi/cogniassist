"""
Module ingestion — Chargement, nettoyage et découpage de documents.

Ce module gère l'import de fichiers PDF, DOCX et TXT,
leur nettoyage et leur découpage en chunks pour l'indexation.
"""

from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.cleaner import TextCleaner

__all__ = ["DocumentLoader", "TextChunker", "TextCleaner"]
