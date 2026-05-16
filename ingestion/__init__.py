"""
Module ingestion — Chargement, nettoyage et découpage de documents.

Usage principal :
    from ingestion import ingest_file
    chunks, stats = ingest_file(file_bytes, "document.pdf")
"""

import logging
from langchain.schema import Document
from ingestion.loader import DocumentLoader
from ingestion.cleaner import TextCleaner
from ingestion.chunker import DocumentChunker

logger = logging.getLogger("cogniassist.ingestion")
__all__ = ["DocumentLoader", "TextCleaner", "DocumentChunker", "ingest_file"]


def ingest_file(file_bytes: bytes, file_name: str) -> tuple[list[Document], dict]:
    """
    Pipeline d'ingestion complet en un seul appel.

    Args:
        file_bytes: Contenu binaire du fichier.
        file_name: Nom du fichier avec extension.

    Returns:
        Tuple (chunks, stats).
    """
    logger.info("Début ingestion : '%s'", file_name)

    loader = DocumentLoader()
    documents = loader.load_from_bytes(file_bytes, file_name)

    cleaner = TextCleaner()
    cleaned = cleaner.clean(documents)

    chunker = DocumentChunker()
    chunks = chunker.chunk(cleaned)
    stats = chunker.get_stats(chunks)

    logger.info("Ingestion terminée : '%s' → %d chunks", file_name, len(chunks))
    return chunks, stats
