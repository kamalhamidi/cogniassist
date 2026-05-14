"""
ingestion/chunker.py — Découpage de texte en chunks.

Divise les documents en morceaux de taille contrôlée avec
chevauchement (overlap) pour maintenir le contexte entre les chunks.
Utilise LangChain RecursiveCharacterTextSplitter.
"""

from typing import Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Représente un chunk de texte avec ses métadonnées."""

    content: str
    index: int
    metadata: dict


class TextChunker:
    """
    Découpe du texte en chunks avec chevauchement.

    Utilise RecursiveCharacterTextSplitter de LangChain pour un
    découpage intelligent respectant les limites de phrases.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> None:
        """
        Initialise le chunker.

        Args:
            chunk_size: Taille maximale de chaque chunk (en caractères).
            chunk_overlap: Chevauchement entre chunks consécutifs.
            separators: Séparateurs à utiliser pour le découpage.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Découpe un texte en chunks.

        Args:
            text: Le texte à découper.
            metadata: Métadonnées à associer à chaque chunk.

        Returns:
            Liste de chunks avec indices et métadonnées.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}
        raw_chunks = self._splitter.split_text(text)

        return [
            Chunk(
                content=chunk,
                index=i,
                metadata={**base_metadata, "chunk_index": i, "total_chunks": len(raw_chunks)},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def split_documents(
        self, documents: list, source_key: str = "filename"
    ) -> list[Chunk]:
        """
        Découpe une liste de documents en chunks.

        Args:
            documents: Liste de Documents (du module loader).
            source_key: Clé de métadonnée pour identifier la source.

        Returns:
            Liste de tous les chunks issus des documents.
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            metadata = {source_key: getattr(doc, "filename", "inconnu"), **getattr(doc, "metadata", {})}
            chunks = self.split(doc.content, metadata=metadata)
            all_chunks.extend(chunks)

        return all_chunks
