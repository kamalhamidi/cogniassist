"""
ingestion/chunker.py — Découpage de documents en chunks.

Utilise LangChain RecursiveCharacterTextSplitter pour découper
les Documents en morceaux de taille contrôlée avec chevauchement.
Propose deux stratégies : par caractères et par tokens (tiktoken).
"""

import logging
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger("cogniassist.ingestion")


class DocumentChunker:
    """
    Découpe des Documents LangChain en chunks avec métadonnées enrichies.

    Deux splitters disponibles :
    - text_splitter : découpage par nombre de caractères (usage général)
    - token_splitter : découpage par nombre de tokens tiktoken (documents techniques)
    """

    def __init__(self) -> None:
        """
        Initialise les deux splitters avec les paramètres des settings.

        text_splitter utilise les séparateurs naturels du français.
        token_splitter utilise le tokenizer gpt2 (gratuit et local).
        """
        # Splitter par caractères — usage général
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        # Splitter par tokens — documents techniques
        self.token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt2",
            chunk_size=400,
            chunk_overlap=40,
        )

        logger.debug(
            "DocumentChunker initialisé (chunk_size=%d, chunk_overlap=%d)",
            settings.CHUNK_SIZE, settings.CHUNK_OVERLAP,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Découpe les documents en chunks avec le text_splitter (par caractères).

        Chaque chunk est enrichi avec les métadonnées suivantes :
        - chunk_id : identifiant unique "{file_name}_{index}"
        - chunk_index : position du chunk dans le document source
        - total_chunks : nombre total de chunks issus de ce document
        - word_count : nombre de mots dans le chunk

        Args:
            documents: Liste de Documents LangChain à découper.

        Returns:
            Liste de tous les chunks avec métadonnées enrichies.
        """
        return self._split_with(self.text_splitter, documents)

    def chunk_by_tokens(self, documents: list[Document]) -> list[Document]:
        """
        Découpe les documents en chunks avec le token_splitter (par tokens).

        Utilisez cette méthode pour les documents techniques où le
        nombre de tokens est plus pertinent que le nombre de caractères.
        Même enrichissement de métadonnées que chunk().

        Args:
            documents: Liste de Documents LangChain à découper.

        Returns:
            Liste de tous les chunks avec métadonnées enrichies.
        """
        return self._split_with(self.token_splitter, documents)

    def _split_with(
        self,
        splitter: RecursiveCharacterTextSplitter,
        documents: list[Document],
    ) -> list[Document]:
        """
        Méthode interne de découpage partagée par les deux stratégies.

        Args:
            splitter: Instance de RecursiveCharacterTextSplitter à utiliser.
            documents: Liste de Documents à découper.

        Returns:
            Liste de chunks avec métadonnées enrichies.
        """
        if not documents:
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            source = doc.metadata.get(
                "file_name",
                doc.metadata.get("source", "inconnu"),
            )

            # Découper le document
            chunks = splitter.split_text(doc.page_content)

            if not chunks:
                logger.warning(
                    "Document '%s' : aucun chunk produit après découpage", source
                )
                continue

            total_chunks = len(chunks)

            for idx, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": f"{source}_{idx}",
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "word_count": len(chunk_text.split()),
                }

                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )

        logger.info(
            "Découpage terminé : %d document(s) → %d chunk(s)",
            len(documents), len(all_chunks),
        )
        return all_chunks

    @staticmethod
    def get_stats(chunks: list[Document]) -> dict:
        """
        Calcule des statistiques sur les chunks produits.

        Args:
            chunks: Liste de chunks LangChain.

        Returns:
            Dictionnaire avec les statistiques d'ingestion :
            - total_chunks : nombre total de chunks
            - avg_chunk_size : taille moyenne en caractères
            - min_chunk_size : taille du plus petit chunk
            - max_chunk_size : taille du plus grand chunk
            - avg_word_count : nombre moyen de mots par chunk
            - total_documents : nombre de fichiers sources uniques
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0.0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "avg_word_count": 0.0,
                "total_documents": 0,
            }

        sizes = [len(c.page_content) for c in chunks]
        word_counts = [len(c.page_content.split()) for c in chunks]

        # Compter les documents sources uniques
        sources: set[str] = set()
        for c in chunks:
            source = c.metadata.get(
                "file_name",
                c.metadata.get("source", "inconnu"),
            )
            sources.add(source)

        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(sizes) / len(sizes), 1),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
            "total_documents": len(sources),
        }

        logger.info(
            "Statistiques : %d chunks, taille moy. %.0f car., %d document(s) source(s)",
            stats["total_chunks"], stats["avg_chunk_size"], stats["total_documents"],
        )
        return stats
