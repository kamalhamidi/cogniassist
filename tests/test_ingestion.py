"""
tests/test_ingestion.py — Tests unitaires du module ingestion.

Tests fonctionnels couvrant le loader, le cleaner, le chunker
et le pipeline ingest_file complet.
"""

import pytest
from pathlib import Path
from langchain.schema import Document


SAMPLE_FRENCH_TEXT = (
    "L'intelligence artificielle est un domaine de l'informatique "
    "qui vise à créer des systèmes capables de simuler l'intelligence humaine. "
    "Les réseaux de neurones profonds ont révolutionné ce domaine depuis 2012. "
    "Le traitement automatique du langage naturel permet aux machines de "
    "comprendre et de générer du texte en français, en arabe et dans bien "
    "d'autres langues. Les modèles de langage comme GPT utilisent des "
    "architectures Transformer pour produire des résultats impressionnants."
)


class TestDocumentLoader:
    """Tests du chargeur de documents."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.loader import DocumentLoader
        assert DocumentLoader is not None

    def test_load_txt(self, tmp_path: Path) -> None:
        """Charge un fichier .txt avec du texte français et vérifie le résultat."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(SAMPLE_FRENCH_TEXT, encoding="utf-8")

        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load(txt_file)

        assert len(docs) >= 1
        assert docs[0].page_content.strip() != ""
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["file_name"] == "sample.txt"

    def test_load_md(self, tmp_path: Path) -> None:
        """Vérifie que les fichiers .md sont traités comme du texte."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Titre\n\nContenu du document markdown.", encoding="utf-8")

        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load(md_file)

        assert len(docs) >= 1
        assert "Titre" in docs[0].page_content

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Vérifie qu'un format non supporté lève ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3", encoding="utf-8")

        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Format non supporté"):
            loader.load(csv_file)

    def test_file_not_found_raises(self) -> None:
        """Vérifie qu'un fichier inexistant lève FileNotFoundError."""
        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/chemin/inexistant/fichier.txt"))

    def test_load_from_bytes(self) -> None:
        """Vérifie le chargement depuis des bytes."""
        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        content = SAMPLE_FRENCH_TEXT.encode("utf-8")
        docs = loader.load_from_bytes(content, "test_bytes.txt")

        assert len(docs) >= 1
        assert "intelligence artificielle" in docs[0].page_content


class TestTextCleaner:
    """Tests du nettoyeur de texte."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.cleaner import TextCleaner
        assert TextCleaner is not None

    def test_clean_removes_extra_whitespace(self) -> None:
        """Vérifie la suppression des espaces et sauts de ligne superflus."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()

        dirty = Document(
            page_content="hello    world\n\n\n\ntest  content  here  with  enough  text  to  pass  the  minimum  length",
            metadata={"source": "test"},
        )
        result = cleaner.clean([dirty])

        assert len(result) == 1
        text = result[0].page_content
        # Pas d'espaces multiples
        assert "    " not in text
        assert "hello world" in text
        # Max 2 newlines consécutifs
        assert "\n\n\n" not in text
        assert "test content here" in text
        # Métadonnées préservées
        assert result[0].metadata["source"] == "test"

    def test_clean_normalizes_unicode(self) -> None:
        """Vérifie la normalisation NFKC des ligatures."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()

        doc = Document(
            page_content="Le ﬁchier contient des ligatures ﬂ et des caractères spéciaux",
            metadata={"source": "test"},
        )
        result = cleaner.clean([doc])

        assert len(result) == 1
        assert "fi" in result[0].page_content
        assert "fl" in result[0].page_content

    def test_clean_removes_control_chars(self) -> None:
        """Vérifie la suppression des caractères de contrôle."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()

        doc = Document(
            page_content="Texte\x00avec\x01des\x02caractères de contrôle mais assez long pour passer",
            metadata={"source": "test"},
        )
        result = cleaner.clean([doc])

        assert len(result) == 1
        assert "\x00" not in result[0].page_content
        assert "\x01" not in result[0].page_content

    def test_clean_filters_short_documents(self) -> None:
        """Vérifie que les documents trop courts sont filtrés."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()

        docs = [
            Document(page_content="Court", metadata={"source": "short.txt"}),
            Document(page_content=SAMPLE_FRENCH_TEXT, metadata={"source": "long.txt"}),
        ]
        result = cleaner.clean(docs)

        assert len(result) == 1
        assert result[0].metadata["source"] == "long.txt"

    def test_clean_empty_list(self) -> None:
        """Vérifie le comportement avec une liste vide."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()
        assert cleaner.clean([]) == []


class TestDocumentChunker:
    """Tests du chunker de documents."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.chunker import DocumentChunker
        assert DocumentChunker is not None

    def test_chunk_enriches_metadata(self, tmp_path: Path) -> None:
        """Vérifie que le chunking enrichit les métadonnées de chaque chunk."""
        # Créer un texte de 2000+ caractères
        long_text = (SAMPLE_FRENCH_TEXT + " ") * 6  # ~2700 caractères

        doc = Document(
            page_content=long_text,
            metadata={"source": "test.txt", "file_name": "test.txt"},
        )

        from ingestion.chunker import DocumentChunker
        chunker = DocumentChunker()
        chunks = chunker.chunk([doc])

        assert len(chunks) > 1, "Le texte devrait produire plusieurs chunks"

        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "word_count" in chunk.metadata
            assert isinstance(chunk.metadata["chunk_index"], int)
            assert isinstance(chunk.metadata["word_count"], int)
            assert chunk.metadata["word_count"] > 0

    def test_chunk_empty_list(self) -> None:
        """Vérifie le comportement avec une liste vide."""
        from ingestion.chunker import DocumentChunker
        chunker = DocumentChunker()
        assert chunker.chunk([]) == []

    def test_get_stats(self) -> None:
        """Vérifie les statistiques de chunks."""
        from ingestion.chunker import DocumentChunker
        chunker = DocumentChunker()

        chunks = [
            Document(page_content="Chunk un contenu", metadata={"file_name": "a.txt"}),
            Document(page_content="Chunk deux contenu ici", metadata={"file_name": "a.txt"}),
            Document(page_content="Chunk trois autre fichier", metadata={"file_name": "b.txt"}),
        ]
        stats = chunker.get_stats(chunks)

        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert stats["avg_chunk_size"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] >= stats["min_chunk_size"]

    def test_get_stats_empty(self) -> None:
        """Vérifie les stats avec une liste vide."""
        from ingestion.chunker import DocumentChunker
        stats = DocumentChunker.get_stats([])
        assert stats["total_chunks"] == 0


class TestIngestFilePipeline:
    """Tests du pipeline complet d'ingestion."""

    def test_ingest_file_pipeline(self) -> None:
        """Vérifie le pipeline complet avec un fichier texte en bytes."""
        from ingestion import ingest_file

        file_bytes = SAMPLE_FRENCH_TEXT.encode("utf-8")
        chunks, stats = ingest_file(file_bytes, "test_pipeline.txt")

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(stats, dict)
        assert stats["total_chunks"] > 0
        assert stats["total_documents"] >= 1

    def test_ingest_file_unsupported_raises(self) -> None:
        """Vérifie que le pipeline lève ValueError pour un format non supporté."""
        from ingestion import ingest_file

        with pytest.raises(ValueError, match="Format non supporté"):
            ingest_file(b"some data", "file.csv")
