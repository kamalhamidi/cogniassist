"""
tests/test_ingestion.py — Tests unitaires du module ingestion.

Tests de smoke pour vérifier que les imports fonctionnent
et que les classes s'instancient correctement.
"""

import pytest
from pathlib import Path


class TestDocumentLoader:
    """Tests du chargeur de documents."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.loader import DocumentLoader
        assert DocumentLoader is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie."""
        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        assert loader is not None
        assert loader.SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".txt"}

    def test_document_dataclass(self) -> None:
        """Vérifie le dataclass Document."""
        from ingestion.loader import Document
        doc = Document(content="Hello", filename="test.txt", file_type="txt")
        assert doc.content == "Hello"
        assert doc.filename == "test.txt"
        assert doc.file_type == "txt"
        assert doc.metadata == {}

    def test_unsupported_format(self) -> None:
        """Vérifie qu'un format non supporté lève une erreur."""
        from ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/fichier/inexistant.pdf"))


class TestTextChunker:
    """Tests du chunker de texte."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.chunker import TextChunker
        assert TextChunker is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie avec les paramètres par défaut."""
        from ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_empty_text(self) -> None:
        """Vérifie que le chunking d'un texte vide retourne une liste vide."""
        from ingestion.chunker import TextChunker
        chunker = TextChunker()
        result = chunker.split("")
        assert result == []


class TestTextCleaner:
    """Tests du nettoyeur de texte."""

    def test_import(self) -> None:
        """Vérifie que l'import fonctionne."""
        from ingestion.cleaner import TextCleaner
        assert TextCleaner is not None

    def test_instantiation(self) -> None:
        """Vérifie que la classe s'instancie."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()
        assert cleaner is not None

    def test_clean_empty_text(self) -> None:
        """Vérifie le nettoyage d'un texte vide."""
        from ingestion.cleaner import TextCleaner
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""

    def test_normalize_whitespace(self) -> None:
        """Vérifie la normalisation des espaces."""
        from ingestion.cleaner import TextCleaner
        result = TextCleaner.normalize_whitespace("hello   world")
        assert result == "hello world"

    def test_normalize_line_breaks(self) -> None:
        """Vérifie la normalisation des sauts de ligne."""
        from ingestion.cleaner import TextCleaner
        result = TextCleaner.normalize_line_breaks("a\n\n\n\nb")
        assert result == "a\n\nb"
