"""
ingestion/loader.py — Chargement de documents multi-formats.

Supporte les formats PDF (via PyMuPDF), DOCX (via python-docx)
et TXT. Retourne le texte brut extrait de chaque document.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Représente un document chargé avec ses métadonnées."""

    content: str
    filename: str
    file_type: str
    metadata: dict = field(default_factory=dict)
    page_count: Optional[int] = None


class DocumentLoader:
    """
    Charge des documents depuis le système de fichiers.

    Supporte les formats :
    - PDF (.pdf) via PyMuPDF (fitz)
    - Word (.docx) via python-docx
    - Texte (.txt) encodage UTF-8
    """

    SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

    def load(self, file_path: Path) -> Document:
        """
        Charge un document depuis un chemin de fichier.

        Args:
            file_path: Chemin vers le fichier à charger.

        Returns:
            Document avec le contenu textuel extrait.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le format n'est pas supporté.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Format non supporté : '{extension}'. "
                f"Formats acceptés : {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if extension == ".pdf":
            return self._load_pdf(file_path)
        elif extension == ".docx":
            return self._load_docx(file_path)
        else:
            return self._load_txt(file_path)

    def _load_pdf(self, file_path: Path) -> Document:
        """Charge un fichier PDF avec PyMuPDF."""
        import fitz  # PyMuPDF

        text_parts: list[str] = []
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                text_parts.append(page.get_text())

            return Document(
                content="\n".join(text_parts),
                filename=file_path.name,
                file_type="pdf",
                page_count=len(doc),
                metadata={"source": str(file_path)},
            )

    def _load_docx(self, file_path: Path) -> Document:
        """Charge un fichier DOCX avec python-docx."""
        from docx import Document as DocxDocument

        doc = DocxDocument(str(file_path))
        text_parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

        return Document(
            content="\n".join(text_parts),
            filename=file_path.name,
            file_type="docx",
            page_count=None,
            metadata={"source": str(file_path)},
        )

    def _load_txt(self, file_path: Path) -> Document:
        """Charge un fichier texte brut."""
        content = file_path.read_text(encoding="utf-8")

        return Document(
            content=content,
            filename=file_path.name,
            file_type="txt",
            page_count=None,
            metadata={"source": str(file_path)},
        )

    def load_directory(self, directory: Path) -> list[Document]:
        """
        Charge tous les documents supportés d'un répertoire.

        Args:
            directory: Chemin vers le répertoire.

        Returns:
            Liste des documents chargés.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Répertoire introuvable : {directory}")

        documents: list[Document] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    documents.append(self.load(file_path))
                except Exception as e:
                    print(f"⚠️  Erreur lors du chargement de {file_path.name} : {e}")

        return documents
