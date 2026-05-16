"""
ingestion/loader.py — Chargement de documents multi-formats.

Détecte automatiquement le type de fichier et utilise le loader
approprié pour extraire le texte. Retourne des objets LangChain Document.

Formats supportés : PDF (.pdf), Word (.docx, .doc), Texte (.txt, .md)
"""

import logging
from pathlib import Path

from langchain.schema import Document

from config import settings

logger = logging.getLogger("cogniassist.ingestion")

# Extensions supportées avec leur type logique
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "txt",
}


class DocumentLoader:
    """
    Charge des documents depuis le système de fichiers ou depuis des bytes.

    Supporte les formats :
    - PDF (.pdf) via PyMuPDF (fitz) — extraction page par page
    - Word (.docx, .doc) via python-docx — paragraphes groupés par 3
    - Texte (.txt, .md) — lecture brute avec détection d'encodage
    """

    SUPPORTED_EXTENSIONS: dict[str, str] = SUPPORTED_EXTENSIONS

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Détecte le type de fichier et appelle le loader approprié.

        Args:
            file_path: Chemin vers le fichier à charger.

        Returns:
            Liste d'objets LangChain Document.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si l'extension n'est pas supportée.
            RuntimeError: Si la lecture du fichier échoue.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Format non supporté : '{extension}'. "
                f"Formats acceptés : {', '.join(sorted(self.SUPPORTED_EXTENSIONS.keys()))}"
            )

        file_type = self.SUPPORTED_EXTENSIONS[extension]
        logger.info("Chargement du fichier '%s' (type : %s)", file_path.name, file_type)

        try:
            if file_type == "pdf":
                return self._load_pdf(file_path)
            elif file_type == "docx":
                return self._load_docx(file_path)
            else:
                return self._load_txt(file_path)
        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error("Erreur lors de la lecture de '%s' : %s", file_path.name, e)
            raise RuntimeError(
                f"Impossible de lire le fichier '{file_path.name}' : {e}"
            ) from e

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """
        Charge un fichier PDF avec PyMuPDF, page par page.

        Chaque page non vide produit un Document distinct avec les métadonnées :
        source, file_name, page_number, total_pages, file_type.

        Args:
            file_path: Chemin absolu vers le fichier PDF.

        Returns:
            Liste de Documents (un par page non vide).
        """
        import fitz  # PyMuPDF

        documents: list[Document] = []

        with fitz.open(str(file_path)) as pdf:
            total_pages = len(pdf)
            logger.debug("PDF '%s' : %d pages détectées", file_path.name, total_pages)

            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()

                # Ignorer les pages vides silencieusement
                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "file_type": "pdf",
                        },
                    )
                )

        logger.info(
            "PDF '%s' : %d pages extraites sur %d",
            file_path.name, len(documents), total_pages,
        )
        return documents

    def _load_docx(self, file_path: Path) -> list[Document]:
        """
        Charge un fichier DOCX avec python-docx.

        Les paragraphes sont groupés par blocs de 3 pour éviter
        de produire trop de petits Documents.

        Args:
            file_path: Chemin absolu vers le fichier DOCX.

        Returns:
            Liste de Documents (un par bloc de 3 paragraphes).
        """
        from docx import Document as DocxDocument

        doc = DocxDocument(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            logger.warning("DOCX '%s' : aucun paragraphe non vide trouvé", file_path.name)
            return []

        # Grouper les paragraphes par blocs de 3
        block_size = 3
        documents: list[Document] = []

        for i in range(0, len(paragraphs), block_size):
            block = paragraphs[i:i + block_size]
            text = "\n".join(block)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "docx",
                    },
                )
            )

        logger.info(
            "DOCX '%s' : %d paragraphes → %d blocs",
            file_path.name, len(paragraphs), len(documents),
        )
        return documents

    def _load_txt(self, file_path: Path) -> list[Document]:
        """
        Charge un fichier texte brut avec détection d'encodage.

        Tente UTF-8 en premier, puis Latin-1 en fallback.
        Retourne un seul Document contenant le texte complet.

        Args:
            file_path: Chemin absolu vers le fichier texte.

        Returns:
            Liste contenant un seul Document.
        """
        content: str | None = None

        # Essayer UTF-8 d'abord
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug(
                "Fichier '%s' : échec UTF-8, tentative Latin-1", file_path.name
            )
            try:
                content = file_path.read_text(encoding="latin-1")
            except UnicodeDecodeError as e:
                logger.error(
                    "Fichier '%s' : impossible de décoder (ni UTF-8 ni Latin-1)",
                    file_path.name,
                )
                raise RuntimeError(
                    f"Impossible de décoder le fichier '{file_path.name}' : {e}"
                ) from e

        if not content or not content.strip():
            logger.warning("Fichier texte '%s' : contenu vide", file_path.name)
            return []

        logger.info(
            "Texte '%s' : %d caractères chargés", file_path.name, len(content)
        )

        return [
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "txt",
                },
            )
        ]

    def load_from_bytes(self, file_bytes: bytes, file_name: str) -> list[Document]:
        """
        Charge un fichier depuis des bytes bruts (ex: Streamlit file_uploader).

        Sauvegarde temporairement le fichier dans data/uploads/,
        appelle load(), puis nettoie le fichier temporaire.

        Args:
            file_bytes: Contenu binaire du fichier.
            file_name: Nom du fichier (avec extension).

        Returns:
            Liste d'objets LangChain Document.

        Raises:
            ValueError: Si l'extension n'est pas supportée.
            RuntimeError: Si la lecture échoue.
        """
        upload_dir = settings.upload_dir_path
        upload_dir.mkdir(parents=True, exist_ok=True)

        temp_path = upload_dir / file_name
        logger.debug("Sauvegarde temporaire : %s", temp_path)

        try:
            temp_path.write_bytes(file_bytes)
            documents = self.load(temp_path)
            return documents
        finally:
            # Toujours nettoyer le fichier temporaire
            if temp_path.exists():
                temp_path.unlink()
                logger.debug("Fichier temporaire supprimé : %s", temp_path)
