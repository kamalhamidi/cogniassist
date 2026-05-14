"""
CogniAssist — Configuration centralisée.

Ce module charge les variables d'environnement depuis le fichier .env,
définit une classe Settings validée par Pydantic, et expose une instance
globale `settings` importable dans tout le projet.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Paramètres globaux de l'application CogniAssist."""

    # === OpenAI ===
    OPENAI_API_KEY: str = "your_openai_api_key_here"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # === Embeddings ===
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

    # === ChromaDB ===
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # === SQLite ===
    SQLITE_DB_PATH: str = "./data/cogniassist.db"

    # === Uploads ===
    UPLOAD_DIR: str = "./data/uploads"

    # === RAG Parameters ===
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_RETRIEVED_DOCS: int = 5

    # === Application ===
    APP_NAME: str = "CogniAssist"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Vérifie que la clé API OpenAI est définie."""
        if v == "your_openai_api_key_here" or not v:
            import warnings
            warnings.warn(
                "⚠️  OPENAI_API_KEY n'est pas configurée. "
                "Veuillez la définir dans le fichier .env pour utiliser les fonctionnalités LLM.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @property
    def chroma_persist_path(self) -> Path:
        """Chemin absolu vers le répertoire ChromaDB."""
        return (BASE_DIR / self.CHROMA_PERSIST_DIR).resolve()

    @property
    def sqlite_db_path(self) -> Path:
        """Chemin absolu vers la base SQLite."""
        return (BASE_DIR / self.SQLITE_DB_PATH).resolve()

    @property
    def upload_dir_path(self) -> Path:
        """Chemin absolu vers le répertoire d'uploads."""
        return (BASE_DIR / self.UPLOAD_DIR).resolve()

    def ensure_directories(self) -> None:
        """Crée les répertoires de données s'ils n'existent pas."""
        for directory in [
            self.chroma_persist_path,
            self.upload_dir_path,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Créer le répertoire parent de la base SQLite
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


# Instance globale — importable partout via `from config import settings`
settings = Settings()
settings.ensure_directories()
