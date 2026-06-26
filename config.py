"""
CogniAssist — Configuration centralisée.

Ce module charge les variables d'environnement depuis le fichier .env,
définit une classe Settings validée par Pydantic, et expose une instance
globale `settings` importable dans tout le projet.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Paramètres globaux de l'application CogniAssist."""

    # === Ollama ===
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="mistral:7b",
        validation_alias="OLLAMA_MODEL",
    )

    # === Embeddings ===
    embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias="EMBEDDING_MODEL",
    )

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

    # === Hybrid Retrieval ===
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_SPARSE_WEIGHT: float = 0.4

    # === Identity Layer (Layer 2 — Second Brain) ===
    PERSONAL_WRITING_COLLECTION: str = "personal_writing"
    IDENTITY_MODE_DEFAULT: bool = False
    MIN_BELIEFS_FOR_IDENTITY_MODE: int = 3
    BELIEF_CONFLICT_THRESHOLD: float = 0.85  # similarité cosinus
    BELIEF_TOPIC_SIMILARITY_THRESHOLD: float = 0.80

    # === Application ===
    APP_NAME: str = "CogniAssist"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

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
