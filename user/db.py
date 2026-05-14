"""
user/db.py — Connexion et gestion de la base SQLite.

Utilise SQLAlchemy pour gérer la connexion à la base de données
SQLite et fournir les sessions pour les opérations CRUD.
"""

from typing import Optional
from pathlib import Path

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase


class Base(DeclarativeBase):
    """Classe de base pour les modèles SQLAlchemy."""
    pass


class DatabaseManager:
    """
    Gestionnaire de base de données SQLite via SQLAlchemy.

    Gère la création de la base, des tables et fournit
    des sessions pour les opérations de lecture/écriture.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialise le gestionnaire de base de données.

        Args:
            db_path: Chemin vers le fichier SQLite.
                    Si None, utilise les settings.
        """
        if db_path is None:
            from config import settings
            db_path = str(settings.sqlite_db_path)

        self.db_path = db_path

        # S'assurer que le répertoire parent existe
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Créer le moteur SQLAlchemy
        self._engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Session factory
        self._session_factory = sessionmaker(bind=self._engine)

    def create_tables(self) -> None:
        """Crée toutes les tables définies dans les modèles."""
        Base.metadata.create_all(self._engine)

    def get_session(self) -> Session:
        """
        Retourne une nouvelle session de base de données.

        Returns:
            Session SQLAlchemy prête à l'emploi.
        """
        return self._session_factory()

    @property
    def engine(self):
        """Retourne le moteur SQLAlchemy."""
        return self._engine
