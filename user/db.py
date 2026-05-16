"""
user/db.py — Connexion et gestion de la base SQLite.

Configure le moteur SQLAlchemy, la session factory et l'initialisation
des tables. Toutes les opérations utilisent check_same_thread=False
pour la compatibilité avec Streamlit.
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from config import settings

logger = logging.getLogger("cogniassist.user")


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles SQLAlchemy du projet."""
    pass


# Module-level engine and session factory (singleton)
_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """
    Crée ou retourne le moteur SQLAlchemy pour la base SQLite.

    Crée les répertoires parents si nécessaire.
    Utilise check_same_thread=False pour Streamlit.

    Returns:
        Moteur SQLAlchemy connecté à la base SQLite.
    """
    global _engine

    if _engine is None:
        db_path = settings.sqlite_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        logger.debug("Moteur SQLAlchemy créé : %s", db_path)

    return _engine


def get_session() -> Session:
    """
    Crée et retourne une nouvelle session SQLAlchemy.

    Returns:
        Session SQLAlchemy prête à l'emploi.
    """
    global _SessionFactory

    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())

    return _SessionFactory()


def init_db() -> None:
    """
    Crée toutes les tables si elles n'existent pas.

    Idempotente — peut être appelée plusieurs fois sans effet.
    Doit être appelée une fois au démarrage de l'application.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Base de données initialisée : %s", settings.sqlite_db_path)
