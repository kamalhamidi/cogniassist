"""
user/db.py — Connexion et gestion de la base SQLite.

Configure le moteur SQLAlchemy, la session factory et l'initialisation
des tables. Toutes les opérations utilisent check_same_thread=False
pour la compatibilité avec Streamlit.
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine, Engine, text, inspect
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

    # Migration à la volée des colonnes ACPE si la table existait déjà
    try:
        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("user_profiles")]
        
        with engine.begin() as conn:
            if "user_type" not in columns:
                conn.execute(
                    text("ALTER TABLE user_profiles ADD COLUMN user_type VARCHAR(20) DEFAULT 'individual'")
                )
                logger.info("Colonne 'user_type' ajoutée à 'user_profiles'.")
            if "onboarding_completed" not in columns:
                conn.execute(
                    text("ALTER TABLE user_profiles ADD COLUMN onboarding_completed BOOLEAN DEFAULT 0")
                )
                logger.info("Colonne 'onboarding_completed' ajoutée à 'user_profiles'.")
            if "role" not in columns:
                conn.execute(
                    text("ALTER TABLE user_profiles ADD COLUMN role VARCHAR(30) NULL")
                )
                logger.info("Colonne 'role' ajoutée à 'user_profiles'.")
            if "organization_data" not in columns:
                conn.execute(
                    text("ALTER TABLE user_profiles ADD COLUMN organization_data TEXT NULL")
                )
                logger.info("Colonne 'organization_data' ajoutée à 'user_profiles'.")
            if "adaptive_learning_enabled" not in columns:
                conn.execute(
                    text("ALTER TABLE user_profiles ADD COLUMN adaptive_learning_enabled BOOLEAN DEFAULT 1")
                )
                logger.info("Colonne 'adaptive_learning_enabled' ajoutée à 'user_profiles'.")
    except Exception as e:
        logger.error("Erreur lors de la migration à la volée du schéma : %s", e)

    logger.info("Base de données initialisée : %s", settings.sqlite_db_path)


def reset_system() -> None:
    """
    Réinitialise complètement le système CogniAssist.
    Supprime toutes les données :
    - Tables de la base SQLite
    - Collection ChromaDB
    - Index BM25 (fichier pickle)
    - Fichiers importés (uploads)
    Puis réinitialise une base de données SQLite propre.
    """
    logger.warning("Début de la réinitialisation complète du système...")

    # 0. Réinitialiser le singleton RAGPipeline si présent
    try:
        from rag import reset_pipeline
        reset_pipeline()
    except Exception as e:
        logger.error("Erreur réinitialisation singleton RAGPipeline : %s", e)

    # 1. Réinitialiser ChromaDB
    try:
        from vectorstore.store import VectorStore
        vs = VectorStore()
        vs.reset_collection()
        logger.info("ChromaDB réinitialisé.")
    except Exception as e:
        logger.error("Erreur réinitialisation ChromaDB : %s", e)

    # 2. Supprimer l'index BM25
    try:
        from config import BASE_DIR
        bm25_full_path = (BASE_DIR / settings.BM25_INDEX_PATH).resolve()
        if bm25_full_path.exists():
            bm25_full_path.unlink()
            logger.info("Index BM25 supprimé.")
    except Exception as e:
        logger.error("Erreur suppression index BM25 : %s", e)

    # 3. Vider le dossier uploads
    try:
        import shutil
        upload_dir = settings.upload_dir_path
        if upload_dir.exists():
            for item in upload_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("Dossier des uploads vidé.")
    except Exception as e:
        logger.error("Erreur nettoyage dossier des uploads : %s", e)

    # 4. Supprimer toutes les tables SQLite et réinitialiser la base
    try:
        from sqlalchemy.orm import close_all_sessions
        # Fermer toutes les sessions actives pour libérer les verrous SQLite
        close_all_sessions()
        
        global _engine, _SessionFactory
        engine = get_engine()
        engine.dispose()

        # Charger tous les modèles pour que metadata.drop_all() sache quelles tables supprimer
        from user.profile import UserProfile, UserPreferences, DocumentAccess
        from user.history import Interaction
        from user.acpe_models import KnowledgeProfile, UsagePattern, ProgressivePrompt
        
        Base.metadata.drop_all(engine)
        
        # Réinitialiser les singletons pour forcer get_engine() à recréer l'engine
        _engine = None
        _SessionFactory = None
        
        init_db()
        logger.info("Base de données SQLite réinitialisée.")
    except Exception as e:
        logger.error("Erreur réinitialisation SQLite : %s", e)
