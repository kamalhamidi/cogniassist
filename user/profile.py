"""
user/profile.py — Modèle de profil utilisateur.

Définit le modèle SQLAlchemy pour le profil utilisateur et
fournit les opérations CRUD associées.
"""

from typing import Optional
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, JSON
from pydantic import BaseModel

from user.db import Base


# === Modèle SQLAlchemy (table) ===

class UserProfileModel(Base):
    """Modèle SQLAlchemy pour la table des profils utilisateurs."""

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True)
    username = Column(String(100), nullable=False, default="Utilisateur")
    email = Column(String(255), nullable=True)
    language = Column(String(10), default="fr")
    expertise_level = Column(String(20), default="intermédiaire")  # débutant, intermédiaire, expert
    interests = Column(Text, default="")  # Séparés par des virgules
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# === Schéma Pydantic (validation) ===

class UserProfile(BaseModel):
    """Schéma Pydantic pour valider un profil utilisateur."""

    user_id: str
    username: str = "Utilisateur"
    email: Optional[str] = None
    language: str = "fr"
    expertise_level: str = "intermédiaire"
    interests: list[str] = []
    preferences: dict = {}

    class Config:
        from_attributes = True


# === Opérations CRUD ===

class UserProfileManager:
    """Gère les opérations CRUD sur les profils utilisateurs."""

    def __init__(self, db_manager) -> None:
        """
        Args:
            db_manager: Instance de DatabaseManager.
        """
        self.db = db_manager

    def create_profile(self, profile: UserProfile) -> UserProfileModel:
        """
        Crée un nouveau profil utilisateur.

        Args:
            profile: Données du profil à créer.

        Returns:
            Le modèle créé.
        """
        session = self.db.get_session()
        try:
            db_profile = UserProfileModel(
                user_id=profile.user_id,
                username=profile.username,
                email=profile.email,
                language=profile.language,
                expertise_level=profile.expertise_level,
                interests=",".join(profile.interests),
                preferences=profile.preferences,
            )
            session.add(db_profile)
            session.commit()
            session.refresh(db_profile)
            return db_profile
        finally:
            session.close()

    def get_profile(self, user_id: str) -> Optional[UserProfileModel]:
        """
        Récupère un profil par son identifiant.

        Args:
            user_id: Identifiant de l'utilisateur.

        Returns:
            Le profil ou None s'il n'existe pas.
        """
        session = self.db.get_session()
        try:
            return session.query(UserProfileModel).filter_by(user_id=user_id).first()
        finally:
            session.close()

    def update_profile(self, user_id: str, **kwargs) -> Optional[UserProfileModel]:
        """
        Met à jour un profil existant.

        Args:
            user_id: Identifiant de l'utilisateur.
            **kwargs: Champs à mettre à jour.

        Returns:
            Le profil mis à jour ou None.
        """
        session = self.db.get_session()
        try:
            profile = session.query(UserProfileModel).filter_by(user_id=user_id).first()
            if profile:
                for key, value in kwargs.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(profile)
            return profile
        finally:
            session.close()

    def delete_profile(self, user_id: str) -> bool:
        """
        Supprime un profil utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur.

        Returns:
            True si le profil a été supprimé.
        """
        session = self.db.get_session()
        try:
            profile = session.query(UserProfileModel).filter_by(user_id=user_id).first()
            if profile:
                session.delete(profile)
                session.commit()
                return True
            return False
        finally:
            session.close()
