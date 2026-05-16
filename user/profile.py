"""
user/profile.py — Modèles et gestion des profils utilisateurs.

Définit les tables SQLAlchemy pour les profils, préférences et
documents, ainsi que le UserProfileManager pour les opérations CRUD.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, ForeignKey, Boolean,
)
from sqlalchemy.orm import relationship

from user.db import Base, get_session

logger = logging.getLogger("cogniassist.user")


# ═══════════════════════════════════════════════════════════
# Modèles SQLAlchemy
# ═══════════════════════════════════════════════════════════

class UserProfile(Base):
    """Table des profils utilisateurs."""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), default="Utilisateur")
    email = Column(String(200), nullable=True)
    avatar = Column(String(10), default="🧠")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPreferences(Base):
    """Table des préférences utilisateurs."""

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
    )
    language = Column(String(10), default="fr")
    response_style = Column(String(20), default="detailed")
    domain_focus = Column(String(200), default="")
    goals = Column(Text, default="")
    expertise_level = Column(String(20), default="intermediate")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentAccess(Base):
    """Table de suivi des documents uploadés et consultés."""

    __tablename__ = "document_access"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
    )
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10))
    upload_date = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)


# ═══════════════════════════════════════════════════════════
# Gestionnaire de profils
# ═══════════════════════════════════════════════════════════

class UserProfileManager:
    """
    Gère les profils utilisateurs, préférences et documents.

    Crée automatiquement le profil par défaut si inexistant.
    Fournit un contexte de personnalisation pour le pipeline RAG.
    """

    def __init__(self, user_id: str = "default") -> None:
        """
        Initialise le gestionnaire pour un utilisateur donné.

        Args:
            user_id: Identifiant de l'utilisateur.
        """
        self.user_id = user_id
        self.session = get_session()
        self._ensure_user_exists()

    def _ensure_user_exists(self) -> None:
        """Crée le profil et les préférences par défaut si inexistants."""
        try:
            existing = (
                self.session.query(UserProfile)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if existing is None:
                profile = UserProfile(user_id=self.user_id)
                prefs = UserPreferences(user_id=self.user_id)
                self.session.add(profile)
                self.session.add(prefs)
                self.session.commit()
                logger.info("Profil créé pour l'utilisateur '%s'.", self.user_id)
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur création profil '%s' : %s", self.user_id, e)
            raise

    def get_profile(self) -> dict:
        """
        Retourne le profil complet de l'utilisateur.

        Returns:
            Dictionnaire avec toutes les informations du profil et préférences.
        """
        profile = (
            self.session.query(UserProfile)
            .filter_by(user_id=self.user_id)
            .first()
        )
        prefs = (
            self.session.query(UserPreferences)
            .filter_by(user_id=self.user_id)
            .first()
        )

        domain_list = []
        if prefs and prefs.domain_focus:
            domain_list = [d.strip() for d in prefs.domain_focus.split(",") if d.strip()]

        return {
            "user_id": self.user_id,
            "name": profile.name if profile else "Utilisateur",
            "email": profile.email if profile else None,
            "avatar": profile.avatar if profile else "🧠",
            "created_at": profile.created_at.isoformat() if profile and profile.created_at else "",
            "language": prefs.language if prefs else "fr",
            "response_style": prefs.response_style if prefs else "detailed",
            "domain_focus": domain_list,
            "goals": prefs.goals if prefs else "",
            "expertise_level": prefs.expertise_level if prefs else "intermediate",
        }

    def update_profile(self, **kwargs) -> None:
        """
        Met à jour les champs du profil utilisateur.

        Args:
            **kwargs: Champs acceptés : name, email, avatar.
        """
        allowed = {"name", "email", "avatar"}
        try:
            profile = (
                self.session.query(UserProfile)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if profile:
                for key, value in kwargs.items():
                    if key in allowed:
                        setattr(profile, key, value)
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur update profil : %s", e)

    def update_preferences(self, **kwargs) -> None:
        """
        Met à jour les préférences utilisateur.

        Args:
            **kwargs: Champs acceptés : language, response_style,
                      expertise_level, goals, domain_focus.
                      domain_focus accepte list[str] ou str CSV.
        """
        allowed = {"language", "response_style", "expertise_level", "goals", "domain_focus"}
        try:
            prefs = (
                self.session.query(UserPreferences)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if prefs:
                for key, value in kwargs.items():
                    if key not in allowed:
                        continue
                    if key == "domain_focus" and isinstance(value, list):
                        value = ",".join(value)
                    setattr(prefs, key, value)
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur update préférences : %s", e)

    def get_personalization_context(self) -> str:
        """
        Construit un résumé du profil pour injection dans le prompt RAG.

        Returns:
            Chaîne de contexte utilisateur formatée.
        """
        p = self.get_profile()
        domains = ", ".join(p["domain_focus"]) if p["domain_focus"] else "Non définis"
        goals = p["goals"][:200] if p["goals"] else "Non définis"

        return (
            f"Profil : {p['name']} | Niveau : {p['expertise_level']} | Langue : {p['language']}\n"
            f"Style de réponse souhaité : {p['response_style']}\n"
            f"Domaines d'intérêt : {domains}\n"
            f"Objectifs : {goals}"
        )

    def register_document(
        self, file_name: str, file_type: str, chunk_count: int,
    ) -> None:
        """
        Enregistre ou met à jour un document pour cet utilisateur.

        Args:
            file_name: Nom du fichier.
            file_type: Type de fichier (pdf, docx, txt).
            chunk_count: Nombre de chunks produits.
        """
        try:
            existing = (
                self.session.query(DocumentAccess)
                .filter_by(user_id=self.user_id, file_name=file_name)
                .first()
            )
            if existing:
                existing.chunk_count = chunk_count
                existing.upload_date = datetime.utcnow()
            else:
                doc = DocumentAccess(
                    user_id=self.user_id,
                    file_name=file_name,
                    file_type=file_type,
                    chunk_count=chunk_count,
                )
                self.session.add(doc)
            self.session.commit()
            logger.info("Document '%s' enregistré pour '%s'.", file_name, self.user_id)
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur enregistrement document : %s", e)

    def record_document_access(self, file_name: str) -> None:
        """
        Incrémente le compteur d'accès et met à jour la date de dernier accès.

        Args:
            file_name: Nom du fichier consulté.
        """
        try:
            doc = (
                self.session.query(DocumentAccess)
                .filter_by(user_id=self.user_id, file_name=file_name)
                .first()
            )
            if doc:
                doc.access_count = (doc.access_count or 0) + 1
                doc.last_accessed = datetime.utcnow()
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur record access : %s", e)

    def store_document_summary(self, file_name: str, summary: str) -> None:
        """
        Sauvegarde le résumé généré pour un document.

        Args:
            file_name: Nom du fichier.
            summary: Texte du résumé.
        """
        try:
            doc = (
                self.session.query(DocumentAccess)
                .filter_by(user_id=self.user_id, file_name=file_name)
                .first()
            )
            if doc:
                doc.summary = summary
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur store summary : %s", e)

    def get_user_documents(self) -> list[dict]:
        """
        Retourne tous les documents de cet utilisateur.

        Returns:
            Liste de dictionnaires ordonnés par date d'upload décroissante.
        """
        docs = (
            self.session.query(DocumentAccess)
            .filter_by(user_id=self.user_id)
            .order_by(DocumentAccess.upload_date.desc())
            .all()
        )
        return [
            {
                "file_name": d.file_name,
                "file_type": d.file_type,
                "upload_date": d.upload_date.isoformat() if d.upload_date else "",
                "access_count": d.access_count or 0,
                "last_accessed": d.last_accessed.isoformat() if d.last_accessed else None,
                "has_summary": bool(d.summary),
                "chunk_count": d.chunk_count or 0,
            }
            for d in docs
        ]

    def delete_document(self, file_name: str) -> None:
        """
        Supprime un document et ses chunks du vectorstore.

        Args:
            file_name: Nom du fichier à supprimer.
        """
        try:
            doc = (
                self.session.query(DocumentAccess)
                .filter_by(user_id=self.user_id, file_name=file_name)
                .first()
            )
            if doc:
                self.session.delete(doc)
                self.session.commit()

            # Supprimer aussi les chunks de ChromaDB
            from vectorstore.store import VectorStore
            vs = VectorStore()
            vs.delete_document(file_name)

            logger.info("Document '%s' supprimé.", file_name)
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur suppression document : %s", e)
