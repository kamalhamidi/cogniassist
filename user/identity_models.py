"""
user/identity_models.py — Modèles d'identité (Layer 2 — Second cerveau).

Définit les tables SQLAlchemy de la couche d'extraction d'identité :
- StyleProfile     : empreinte stylistique de l'écriture de l'utilisateur.
- BeliefStore      : croyances / positions extraites des écrits personnels.
- StyleCorrection  : corrections stylistiques fournies par l'utilisateur.

Suit le même pattern que user/acpe_models.py.
"""

import json
import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, Date, DateTime,
)

from user.db import Base

logger = logging.getLogger("cogniassist.user")


class StyleProfile(Base):
    """Empreinte stylistique de l'écriture personnelle de l'utilisateur.

    Une seule ligne existe à la fois : le dernier profil calculé.
    """

    __tablename__ = "style_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    avg_sentence_len = Column(Float, default=0.0)
    vocabulary_richness = Column(Float, default=0.0)
    formality_score = Column(Float, default=0.0)
    first_person_ratio = Column(Float, default=0.0)
    hedging_ratio = Column(Float, default=0.0)
    example_preference = Column(String(20), default="balanced")
    preferred_length = Column(String(10), default="medium")
    tone = Column(String(20), default="analytical")
    style_prompt_fragment = Column(Text, default="")
    source_word_count = Column(Integer, default=0)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "avg_sentence_len": self.avg_sentence_len,
            "vocabulary_richness": self.vocabulary_richness,
            "formality_score": self.formality_score,
            "first_person_ratio": self.first_person_ratio,
            "hedging_ratio": self.hedging_ratio,
            "example_preference": self.example_preference,
            "preferred_length": self.preferred_length,
            "tone": self.tone,
            "style_prompt_fragment": self.style_prompt_fragment,
            "source_word_count": self.source_word_count,
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class BeliefStore(Base):
    """Croyances et positions extraites des écrits personnels de l'utilisateur."""

    __tablename__ = "belief_store"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(200), nullable=False)
    position = Column(Text, nullable=False)
    confidence = Column(String(10), default="medium")
    source_chunk_id = Column(String(100), nullable=True)
    date_written = Column(Date, nullable=True)
    status = Column(String(20), default="active")
    embedding_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    def get_embedding(self) -> list[float]:
        """Décode le vecteur d'embedding JSON."""
        try:
            return json.loads(self.embedding_json) if self.embedding_json else []
        except (json.JSONDecodeError, TypeError):
            return []

    def set_embedding(self, vector: list[float]) -> None:
        """Encode le vecteur d'embedding en JSON."""
        self.embedding_json = json.dumps(vector)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire (sans l'embedding volumineux)."""
        return {
            "id": self.id,
            "topic": self.topic,
            "position": self.position,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
            "date_written": self.date_written.isoformat() if self.date_written else None,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class StyleCorrection(Base):
    """Corrections stylistiques fournies par l'utilisateur sur les réponses générées."""

    __tablename__ = "style_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    generated = Column(Text, nullable=False)
    corrected = Column(Text, nullable=False)
    diff_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "query": self.query,
            "generated": self.generated,
            "corrected": self.corrected,
            "diff_summary": self.diff_summary,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
