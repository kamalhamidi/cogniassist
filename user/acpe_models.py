"""
user/acpe_models.py — Modèles ACPE (Adaptive Cognitive Profiling Engine).

Définit les tables SQLAlchemy pour le profil de connaissances,
les patterns d'utilisation, et le progressive profiling.
"""

import json
import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, ForeignKey,
)

from user.db import Base

logger = logging.getLogger("cogniassist.user")


class KnowledgeProfile(Base):
    """Table des profils de connaissances par domaine."""

    __tablename__ = "knowledge_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
        index=True,
    )
    domain = Column(String(100), nullable=False)
    mastery_score = Column(Integer, default=50)
    confidence = Column(Integer, default=0)
    interaction_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "domain": self.domain,
            "mastery_score": self.mastery_score,
            "confidence": self.confidence,
            "interaction_count": self.interaction_count,
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class UsagePattern(Base):
    """Table des patterns d'utilisation comportementaux."""

    __tablename__ = "usage_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
        index=True,
    )
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_value(self) -> any:
        """Décode la valeur JSON."""
        try:
            return json.loads(self.metric_value)
        except (json.JSONDecodeError, TypeError):
            return self.metric_value

    def set_value(self, value: any) -> None:
        """Encode la valeur en JSON."""
        self.metric_value = json.dumps(value, ensure_ascii=False)


class ProgressivePrompt(Base):
    """Table des suggestions de profiling progressif."""

    __tablename__ = "progressive_prompts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(50),
        ForeignKey("user_profiles.user_id"),
        nullable=False,
        index=True,
    )
    prompt_key = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    action_data = Column(Text, nullable=True)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    def get_action_data(self) -> dict:
        """Décode les données d'action JSON."""
        try:
            return json.loads(self.action_data) if self.action_data else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "prompt_key": self.prompt_key,
            "message": self.message,
            "action_data": self.get_action_data(),
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
