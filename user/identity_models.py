"""
user/identity_models.py — Modèles d'identité (Layer 2 — Second cerveau).

Définit les tables SQLAlchemy de la couche d'extraction d'identité :
- StyleProfile     : empreinte stylistique de l'écriture de l'utilisateur.
- BeliefStore      : croyances / positions extraites des écrits personnels.
- StyleCorrection  : corrections stylistiques fournies par l'utilisateur.

Layer 6 — La boucle de rétroaction (apprentissage continu) :
- FeedbackSignal       : chaque évènement de rétroaction et son contexte.
- StyleCalibrationLog  : journal des recalculs du profil de style.
- BeliefTimeline       : historique immuable des changements de croyances.
- IdentityFidelityLog  : score de fidélité vocale par réponse.

Suit le même pattern que user/acpe_models.py.
"""

import json
import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, Date, DateTime, Boolean, ForeignKey,
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
    # Layer 6 — nombre de 👍 ayant renforcé ce profil de style
    reinforcement_count = Column(Integer, default=0)
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
            "reinforcement_count": self.reinforcement_count or 0,
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
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "query": self.query,
            "generated": self.generated,
            "corrected": self.corrected,
            "diff_summary": self.diff_summary,
            "processed": bool(self.processed),
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


# ═══════════════════════════════════════════════════════════════════════
# Layer 6 — La boucle de rétroaction (Feedback Loop)
# ═══════════════════════════════════════════════════════════════════════


class FeedbackSignal(Base):
    """Trace chaque évènement de rétroaction avec son contexte d'apprentissage.

    Chaque interaction (correction, pouce, réécriture, édition de croyance)
    devient un signal d'entraînement pour affiner le modèle d'identité.
    """

    __tablename__ = "feedback_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    interaction_id = Column(
        Integer, ForeignKey("interactions.id"), nullable=True, index=True,
    )
    # style_correction | belief_confirm | belief_reject | thumbs_up |
    # thumbs_down | belief_changed | manual_edit
    signal_type = Column(String(30), nullable=False, index=True)
    context_json = Column(Text, nullable=True)
    delta_json = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def get_context(self) -> dict:
        """Décode le contexte JSON."""
        try:
            return json.loads(self.context_json) if self.context_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_delta(self) -> dict:
        """Décode le delta JSON."""
        try:
            return json.loads(self.delta_json) if self.delta_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "interaction_id": self.interaction_id,
            "signal_type": self.signal_type,
            "context": self.get_context(),
            "delta": self.get_delta(),
            "processed": bool(self.processed),
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class StyleCalibrationLog(Base):
    """Journal append-only de chaque recalcul du profil de style et de sa cause."""

    __tablename__ = "style_calibration_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # new_writing | drift_detected | correction_threshold | manual
    trigger_reason = Column(String(50), nullable=False)
    corrections_used = Column(Integer, default=0)
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    delta_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "trigger_reason": self.trigger_reason,
            "corrections_used": self.corrections_used,
            "before": json.loads(self.before_json) if self.before_json else {},
            "after": json.loads(self.after_json) if self.after_json else {},
            "delta_score": self.delta_score,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class BeliefTimeline(Base):
    """Journal immuable de chaque changement d'état d'une croyance.

    C'est l'historique de l'évolution de la pensée de l'utilisateur.
    Table append-only : jamais de mise à jour ni de suppression de lignes.
    """

    __tablename__ = "belief_timeline"

    id = Column(Integer, primary_key=True, autoincrement=True)
    belief_id = Column(
        Integer, ForeignKey("belief_store.id"), nullable=True, index=True,
    )
    topic = Column(String(200), nullable=False)
    old_position = Column(Text, nullable=True)
    new_position = Column(Text, nullable=True)
    # user_correction | conflict_resolved | thumbs_feedback | explicit_update
    change_reason = Column(String(30), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "belief_id": self.belief_id,
            "topic": self.topic,
            "old_position": self.old_position,
            "new_position": self.new_position,
            "change_reason": self.change_reason,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class IdentityFidelityLog(Base):
    """Score de fidélité vocale par réponse en mode identité.

    Mesure à quel point une réponse générée correspond au style de l'utilisateur.
    """

    __tablename__ = "identity_fidelity_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    interaction_id = Column(
        Integer, ForeignKey("interactions.id"), nullable=True, index=True,
    )
    sentence_len_delta = Column(Float, default=0.0)
    formality_delta = Column(Float, default=0.0)
    first_person_delta = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "interaction_id": self.interaction_id,
            "sentence_len_delta": self.sentence_len_delta,
            "formality_delta": self.formality_delta,
            "first_person_delta": self.first_person_delta,
            "overall_score": self.overall_score,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
