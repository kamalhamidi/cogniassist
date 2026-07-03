"""
tests/test_feedback_loop.py — Tests de la Layer 6 (boucle de rétroaction).

Couvre les six signaux du FeedbackEngine : correction stylistique,
confirmation/rejet de croyance, amplification des pouces, détection de
dérive, changement d'avis explicite et score de fidélité.

Utilise une base SQLite temporaire et des doubles (fakes) pour le LLM et
les embeddings — aucun appel réseau réel.
"""

import json

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Configure une base SQLite temporaire pour chaque test."""
    import user.db as db_module

    db_module._engine = None
    db_module._SessionFactory = None

    test_db = tmp_path / "test_feedback.db"
    monkeypatch.setattr("config.settings.SQLITE_DB_PATH", str(test_db))
    monkeypatch.setattr(
        "config.Settings.sqlite_db_path",
        property(lambda self: test_db),
    )

    # Importer les modèles pour que create_all() les enregistre
    import user.profile  # noqa: F401
    import user.history  # noqa: F401

    from user.db import init_db
    init_db()

    yield


# ─── Doubles (fakes) ──────────────────────────────────────────────────

class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Faux LLM qui renvoie une réponse prédéfinie."""

    def __init__(self, response: str = "Le ton est devenu plus direct et concis.") -> None:
        self.response = response

    def invoke(self, messages):
        return _Resp(self.response)


class FakeEmbedder:
    """Faux gestionnaire d'embeddings déterministe."""

    def __init__(self, vector=None) -> None:
        self._vector = vector or [1.0, 0.0, 0.0]

    def embed_query(self, text: str):
        return list(self._vector)

    def embed_documents(self, texts):
        return [list(self._vector) for _ in texts]


# ─── Constructeurs utilitaires ────────────────────────────────────────

def _make_engine(llm_response: str = "résumé du changement"):
    """Construit un FeedbackEngine avec des dépendances réelles + fakes."""
    from user.feedback_engine import FeedbackEngine
    from user.style_analyzer import StyleAnalyzer
    from user.belief_extractor import BeliefExtractor
    from user.knowledge_engine import KnowledgeProfileEngine
    from user.db import db_session

    embedder = FakeEmbedder()
    belief_extractor = BeliefExtractor(
        llm=FakeLLM(llm_response), embedder=embedder,
    )
    return FeedbackEngine(
        style_analyzer=StyleAnalyzer(),
        belief_extractor=belief_extractor,
        knowledge_engine=KnowledgeProfileEngine("default"),
        embedder=embedder,
    )


def _seed_style_profile(engine):
    """Crée un profil de style à partir d'un texte d'exemple."""
    text = (
        "Je pense que ce projet avance bien. Nous devons rester concentrés. "
        "Les résultats sont encourageants et clairs. Continuons sur cette voie."
    )
    metrics = engine.style_analyzer.analyze([text])
    with db_session() as session:
        engine.style_analyzer.save_profile(metrics, session)
    return metrics


def _make_belief(session, topic, position, confidence="high", status="active"):
    """Insère une croyance et retourne son id."""
    from user.identity_models import BeliefStore
    b = BeliefStore(
        topic=topic, position=position, confidence=confidence, status=status,
    )
    session.add(b)
    session.commit()
    return b.id


def _make_interaction(session, beliefs_used):
    """Insère une interaction référençant des croyances et retourne son id."""
    from user.history import Interaction
    inter = Interaction(
        user_id="default",
        question="Que penses-tu de ce sujet ?",
        answer="Voici ma position sur la question.",
        beliefs_used_json=json.dumps(beliefs_used),
    )
    session.add(inter)
    session.commit()
    session.refresh(inter)
    return inter.id


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 1 — Correction stylistique
# ═══════════════════════════════════════════════════════════════════════

def test_style_correction_saves_signal():
    """on_style_correction → feedback_signals + style_corrections peuplés."""
    from user.identity_models import FeedbackSignal, StyleCorrection
    from user.db import db_session

    engine = _make_engine()
    result = engine.on_style_correction(
        interaction_id=None,
        query="Explique le RAG",
        generated="Le RAG est une technique complexe et sophistiquée.",
        corrected="Le RAG, c'est simple : on cherche puis on génère.",
    )

    assert result["status"] == "saved"
    assert result["recalibrated"] is False  # 1 correction < seuil

    with db_session() as session:
        signals = (
            session.query(FeedbackSignal)
            .filter_by(signal_type="style_correction").all()
        )
        assert len(signals) == 1

        corrections = session.query(StyleCorrection).all()
        assert len(corrections) == 1
        assert corrections[0].corrected.startswith("Le RAG, c'est simple")


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 2 — Rejet / confirmation de croyance
# ═══════════════════════════════════════════════════════════════════════

def test_belief_rejection_archives_timeline():
    """Rejet d'une croyance → timeline peuplée + statut superseded."""
    from user.identity_models import BeliefStore, BeliefTimeline
    from user.db import db_session

    engine = _make_engine()
    with db_session() as session:
        bid = _make_belief(
            session, "télétravail", "Le télétravail est plus productif.",
        )

    engine.on_belief_rejected(bid)

    with db_session() as session:
        belief = session.query(BeliefStore).filter_by(id=bid).first()
        assert belief.status == "superseded"

        timeline = session.query(BeliefTimeline).filter_by(belief_id=bid).all()
        assert len(timeline) == 1
        assert timeline[0].change_reason == "user_correction"
        assert timeline[0].old_position == "Le télétravail est plus productif."


def test_belief_rejection_with_replacement():
    """Rejet + remplacement → nouvelle croyance user_confirmed créée."""
    from user.identity_models import BeliefStore
    from user.db import db_session

    engine = _make_engine()
    with db_session() as session:
        bid = _make_belief(
            session, "télétravail", "Le télétravail est plus productif.",
        )

    engine.on_belief_rejected(
        bid, replacement_position="Le présentiel favorise la collaboration.",
    )

    with db_session() as session:
        new = (
            session.query(BeliefStore)
            .filter_by(status="user_confirmed").first()
        )
        assert new is not None
        assert new.topic == "télétravail"
        assert new.position == "Le présentiel favorise la collaboration."
        assert new.confidence == "high"


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 3 — Amplification des pouces
# ═══════════════════════════════════════════════════════════════════════

def test_thumbs_down_decays_confidence():
    """3 👎 sur la même croyance → statut conflicted."""
    from user.identity_models import BeliefStore
    from user.db import db_session

    engine = _make_engine()
    with db_session() as session:
        bid = _make_belief(
            session, "IA générative", "C'est révolutionnaire.",
            confidence="high",
        )
        iid = _make_interaction(session, [bid])

    for _ in range(3):
        engine.on_thumbs_down(iid)

    with db_session() as session:
        belief = session.query(BeliefStore).filter_by(id=bid).first()
        assert belief.status == "conflicted"


def test_thumbs_up_boosts_low_confidence():
    """👍 sur une croyance low → passe à medium."""
    from user.identity_models import BeliefStore
    from user.db import db_session

    engine = _make_engine()
    with db_session() as session:
        bid = _make_belief(
            session, "Python", "Python est idéal pour le prototypage.",
            confidence="low",
        )
        iid = _make_interaction(session, [bid])

    engine.on_thumbs_up(iid)

    with db_session() as session:
        belief = session.query(BeliefStore).filter_by(id=bid).first()
        assert belief.confidence == "medium"


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 4 — Détection de dérive
# ═══════════════════════════════════════════════════════════════════════

def test_drift_detection_no_drift():
    """Aucune correction → score de dérive nul, pas de recalibration."""
    engine = _make_engine()
    _seed_style_profile(engine)

    result = engine.check_for_drift()

    assert result["drift_detected"] is False
    assert result["drift_score"] == 0.0
    assert result["recalibrated"] is False


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 6 — Score de fidélité
# ═══════════════════════════════════════════════════════════════════════

def test_fidelity_score_logged():
    """score_response_fidelity → entrée loggée + score entre 0 et 1."""
    from user.identity_models import IdentityFidelityLog
    from user.db import db_session

    engine = _make_engine()
    _seed_style_profile(engine)

    score = engine.score_response_fidelity(
        interaction_id=1,
        response_text=(
            "Je pense que la réponse est claire. Restons concentrés sur l'essentiel."
        ),
    )

    assert 0.0 <= score <= 1.0
    with db_session() as session:
        logs = session.query(IdentityFidelityLog).all()
        assert len(logs) == 1
        assert 0.0 <= logs[0].overall_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 5 — Changement d'avis explicite
# ═══════════════════════════════════════════════════════════════════════

def test_explicit_mind_change_creates_timeline():
    """Changement d'avis → ancienne archivée, nouvelle user_confirmed, timeline."""
    from user.identity_models import BeliefStore, BeliefTimeline
    from user.db import db_session

    engine = _make_engine()
    with db_session() as session:
        old_id = _make_belief(
            session, "frameworks JS", "React est le meilleur framework.",
            status="active",
        )

    new_id = engine.on_explicit_mind_change(
        topic="frameworks JS",
        new_position="Je préfère désormais Svelte pour sa simplicité.",
    )

    assert new_id > 0
    with db_session() as session:
        old = session.query(BeliefStore).filter_by(id=old_id).first()
        assert old.status == "superseded"

        new = session.query(BeliefStore).filter_by(id=new_id).first()
        assert new.status == "user_confirmed"
        assert new.confidence == "high"

        timeline = session.query(BeliefTimeline).all()
        assert len(timeline) >= 1
        assert any(t.change_reason == "explicit_update" for t in timeline)


# ═══════════════════════════════════════════════════════════════════════
# Recalibration
# ═══════════════════════════════════════════════════════════════════════

def test_recalibration_logs_before_after():
    """_recalibrate_style → log avec before_json et after_json peuplés."""
    from user.identity_models import StyleCorrection, StyleCalibrationLog
    from user.db import db_session

    engine = _make_engine()
    _seed_style_profile(engine)

    with db_session() as session:
        session.add(StyleCorrection(
            query="q", generated="texte généré verbeux et long",
            corrected="Court. Net. Direct.", processed=False,
        ))
        session.commit()

    engine._recalibrate_style("manual")

    with db_session() as session:
        logs = session.query(StyleCalibrationLog).all()
        assert len(logs) == 1
        assert logs[0].before_json not in (None, "", "{}")
        assert logs[0].after_json not in (None, "", "{}")
        assert logs[0].trigger_reason == "manual"


# ═══════════════════════════════════════════════════════════════════════
# Résumé d'apprentissage
# ═══════════════════════════════════════════════════════════════════════

def test_get_learning_summary_structure():
    """get_learning_summary → toutes les clés présentes, aucun crash si vide."""
    engine = _make_engine()
    summary = engine.get_learning_summary()

    expected_keys = {
        "total_corrections", "style_calibrations", "beliefs_confirmed",
        "beliefs_rejected", "avg_fidelity_score", "fidelity_trend",
        "last_calibration", "pending_conflicts",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["total_corrections"] == 0
    assert summary["pending_conflicts"] == 0
