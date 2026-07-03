"""
tests/test_identity.py — Tests de la couche d'identité (Layer 2 — Second cerveau).

Couvre l'analyseur de style, l'extracteur de croyances (robustesse JSON,
absence d'opinion), le constructeur de prompt d'identité (fallback et
disponibilité) et la résolution de conflits.

Utilise une base SQLite temporaire et des doubles (fakes) pour le LLM et
les embeddings — aucun appel réseau réel.
"""

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

    test_db = tmp_path / "test_identity.db"
    monkeypatch.setattr("config.settings.SQLITE_DB_PATH", str(test_db))
    monkeypatch.setattr(
        "config.Settings.sqlite_db_path",
        property(lambda self: test_db),
    )

    from user.db import init_db
    init_db()

    yield


# ─── Doubles (fakes) ──────────────────────────────────────────────────

class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Faux LLM qui renvoie une réponse prédéfinie."""

    def __init__(self, response: str = '{"beliefs": []}') -> None:
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


# ═══════════════════════════════════════════════════════════════════════
# Style Analyzer
# ═══════════════════════════════════════════════════════════════════════

def test_style_analyzer_basic():
    """5 phrases courtes et directes → preferred_length='short', tone='direct'."""
    from user.style_analyzer import StyleAnalyzer

    analyzer = StyleAnalyzer()
    texts = ["Go now.", "Do it.", "Stop waiting.", "Act fast.", "Be bold."]
    metrics = analyzer.analyze(texts)

    assert metrics["preferred_length"] == "short"
    assert metrics["tone"] == "direct"
    assert metrics["style_prompt_fragment"]


def test_style_analyzer_formality():
    """Texte académique formel → formality_score > 0.7."""
    from user.style_analyzer import StyleAnalyzer

    analyzer = StyleAnalyzer()
    formal = [
        "Therefore, the analysis demonstrates a significant correlation. "
        "Furthermore, the methodology, which was rigorously applied, yields "
        "consequential results. Moreover, the framework, whereas complex, "
        "remains coherent. Consequently, the conclusion follows logically. "
        "Nevertheless, further investigation is therefore warranted."
    ]
    metrics = analyzer.analyze(formal)

    assert metrics["formality_score"] > 0.7


# ═══════════════════════════════════════════════════════════════════════
# Belief Extractor
# ═══════════════════════════════════════════════════════════════════════

def test_belief_extractor_json_parse():
    """Un JSON malformé renvoyé par le LLM ne doit pas crasher → []."""
    from user.belief_extractor import BeliefExtractor

    extractor = BeliefExtractor(
        llm=FakeLLM(response="ceci n'est pas du JSON valide !!!"),
        embedder=FakeEmbedder(),
    )
    result = extractor.extract_from_chunk("un texte quelconque", "chunk_1")
    assert result == []


def test_belief_extractor_no_opinion_text():
    """Texte factuel neutre → 0 croyance extraite."""
    from user.belief_extractor import BeliefExtractor

    extractor = BeliefExtractor(
        llm=FakeLLM(response='{"beliefs": []}'),
        embedder=FakeEmbedder(),
    )
    result = extractor.extract_from_chunk(
        "L'eau bout à 100 degrés Celsius au niveau de la mer.", "chunk_2",
    )
    assert result == []
    assert extractor.get_all_beliefs() == []


def test_belief_extractor_saves_belief():
    """Une réponse LLM valide doit sauvegarder une croyance active."""
    from user.belief_extractor import BeliefExtractor

    response = (
        '{"beliefs": [{"topic": "apprentissage par renforcement", '
        '"position": "C\'est l\'avenir de l\'IA.", "confidence": "high"}]}'
    )
    extractor = BeliefExtractor(
        llm=FakeLLM(response=response),
        embedder=FakeEmbedder(),
    )
    saved = extractor.extract_from_chunk("texte avec opinion", "chunk_3")
    assert len(saved) == 1
    assert saved[0]["topic"] == "apprentissage par renforcement"

    all_beliefs = extractor.get_all_beliefs()
    assert len(all_beliefs) == 1
    assert all_beliefs[0]["status"] == "active"


# ═══════════════════════════════════════════════════════════════════════
# Identity Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def test_identity_prompt_builder_fallback():
    """Sans style_profile → prompt valide utilisant les valeurs par défaut."""
    from user.style_analyzer import StyleAnalyzer
    from user.belief_extractor import BeliefExtractor
    from user.identity_prompt_builder import IdentityPromptBuilder
    from user import get_user_manager
    builder = IdentityPromptBuilder(
        style_analyzer=StyleAnalyzer(),
        belief_extractor=BeliefExtractor(
            llm=FakeLLM(), embedder=FakeEmbedder(),
        ),
        profile_manager=get_user_manager("default"),
    )

    prompt = builder.build_system_prompt("Quelle est ta position sur l'IA ?")
    assert isinstance(prompt, str)
    assert "RÈGLES ABSOLUES" in prompt
    assert "STYLE D'ÉCRITURE" in prompt
    # Fallback : aucune position connue
    assert "Aucune position connue" in prompt


def test_identity_mode_ready_false():
    """Avec 0 croyance → is_identity_mode_ready() == False."""
    from user.style_analyzer import StyleAnalyzer
    from user.belief_extractor import BeliefExtractor
    from user.identity_prompt_builder import IdentityPromptBuilder
    from user import get_user_manager
    builder = IdentityPromptBuilder(
        style_analyzer=StyleAnalyzer(),
        belief_extractor=BeliefExtractor(
            llm=FakeLLM(), embedder=FakeEmbedder(),
        ),
        profile_manager=get_user_manager("default"),
    )
    assert builder.is_identity_mode_ready() is False


def test_identity_mode_ready_true():
    """Avec un style_profile + >= MIN croyances → is_identity_mode_ready() True."""
    from user.style_analyzer import StyleAnalyzer
    from user.belief_extractor import BeliefExtractor
    from user.identity_prompt_builder import IdentityPromptBuilder
    from user.identity_models import BeliefStore
    from user import get_user_manager
    from user.db import db_session
    from config import settings

    analyzer = StyleAnalyzer()

    with db_session() as session:
        metrics = analyzer.analyze(["Une phrase de test pour générer un profil."])
        analyzer.save_profile(metrics, session)

        for i in range(settings.MIN_BELIEFS_FOR_IDENTITY_MODE):
            b = BeliefStore(
                topic=f"sujet {i}", position=f"position {i}",
                confidence="high", status="active",
            )
            session.add(b)
        session.commit()

    builder = IdentityPromptBuilder(
        style_analyzer=analyzer,
        belief_extractor=BeliefExtractor(
            llm=FakeLLM(), embedder=FakeEmbedder(),
        ),
        profile_manager=get_user_manager("default"),
    )
    assert builder.is_identity_mode_ready() is True


# ═══════════════════════════════════════════════════════════════════════
# Conflict resolution
# ═══════════════════════════════════════════════════════════════════════

def test_conflict_resolution():
    """Deux croyances en conflit, on en résout une → statuts corrects."""
    from user.belief_extractor import BeliefExtractor
    from user.identity_models import BeliefStore
    from user.db import db_session

    with db_session() as session:
        b1 = BeliefStore(
            topic="télétravail", position="Le télétravail est plus productif.",
            confidence="high", status="conflicted",
        )
        b2 = BeliefStore(
            topic="télétravail", position="Le bureau est plus productif.",
            confidence="medium", status="conflicted",
        )
        session.add_all([b1, b2])
        session.commit()
        keep_id, drop_id = b1.id, b2.id

    extractor = BeliefExtractor(
        llm=FakeLLM(), embedder=FakeEmbedder(),
    )
    extractor.resolve_conflict(keep_id, drop_id)

    with db_session() as session:
        kept = session.query(BeliefStore).filter_by(id=keep_id).first()
        dropped = session.query(BeliefStore).filter_by(id=drop_id).first()

        assert kept.status == "user_confirmed"
        assert dropped.status == "superseded"
