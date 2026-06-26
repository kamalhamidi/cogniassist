"""
user/style_analyzer.py — Analyse stylistique des écrits personnels.

Extrait une « empreinte de style » à partir des écrits personnels de
l'utilisateur (longueur de phrase, richesse du vocabulaire, formalité,
ton, etc.) et la compile en une instruction en langage naturel injectable
dans le prompt système.

Aucune dépendance NLP externe : uniquement la bibliothèque standard
(re, string, collections). Conçu pour être rapide et non bloquant.
"""

import logging
import re
import string
from collections import Counter
from datetime import datetime

from user.identity_models import StyleProfile

logger = logging.getLogger("cogniassist.user")


# ═══════════════════════════════════════════════════════════════════════
# Lexiques (FR + EN)
# ═══════════════════════════════════════════════════════════════════════

_FIRST_PERSON = frozenset([
    # Anglais
    "i", "my", "me", "myself", "mine", "i'm", "i've", "i'll", "i'd",
    # Français
    "je", "j'", "mon", "ma", "mes", "moi", "moi-même", "mien", "mienne",
])

_HEDGING_PATTERNS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bi think\b",
    r"\bi believe\b", r"\bi suppose\b", r"\bmight\b", r"\bcould\b",
    r"\bprobably\b", r"\bseems\b", r"\bappears\b",
    r"\bpeut-être\b", r"\bje pense\b", r"\bje crois\b", r"\bprobablement\b",
    r"\bil me semble\b", r"\bsans doute\b",
]

_INFORMAL_PATTERNS = [
    r"\b\w+'\w+\b",          # contractions (don't, can't, t'as, j'ai)
    r"\blol\b", r"\bhaha+\b", r"\bmdr\b", r"\bptdr\b",
    r"\bok\b", r"\byeah\b", r"\bouais\b", r"\bgenre\b", r"\bbref\b",
]

_FORMAL_PATTERNS = [
    r"\btherefore\b", r"\bfurthermore\b", r"\bmoreover\b", r"\bhowever\b",
    r"\bconsequently\b", r"\bnevertheless\b", r"\bthus\b", r"\bhence\b",
    r"\bainsi\b", r"\bnéanmoins\b", r"\btoutefois\b", r"\bpar conséquent\b",
    r"\ben outre\b", r"\bcependant\b", r"\bde plus\b",
    # marqueurs de subordination / structure
    r"\bwhich\b", r"\bwhereas\b", r"\bdont\b", r"\blequel\b", r"\bafin de\b",
]

_EXAMPLE_PATTERNS = [
    r"\bfor example\b", r"\bfor instance\b", r"\bsuch as\b", r"\be\.g\.\b",
    r"\blike when\b", r"\bpar exemple\b", r"\bcomme\b", r"\bnotamment\b",
]

_ANALYTICAL_PATTERNS = [
    r"\bbecause\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
    r"\bparce que\b", r"\bcar\b", r"\bdonc\b", r"\bainsi\b",
    r"\bfirstly\b", r"\bsecondly\b", r"\bpremièrement\b", r"\bdeuxièmement\b",
]

_DIPLOMATIC_PATTERNS = [
    r"\bhowever\b", r"\bon the other hand\b", r"\bthat said\b",
    r"\bcependant\b", r"\bd'un autre côté\b", r"\bceci dit\b", r"\bnuance\b",
]

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]"
)


class StyleAnalyzer:
    """
    Analyse les chunks d'écriture personnelle pour extraire une empreinte
    de style. Appelé après l'import d'écrits personnels.
    """

    def analyze(self, chunks: list[str]) -> dict:
        """
        Analyse une liste de chunks de texte (les écrits personnels).

        Args:
            chunks: Liste de textes bruts.

        Returns:
            Dictionnaire correspondant aux colonnes de `style_profile`.
        """
        texts = [c for c in (chunks or []) if c and c.strip()]
        if not texts:
            return self._default_metrics()

        metrics = {
            "avg_sentence_len": round(self._avg_sentence_len(texts), 2),
            "vocabulary_richness": round(self._vocabulary_richness(texts), 4),
            "formality_score": round(self._formality_score(texts), 3),
            "first_person_ratio": round(self._first_person_ratio(texts), 4),
            "hedging_ratio": round(self._hedging_ratio(texts), 4),
            "example_preference": self._example_preference(texts),
            "preferred_length": self._preferred_length(texts),
            "tone": self._detect_tone(texts),
            "source_word_count": self._total_words(texts),
        }
        metrics["style_prompt_fragment"] = self._compile_style_prompt(metrics)
        return metrics

    # ─────────────────────────────────────────────────────────────────
    # Métriques
    # ─────────────────────────────────────────────────────────────────

    def _avg_sentence_len(self, texts: list[str]) -> float:
        """Nombre moyen de mots par phrase."""
        sentences = self._split_sentences(" ".join(texts))
        if not sentences:
            return 0.0
        word_counts = [len(s.split()) for s in sentences if s.split()]
        if not word_counts:
            return 0.0
        return sum(word_counts) / len(word_counts)

    def _vocabulary_richness(self, texts: list[str]) -> float:
        """Type-token ratio : tokens uniques / tokens totaux."""
        tokens = self._tokenize(" ".join(texts))
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _formality_score(self, texts: list[str]) -> float:
        """Score de formalité entre 0 (très familier) et 1 (très formel)."""
        joined = " ".join(texts).lower()

        formal_count = sum(
            len(re.findall(p, joined)) for p in _FORMAL_PATTERNS
        )
        informal_count = sum(
            len(re.findall(p, joined)) for p in _INFORMAL_PATTERNS
        )
        informal_count += joined.count("!")
        informal_count += len(_EMOJI_RE.findall(" ".join(texts)))

        total = formal_count + informal_count
        if total == 0:
            return 0.5  # neutre par défaut
        return max(0.0, min(1.0, formal_count / total))

    def _first_person_ratio(self, texts: list[str]) -> float:
        """Fréquence des marqueurs de première personne sur le total de mots."""
        tokens = self._tokenize(" ".join(texts))
        if not tokens:
            return 0.0
        fp = sum(1 for t in tokens if t in _FIRST_PERSON)
        # gérer les élisions type j'/j'ai (le tokenizer garde l'apostrophe)
        joined = " ".join(texts).lower()
        fp += len(re.findall(r"\bj'", joined))
        return fp / len(tokens)

    def _hedging_ratio(self, texts: list[str]) -> float:
        """Fréquence des marqueurs d'atténuation sur le nombre de phrases."""
        sentences = self._split_sentences(" ".join(texts))
        n_sentences = max(1, len(sentences))
        joined = " ".join(texts).lower()
        hedges = sum(len(re.findall(p, joined)) for p in _HEDGING_PATTERNS)
        return hedges / n_sentences

    def _example_preference(self, texts: list[str]) -> str:
        """Détecte si l'utilisateur place les exemples avant ou après la théorie."""
        first_half_hits = 0
        second_half_hits = 0

        for text in texts:
            paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            half = max(1, len(paragraphs) // 2)
            first = " ".join(paragraphs[:half]).lower()
            second = " ".join(paragraphs[half:]).lower()
            for p in _EXAMPLE_PATTERNS:
                first_half_hits += len(re.findall(p, first))
                second_half_hits += len(re.findall(p, second))

        if first_half_hits == 0 and second_half_hits == 0:
            return "balanced"
        if first_half_hits > second_half_hits * 1.2:
            return "examples_first"
        if second_half_hits > first_half_hits * 1.2:
            return "theory_first"
        return "balanced"

    def _preferred_length(self, texts: list[str]) -> str:
        """Longueur préférée selon le nombre moyen de mots par paragraphe."""
        paragraphs: list[str] = []
        for text in texts:
            parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            paragraphs.extend(parts if parts else [text])

        if not paragraphs:
            return "medium"

        avg = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        if avg < 50:
            return "short"
        if avg <= 150:
            return "medium"
        return "long"

    def _detect_tone(self, texts: list[str]) -> str:
        """Détermine le ton dominant : direct, analytical, diplomatic, enthusiastic."""
        joined = " ".join(texts)
        lower = joined.lower()

        scores: Counter = Counter()

        # Enthusiastic : exclamations, emoji, superlatifs
        scores["enthusiastic"] += joined.count("!") * 2
        scores["enthusiastic"] += len(_EMOJI_RE.findall(joined)) * 2
        scores["enthusiastic"] += len(re.findall(
            r"\b(amazing|génial|incroyable|super|excellent|adore|love)\b", lower
        ))

        # Analytical : connecteurs logiques, énumération
        scores["analytical"] += sum(
            len(re.findall(p, lower)) for p in _ANALYTICAL_PATTERNS
        )

        # Diplomatic : nuances + atténuation
        scores["diplomatic"] += sum(
            len(re.findall(p, lower)) for p in _DIPLOMATIC_PATTERNS
        )
        scores["diplomatic"] += sum(
            len(re.findall(p, lower)) for p in _HEDGING_PATTERNS
        )

        # Direct : phrases courtes + peu d'atténuation + impératif
        avg_len = self._avg_sentence_len(texts)
        if avg_len and avg_len < 12:
            scores["direct"] += 3
        if self._hedging_ratio(texts) < 0.1:
            scores["direct"] += 2

        if not scores or all(v == 0 for v in scores.values()):
            return "analytical"
        return scores.most_common(1)[0][0]

    # ─────────────────────────────────────────────────────────────────
    # Compilation du prompt de style
    # ─────────────────────────────────────────────────────────────────

    def _compile_style_prompt(self, metrics: dict) -> str:
        """Convertit les métriques numériques en instruction en langage naturel."""
        parts: list[str] = []

        avg = metrics.get("avg_sentence_len", 0) or 0
        if avg and avg < 12:
            parts.append(f"Écris des phrases concises (environ {int(avg)} mots).")
        elif avg and avg > 22:
            parts.append(
                f"Écris des phrases développées et nuancées "
                f"(environ {int(avg)} mots)."
            )
        elif avg:
            parts.append(f"Écris des phrases de longueur moyenne (~{int(avg)} mots).")

        tone_map = {
            "direct": "Adopte un ton direct et affirmé.",
            "analytical": "Adopte un ton analytique et structuré.",
            "diplomatic": "Adopte un ton diplomate et nuancé.",
            "enthusiastic": "Adopte un ton enthousiaste et énergique.",
        }
        parts.append(tone_map.get(metrics.get("tone"), tone_map["analytical"]))

        ex_map = {
            "examples_first": "Commence par des exemples concrets avant la théorie.",
            "theory_first": "Pose d'abord le concept puis illustre avec des exemples.",
            "balanced": "Équilibre concepts et exemples concrets.",
        }
        parts.append(ex_map.get(metrics.get("example_preference"), ex_map["balanced"]))

        length_map = {
            "short": "Privilégie des réponses courtes et denses.",
            "medium": "Vise des réponses de longueur modérée.",
            "long": "Développe les réponses en profondeur.",
        }
        parts.append(length_map.get(metrics.get("preferred_length"), length_map["medium"]))

        if (metrics.get("first_person_ratio") or 0) > 0.04:
            parts.append("Utilise naturellement la première personne.")

        if (metrics.get("hedging_ratio") or 0) < 0.1:
            parts.append("Évite les formules vagues — exprime les positions clairement.")
        elif (metrics.get("hedging_ratio") or 0) > 0.4:
            parts.append("Nuance les affirmations comme le ferait l'auteur.")

        formality = metrics.get("formality_score", 0.5) or 0.5
        if formality > 0.65:
            parts.append("Maintiens un registre soutenu et professionnel.")
        elif formality < 0.35:
            parts.append("Garde un registre familier et accessible.")

        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────────────────────────

    def save_profile(self, metrics: dict, session) -> None:
        """Upsert dans `style_profile` (une seule ligne — le dernier profil)."""
        try:
            profile = session.query(StyleProfile).first()
            if profile is None:
                profile = StyleProfile()
                session.add(profile)

            profile.avg_sentence_len = metrics.get("avg_sentence_len", 0.0)
            profile.vocabulary_richness = metrics.get("vocabulary_richness", 0.0)
            profile.formality_score = metrics.get("formality_score", 0.0)
            profile.first_person_ratio = metrics.get("first_person_ratio", 0.0)
            profile.hedging_ratio = metrics.get("hedging_ratio", 0.0)
            profile.example_preference = metrics.get("example_preference", "balanced")
            profile.preferred_length = metrics.get("preferred_length", "medium")
            profile.tone = metrics.get("tone", "analytical")
            profile.style_prompt_fragment = metrics.get("style_prompt_fragment", "")
            profile.source_word_count = metrics.get("source_word_count", 0)
            profile.updated_at = datetime.utcnow()

            session.commit()
            logger.info("Profil de style sauvegardé (%d mots analysés).",
                        profile.source_word_count)
        except Exception as e:
            session.rollback()
            logger.error("Erreur sauvegarde profil de style : %s", e)

    def get_profile(self, session) -> dict | None:
        """Retourne le dernier profil de style en dict, ou None s'il n'existe pas."""
        try:
            profile = session.query(StyleProfile).first()
            return profile.to_dict() if profile else None
        except Exception as e:
            logger.error("Erreur lecture profil de style : %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Découpe en phrases sur '. ', '! ', '? ' (et fins de chaîne)."""
        # Normaliser les fins de phrase puis découper
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise par mots (minuscules), sans la ponctuation isolée."""
        text = text.lower()
        tokens = re.findall(r"[a-zà-ÿ0-9']+", text)
        # retirer les tokens purement ponctuation/apostrophe
        return [t for t in tokens if t.strip(string.punctuation + "'")]

    def _total_words(self, texts: list[str]) -> int:
        """Nombre total de mots analysés."""
        return sum(len(t.split()) for t in texts)

    @staticmethod
    def _default_metrics() -> dict:
        """Métriques par défaut quand aucun texte n'est fourni."""
        return {
            "avg_sentence_len": 0.0,
            "vocabulary_richness": 0.0,
            "formality_score": 0.5,
            "first_person_ratio": 0.0,
            "hedging_ratio": 0.0,
            "example_preference": "balanced",
            "preferred_length": "medium",
            "tone": "analytical",
            "source_word_count": 0,
            "style_prompt_fragment": (
                "Écris de façon claire et naturelle, avec un ton analytique équilibré."
            ),
        }
