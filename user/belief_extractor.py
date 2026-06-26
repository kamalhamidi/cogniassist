"""
user/belief_extractor.py — Extraction de croyances depuis les écrits personnels.

Utilise le LLM local (Mistral via ChatOllama) pour extraire des triplets
(topic, position, confidence) depuis chaque chunk d'écriture personnelle,
détecte les conflits sémantiques entre positions, et persiste le tout
dans la table `belief_store`.

Robustesse : tout appel LLM est encapsulé dans try/except. En cas d'échec
(JSON malformé, Ollama indisponible), on journalise et on retourne une liste
vide — jamais de crash. Aucune relance au-delà d'une fois par chunk.
"""

import json
import logging
import math
import re
from datetime import date, datetime

from langchain_core.messages import HumanMessage

from config import settings
from user.db import get_session
from user.identity_models import BeliefStore

logger = logging.getLogger("cogniassist.user")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs. Retourne 0.0 si invalide."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class BeliefExtractor:
    """
    Extrait des croyances (topic, position, confidence) des écrits personnels
    via le LLM local, avec détection de conflits et persistance.
    """

    EXTRACTION_PROMPT = """Analyse ce texte et extrais les opinions, croyances et positions de son auteur.

Texte: {chunk_text}

Retourne UNIQUEMENT un JSON valide, sans markdown, sans explication:
{{
  "beliefs": [
    {{
      "topic": "sujet court et précis (max 8 mots)",
      "position": "position exprimée en 1-2 phrases claires",
      "confidence": "high|medium|low"
    }}
  ]
}}

Règles strictes:
- N'extrais QUE les positions EXPLICITEMENT exprimées dans le texte
- confidence=high si affirmé sans réserve, low si nuancé ou conditionnel  
- Si aucune opinion claire n'est présente, retourne {{"beliefs": []}}
- Maximum 3 beliefs par chunk
- Le topic doit être spécifique, pas générique ("apprentissage par renforcement" pas "IA")"""

    def __init__(self, llm, embedder, session=None) -> None:
        """
        Args:
            llm: Instance ChatOllama existante (depuis RAGPipeline).
            embedder: Instance EmbeddingManager.
            session: Session SQLAlchemy (optionnelle — créée sinon).
        """
        self.llm = llm
        self.embedder = embedder
        self.session = session or get_session()

    # ─────────────────────────────────────────────────────────────────
    # Extraction
    # ─────────────────────────────────────────────────────────────────

    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        date_written: date | None = None,
    ) -> list[dict]:
        """
        Exécute l'extraction LLM sur un chunk. Sauvegarde et retourne les croyances.

        Args:
            chunk_text: Texte du chunk.
            chunk_id: Identifiant du chunk source.
            date_written: Date approximative d'écriture (optionnelle).

        Returns:
            Liste de dicts de croyances sauvegardées (vide en cas d'échec).
        """
        if not chunk_text or not chunk_text.strip():
            return []

        # 1. Appel LLM (une seule tentative, jamais de crash)
        try:
            prompt = self.EXTRACTION_PROMPT.format(chunk_text=chunk_text[:2000])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("Extraction LLM échouée (chunk %s) : %s", chunk_id, e)
            return []

        # 2. Parser le JSON (robuste)
        beliefs = self._parse_beliefs(raw)
        if not beliefs:
            return []

        # 3. Pour chaque croyance : embedding, conflit, sauvegarde
        saved: list[dict] = []
        for b in beliefs[:3]:
            topic = (b.get("topic") or "").strip()
            position = (b.get("position") or "").strip()
            confidence = (b.get("confidence") or "medium").strip().lower()
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            if not topic or not position:
                continue

            try:
                embedding = self.embedder.embed_query(f"{topic}. {position}")
            except Exception as e:
                logger.warning("Embedding croyance échoué : %s", e)
                embedding = []

            conflict_id = self._check_conflict(topic, position, embedding)
            self._save_belief(
                topic=topic,
                position=position,
                confidence=confidence,
                chunk_id=chunk_id,
                date_written=date_written,
                conflict_id=conflict_id,
                embedding=embedding,
            )
            saved.append({
                "topic": topic,
                "position": position,
                "confidence": confidence,
            })

        return saved

    def extract_from_chunks(
        self,
        chunks: list[dict],
        date_written: date | None = None,
    ) -> int:
        """
        Extraction par lot. Affiche une barre de progression Streamlit.

        Args:
            chunks: Liste de dicts {text, chunk_id}.
            date_written: Date approximative commune (optionnelle).

        Returns:
            Nombre total de croyances extraites.
        """
        if not chunks:
            return 0

        # Barre de progression Streamlit (import paresseux, optionnel)
        progress = None
        try:
            import streamlit as st
            progress = st.progress(0.0, text="Extraction des croyances…")
        except Exception:
            progress = None

        total = len(chunks)
        count = 0
        for i, ch in enumerate(chunks):
            text = ch.get("text", "")
            chunk_id = ch.get("chunk_id", f"pw_{i}")
            beliefs = self.extract_from_chunk(text, chunk_id, date_written)
            count += len(beliefs)

            if progress is not None:
                try:
                    progress.progress(
                        (i + 1) / total,
                        text=f"Extraction des croyances… ({i + 1}/{total})",
                    )
                except Exception:
                    pass

        if progress is not None:
            try:
                progress.empty()
            except Exception:
                pass

        logger.info("%d croyance(s) extraite(s) depuis %d chunk(s).", count, total)
        return count

    # ─────────────────────────────────────────────────────────────────
    # Conflits
    # ─────────────────────────────────────────────────────────────────

    def _check_conflict(
        self, topic: str, new_position: str, new_embedding: list[float] | None = None,
    ) -> int | None:
        """
        Vérifie s'il existe déjà une croyance sur le même sujet avec une
        position potentiellement contradictoire.

        Args:
            topic: Sujet de la nouvelle croyance.
            new_position: Position de la nouvelle croyance.
            new_embedding: Embedding pré-calculé (optionnel).

        Returns:
            ID de la croyance existante en conflit, ou None.
        """
        try:
            if new_embedding is None:
                new_embedding = self.embedder.embed_query(f"{topic}. {new_position}")
        except Exception:
            return None

        if not new_embedding:
            return None

        try:
            existing = (
                self.session.query(BeliefStore)
                .filter(BeliefStore.status.in_(["active", "user_confirmed"]))
                .all()
            )
        except Exception:
            return None

        threshold = settings.BELIEF_CONFLICT_THRESHOLD
        for belief in existing:
            emb = belief.get_embedding()
            sim = _cosine_similarity(new_embedding, emb)
            # Même sujet (haute similarité) mais position différente → conflit potentiel
            if sim >= threshold:
                if not self._positions_equivalent(belief.position, new_position):
                    return belief.id
        return None

    @staticmethod
    def _positions_equivalent(pos_a: str, pos_b: str) -> bool:
        """Heuristique simple : positions quasi identiques → pas un conflit."""
        a = re.sub(r"\s+", " ", (pos_a or "").lower()).strip()
        b = re.sub(r"\s+", " ", (pos_b or "").lower()).strip()
        if not a or not b:
            return False
        if a == b:
            return True
        # Chevauchement de mots élevé → considérées équivalentes
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return False
        overlap = len(wa & wb) / len(wa | wb)
        return overlap > 0.7

    def _save_belief(
        self,
        topic: str,
        position: str,
        confidence: str,
        chunk_id: str,
        date_written: date | None,
        conflict_id: int | None,
        embedding: list[float] | None = None,
    ) -> None:
        """
        Sauvegarde dans `belief_store`. Si conflict_id : marque les deux
        croyances comme 'conflicted'.
        """
        try:
            status = "conflicted" if conflict_id is not None else "active"

            belief = BeliefStore(
                topic=topic,
                position=position,
                confidence=confidence,
                source_chunk_id=chunk_id,
                date_written=date_written,
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            if embedding:
                belief.set_embedding(embedding)

            self.session.add(belief)

            # Marquer la croyance existante en conflit
            if conflict_id is not None:
                existing = (
                    self.session.query(BeliefStore)
                    .filter_by(id=conflict_id)
                    .first()
                )
                if existing:
                    existing.status = "conflicted"
                    existing.updated_at = datetime.utcnow()

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur sauvegarde croyance : %s", e)

    # ─────────────────────────────────────────────────────────────────
    # Lecture / récupération
    # ─────────────────────────────────────────────────────────────────

    def get_beliefs_for_topic(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Au moment d'une requête : trouve les croyances les plus pertinentes.

        Args:
            query: Requête de l'utilisateur.
            top_k: Nombre de croyances à retourner.

        Returns:
            Liste de dicts {topic, position, confidence, date_written}.
        """
        try:
            query_emb = self.embedder.embed_query(query)
        except Exception as e:
            logger.warning("Embedding requête (beliefs) échoué : %s", e)
            return []

        if not query_emb:
            return []

        try:
            beliefs = (
                self.session.query(BeliefStore)
                .filter(BeliefStore.status.in_(["active", "user_confirmed"]))
                .all()
            )
        except Exception:
            return []

        scored: list[tuple[float, BeliefStore]] = []
        for b in beliefs:
            sim = _cosine_similarity(query_emb, b.get_embedding())
            scored.append((sim, b))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict] = []
        for sim, b in scored[:top_k]:
            if sim <= 0.0:
                continue
            results.append({
                "topic": b.topic,
                "position": b.position,
                "confidence": b.confidence,
                "date_written": b.date_written.isoformat() if b.date_written else None,
            })
        return results

    def get_all_beliefs(self) -> list[dict]:
        """Retourne toutes les croyances triées par updated_at DESC (pour l'UI)."""
        try:
            beliefs = (
                self.session.query(BeliefStore)
                .order_by(BeliefStore.updated_at.desc())
                .all()
            )
            return [b.to_dict() for b in beliefs]
        except Exception as e:
            logger.error("Erreur lecture croyances : %s", e)
            return []

    def resolve_conflict(self, belief_id_keep: int, belief_id_drop: int) -> None:
        """L'utilisateur choisit la croyance courante. Met à jour les statuts."""
        try:
            keep = (
                self.session.query(BeliefStore).filter_by(id=belief_id_keep).first()
            )
            drop = (
                self.session.query(BeliefStore).filter_by(id=belief_id_drop).first()
            )
            if keep:
                keep.status = "user_confirmed"
                keep.updated_at = datetime.utcnow()
            if drop:
                drop.status = "superseded"
                drop.updated_at = datetime.utcnow()
            self.session.commit()
            logger.info(
                "Conflit résolu : #%s conservée, #%s remplacée.",
                belief_id_keep, belief_id_drop,
            )
        except Exception as e:
            self.session.rollback()
            logger.error("Erreur résolution conflit : %s", e)

    # ─────────────────────────────────────────────────────────────────
    # Parsing JSON robuste
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_beliefs(raw: str) -> list[dict]:
        """
        Parse la réponse LLM en liste de croyances. Robuste aux réponses
        entourées de markdown ou de texte parasite. Retourne [] si échec.
        """
        if not raw:
            return []

        text = raw.strip()

        # Retirer les clôtures markdown ```json ... ```
        text = re.sub(r"^```(?:json)?", "", text.strip())
        text = re.sub(r"```$", "", text.strip()).strip()

        # Tentative directe
        candidates: list[str] = [text]

        # Extraire le premier objet JSON {...} de la chaîne
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for cand in candidates:
            try:
                data = json.loads(cand)
                beliefs = data.get("beliefs", []) if isinstance(data, dict) else []
                if isinstance(beliefs, list):
                    return [b for b in beliefs if isinstance(b, dict)]
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        logger.debug("Impossible de parser la réponse LLM en JSON croyances.")
        return []
