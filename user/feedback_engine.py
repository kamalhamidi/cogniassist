"""
user/feedback_engine.py — Layer 6 : la boucle de rétroaction.

Coordinateur central de l'apprentissage continu du second cerveau.
Chaque interaction de l'utilisateur (correction, pouce, réécriture,
édition de croyance) devient un signal qui affine le modèle d'identité.

Principes :
- Non bloquant : toutes les méthodes sont rapides et encapsulées dans
  try/except — un échec ne doit jamais faire planter l'UI.
- Source unique de vérité : tous les seuils proviennent de config.py.
- Historique immuable : belief_timeline et style_calibration_log sont
  append-only (jamais de UPDATE ni de DELETE sur ces tables).
"""

import json
import logging
import math
from datetime import datetime

from config import settings
from user.db import uses_db_session
from user.identity_models import (
    StyleProfile,
    StyleCorrection,
    BeliefStore,
    FeedbackSignal,
    StyleCalibrationLog,
    BeliefTimeline,
    IdentityFidelityLog,
)
from user.history import Interaction

logger = logging.getLogger("cogniassist.user")

# Ordre croissant des niveaux de confiance
_CONFIDENCE_ORDER = ["low", "medium", "high"]


class FeedbackEngine:
    """
    Moteur central de la Layer 6. Reçoit les signaux de rétroaction depuis
    l'UI et les traduit en mises à jour du modèle d'identité.

    Toutes les méthodes sont non bloquantes — elles s'exécutent rapidement
    et ne plantent jamais l'UI même en cas d'erreur interne.
    """

    def __init__(
        self,
        style_analyzer,
        belief_extractor,
        knowledge_engine,
        embedder,
    ) -> None:
        self.style_analyzer = style_analyzer
        self.belief_extractor = belief_extractor
        self.knowledge_engine = knowledge_engine
        self.embedder = embedder
        self.settings = settings

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 1 : Correction stylistique
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def on_style_correction(
        self, session, interaction_id, query: str, generated: str, corrected: str,
    ) -> dict:
        """
        Appelée quand l'utilisateur réécrit une réponse.

        1. Sauvegarde dans style_corrections.
        2. Extrait ce qui a changé stylistiquement (mini-appel LLM).
        3. Enregistre un feedback_signal.
        4. Vérifie le seuil de corrections → recalibration éventuelle.
        """
        result = {"status": "error", "recalibrated": False, "diff_summary": ""}
        try:
            diff_summary = self._extract_diff_summary(generated, corrected)

            rec = StyleCorrection(
                query=query or "",
                generated=generated or "",
                corrected=corrected or "",
                diff_summary=diff_summary or None,
                processed=False,
            )
            session.add(rec)
            session.commit()

            self._log_signal(
                signal_type="style_correction",
                interaction_id=interaction_id,
                context={
                    "query": (query or "")[:300],
                    "generated": (generated or "")[:300],
                    "corrected": (corrected or "")[:300],
                },
                delta={"diff_summary": diff_summary},
            )

            recalibrated = self._check_correction_threshold()

            result = {
                "status": "saved",
                "recalibrated": recalibrated,
                "diff_summary": diff_summary,
            }
        except Exception as e:
            session.rollback()
            logger.error("on_style_correction a échoué : %s", e)
        return result

    def _extract_diff_summary(self, generated: str, corrected: str) -> str:
        """Court appel LLM résumant ce qui a changé. Retourne "" en cas d'erreur."""
        if not corrected or not corrected.strip():
            return ""
        try:
            from langchain_core.messages import HumanMessage

            llm = getattr(self.belief_extractor, "llm", None)
            if llm is None:
                return ""
            prompt = (
                "En une seule phrase, qu'est-ce qui a changé stylistiquement "
                "entre ces deux textes ? Concentre-toi sur le ton, la longueur, "
                "la structure et la formalité. Réponds uniquement par la phrase.\n\n"
                f"Texte original :\n{generated[:1000]}\n\n"
                f"Texte corrigé :\n{corrected[:1000]}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            return (raw or "").strip()[:500]
        except Exception as e:
            logger.debug("_extract_diff_summary a échoué : %s", e)
            return ""

    @uses_db_session
    def _check_correction_threshold(self) -> bool:
        """Recalibre si assez de corrections non traitées se sont accumulées."""
        try:
            pending = (
                session.query(StyleCorrection)
                .filter_by(processed=False)
                .count()
            )
            if pending >= self.settings.STYLE_RECALIBRATION_THRESHOLD:
                self._recalibrate_style("correction_threshold")
                return True
        except Exception as e:
            logger.debug("_check_correction_threshold a échoué : %s", e)
        return False

    @uses_db_session
    def _recalibrate_style(self, trigger_reason: str) -> None:
        """
        Recalcule le profil de style à partir des textes CORRIGÉS (ce que
        l'utilisateur voulait réellement) et journalise le changement.
        """
        if self.style_analyzer is None:
            return
        try:
            before = self.style_analyzer.get_profile(session) or {}

            corrections = session.query(StyleCorrection).all()
            corrected_texts = [c.corrected for c in corrections if c.corrected]

            # On combine les écrits personnels (base stable) et les textes
            # corrigés pour ne pas effondrer le profil sur peu de corrections.
            base_texts: list[str] = []
            try:
                from vectorstore.store import VectorStore
                base_texts = [
                    ch["text"]
                    for ch in VectorStore().get_personal_writing_chunks()
                ]
            except Exception:
                base_texts = []

            all_texts = base_texts + corrected_texts
            if not all_texts:
                return

            metrics = self.style_analyzer.analyze(all_texts)
            after = dict(metrics)

            delta_score = self._profile_delta(before, after)

            # Sauvegarder le nouveau profil
            self.style_analyzer.save_profile(metrics, session)

            # Journaliser (append-only)
            log = StyleCalibrationLog(
                trigger_reason=trigger_reason,
                corrections_used=len(corrected_texts),
                before_json=json.dumps(self._numeric_metrics(before), ensure_ascii=False),
                after_json=json.dumps(self._numeric_metrics(after), ensure_ascii=False),
                delta_score=round(delta_score, 4),
            )
            session.add(log)

            # Marquer les corrections comme traitées
            for c in corrections:
                c.processed = True

            session.commit()
            logger.info(
                "Style recalibré (%s) — %d correction(s), delta=%.3f.",
                trigger_reason, len(corrected_texts), delta_score,
            )
        except Exception as e:
            session.rollback()
            logger.error("_recalibrate_style a échoué : %s", e)

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 2 : Confirmation / rejet de croyance
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def on_belief_confirmed(self, belief_id: int) -> None:
        """L'utilisateur confirme une croyance → user_confirmed + confiance haute."""
        try:
            belief = (
                session.query(BeliefStore).filter_by(id=belief_id).first()
            )
            if belief is None:
                return
            old_conf = belief.confidence
            belief.status = "user_confirmed"
            if belief.confidence in ("medium", "low"):
                belief.confidence = "high"
            belief.updated_at = datetime.utcnow()
            session.commit()

            self._log_signal(
                signal_type="belief_confirm",
                interaction_id=None,
                context={"belief_id": belief_id, "topic": belief.topic},
                delta={
                    "status": {"old": "—", "new": "user_confirmed"},
                    "confidence": {"old": old_conf, "new": belief.confidence},
                },
            )
        except Exception as e:
            session.rollback()
            logger.error("on_belief_confirmed a échoué : %s", e)

    @uses_db_session
    def on_belief_rejected(
        self, session, belief_id: int, replacement_position: str | None = None,
    ) -> None:
        """L'utilisateur rejette une croyance → superseded + timeline (+ remplacement)."""
        try:
            belief = (
                session.query(BeliefStore).filter_by(id=belief_id).first()
            )
            if belief is None:
                return

            topic = belief.topic
            old_position = belief.position
            belief.status = "superseded"
            belief.updated_at = datetime.utcnow()

            # Timeline immuable (append-only)
            session.add(BeliefTimeline(
                belief_id=belief_id,
                topic=topic,
                old_position=old_position,
                new_position=replacement_position,
                change_reason="user_correction",
            ))

            new_belief_id = None
            if replacement_position and replacement_position.strip():
                new_belief_id = self._create_belief(
                    topic=topic,
                    position=replacement_position.strip(),
                    confidence="high",
                    status="user_confirmed",
                )

            session.commit()

            self._log_signal(
                signal_type="belief_reject",
                interaction_id=None,
                context={"belief_id": belief_id, "topic": topic},
                delta={
                    "status": {"old": "—", "new": "superseded"},
                    "replacement_belief_id": new_belief_id,
                },
            )
        except Exception as e:
            session.rollback()
            logger.error("on_belief_rejected a échoué : %s", e)

    @uses_db_session
    def on_belief_updated(
        self, session, belief_id: int, new_position: str, new_confidence: str,
    ) -> None:
        """L'utilisateur édite manuellement une croyance dans l'UI."""
        try:
            belief = (
                session.query(BeliefStore).filter_by(id=belief_id).first()
            )
            if belief is None:
                return

            old_position = belief.position
            confidence = (new_confidence or belief.confidence or "medium").lower()
            if confidence not in _CONFIDENCE_ORDER:
                confidence = "medium"

            # Timeline immuable
            session.add(BeliefTimeline(
                belief_id=belief_id,
                topic=belief.topic,
                old_position=old_position,
                new_position=new_position,
                change_reason="explicit_update",
            ))

            belief.position = new_position
            belief.confidence = confidence
            belief.updated_at = datetime.utcnow()
            embedding = self._embed(f"{belief.topic}. {new_position}")
            if embedding:
                belief.set_embedding(embedding)

            session.commit()

            self._log_signal(
                signal_type="manual_edit",
                interaction_id=None,
                context={"belief_id": belief_id, "topic": belief.topic},
                delta={"position": {"old": old_position, "new": new_position}},
            )
        except Exception as e:
            session.rollback()
            logger.error("on_belief_updated a échoué : %s", e)

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 3 : Amplification des pouces 👍 / 👎
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def on_thumbs_up(self, interaction_id: int) -> None:
        """👍 → renforce les croyances et le style utilisés dans la réponse."""
        try:
            context = self._get_interaction_context(interaction_id)
            belief_ids = context.get("beliefs_used", [])

            if self.settings.THUMBS_CONFIDENCE_BOOST:
                for bid in belief_ids:
                    belief = (
                        session.query(BeliefStore).filter_by(id=bid).first()
                    )
                    if belief is not None:
                        belief.confidence = self._boost_confidence(belief.confidence)
                        belief.updated_at = datetime.utcnow()

            # Incrémenter le compteur de renforcement sur le profil de style
            profile = session.query(StyleProfile).first()
            if profile is not None:
                profile.reinforcement_count = (profile.reinforcement_count or 0) + 1

            session.commit()

            self._log_signal(
                signal_type="thumbs_up",
                interaction_id=interaction_id,
                context={"beliefs_used": belief_ids},
                delta={"reinforced_beliefs": belief_ids},
            )
        except Exception as e:
            session.rollback()
            logger.error("on_thumbs_up a échoué : %s", e)

    @uses_db_session
    def on_thumbs_down(self, interaction_id: int, reason: str | None = None) -> None:
        """👎 → fait décroître la confiance des croyances utilisées."""
        try:
            context = self._get_interaction_context(interaction_id)
            belief_ids = context.get("beliefs_used", [])

            self._log_signal(
                signal_type="thumbs_down",
                interaction_id=interaction_id,
                context={"beliefs_used": belief_ids, "reason": reason},
                delta={"decayed_beliefs": belief_ids},
            )

            for bid in belief_ids:
                belief = (
                    session.query(BeliefStore).filter_by(id=bid).first()
                )
                if belief is None:
                    continue
                belief.confidence = self._decay_confidence(belief.confidence)
                belief.updated_at = datetime.utcnow()

                # 3+ 👎 consécutifs sur la même croyance → conflicted
                downs = self._count_thumbs_down_for_belief(bid)
                if downs >= self.settings.BELIEF_CONFIDENCE_DECAY_STEPS:
                    belief.status = "conflicted"

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("on_thumbs_down a échoué : %s", e)

    @uses_db_session
    def _get_interaction_context(self, interaction_id: int) -> dict:
        """Récupère le contexte stocké d'une interaction passée. {} si absente."""
        try:
            if not interaction_id:
                return {}
            inter = (
                session.query(Interaction)
                .filter_by(id=interaction_id)
                .first()
            )
            if inter is None:
                return {}
            beliefs_used = []
            raw = getattr(inter, "beliefs_used_json", None)
            if raw:
                try:
                    beliefs_used = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    beliefs_used = []
            return {
                "query": inter.question,
                "response": inter.answer,
                "beliefs_used": beliefs_used if isinstance(beliefs_used, list) else [],
            }
        except Exception as e:
            logger.debug("_get_interaction_context a échoué : %s", e)
            return {}

    @uses_db_session
    def _count_thumbs_down_for_belief(self, belief_id: int) -> int:
        """Compte les signaux 👎 référençant une croyance donnée."""
        try:
            signals = (
                session.query(FeedbackSignal)
                .filter_by(signal_type="thumbs_down")
                .all()
            )
            count = 0
            for s in signals:
                ctx = s.get_context()
                if belief_id in (ctx.get("beliefs_used") or []):
                    count += 1
            return count
        except Exception:
            return 0

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 4 : Détection de dérive
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def check_for_drift(self) -> dict:
        """Compare les corrections récentes au profil de style courant."""
        result = {"drift_detected": False, "drift_score": 0.0, "recalibrated": False}
        try:
            window = self.settings.FIDELITY_SCORE_WINDOW
            corrections = (
                session.query(StyleCorrection)
                .order_by(StyleCorrection.created_at.desc())
                .limit(window)
                .all()
            )
            recent = [{"corrected": c.corrected} for c in corrections if c.corrected]
            if not recent:
                return result

            drift_score = self._compute_drift_score(recent)
            result["drift_score"] = round(drift_score, 4)

            if drift_score > self.settings.DRIFT_THRESHOLD:
                result["drift_detected"] = True
                self._recalibrate_style("drift_detected")
                result["recalibrated"] = True
        except Exception as e:
            logger.debug("check_for_drift a échoué : %s", e)
        return result

    @uses_db_session
    def _compute_drift_score(self, recent_corrections: list[dict]) -> float:
        """Écart moyen entre les textes corrigés et le profil de style courant."""
        if self.style_analyzer is None:
            return 0.0
        try:
            profile = self.style_analyzer.get_profile(session)
            if not profile:
                return 0.0
            texts = [
                c.get("corrected", "")
                for c in recent_corrections
                if c.get("corrected")
            ]
            if not texts:
                return 0.0
            corrected_metrics = self.style_analyzer.analyze(texts)
            return self._profile_delta(profile, corrected_metrics)
        except Exception as e:
            logger.debug("_compute_drift_score a échoué : %s", e)
            return 0.0

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 5 : « J'ai changé d'avis »
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def on_explicit_mind_change(self, topic: str, new_position: str) -> int:
        """L'utilisateur déclare explicitement une nouvelle position sur un sujet."""
        try:
            if not topic or not new_position:
                return -1

            match = self._find_belief_by_topic(topic)
            if match is not None:
                session.add(BeliefTimeline(
                    belief_id=match.id,
                    topic=match.topic,
                    old_position=match.position,
                    new_position=new_position,
                    change_reason="explicit_update",
                ))
                match.status = "superseded"
                match.updated_at = datetime.utcnow()
                topic_to_use = match.topic
            else:
                topic_to_use = topic

            new_belief_id = self._create_belief(
                topic=topic_to_use,
                position=new_position.strip(),
                confidence="high",
                status="user_confirmed",
            )

            # Si aucune croyance existante, tracer quand même l'apparition
            if match is None:
                session.add(BeliefTimeline(
                    belief_id=new_belief_id,
                    topic=topic_to_use,
                    old_position=None,
                    new_position=new_position,
                    change_reason="explicit_update",
                ))

            session.commit()

            self._log_signal(
                signal_type="belief_changed",
                interaction_id=None,
                context={"topic": topic_to_use},
                delta={
                    "old_belief_id": match.id if match else None,
                    "new_belief_id": new_belief_id,
                },
            )
            return new_belief_id
        except Exception as e:
            session.rollback()
            logger.error("on_explicit_mind_change a échoué : %s", e)
            return -1

    @uses_db_session
    def _find_belief_by_topic(self, topic: str):
        """Trouve une croyance active/confirmée correspondant au sujet (sémantique)."""
        try:
            beliefs = (
                session.query(BeliefStore)
                .filter(BeliefStore.status.in_(["active", "user_confirmed"]))
                .all()
            )
            if not beliefs:
                return None

            query_emb = self._embed(topic)
            threshold = self.settings.BELIEF_TOPIC_SIMILARITY_THRESHOLD
            best = None
            best_sim = 0.0
            for b in beliefs:
                # Correspondance textuelle directe d'abord
                if b.topic and topic.lower().strip() == b.topic.lower().strip():
                    return b
                if query_emb:
                    sim = _cosine_similarity(query_emb, b.get_embedding())
                    if sim > best_sim:
                        best_sim, best = sim, b
            if best is not None and best_sim >= threshold:
                return best
            return None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 6 : Score de fidélité d'identité
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def score_response_fidelity(self, interaction_id: int, response_text: str) -> float:
        """Mesure la fidélité d'une réponse au profil de style. Journalise le score."""
        if self.style_analyzer is None or not response_text:
            return 0.0
        try:
            profile = self.style_analyzer.get_profile(session)
            if not profile:
                return 0.0

            resp = self.style_analyzer.analyze([response_text])

            sl_delta = self._metric_delta(
                resp.get("avg_sentence_len"), profile.get("avg_sentence_len"),
            )
            form_delta = self._metric_delta(
                resp.get("formality_score"), profile.get("formality_score"),
            )
            fp_delta = self._metric_delta(
                resp.get("first_person_ratio"), profile.get("first_person_ratio"),
            )

            deltas = [sl_delta, form_delta, fp_delta]
            overall = 1.0 - (sum(deltas) / len(deltas))
            overall = max(0.0, min(1.0, overall))

            log = IdentityFidelityLog(
                interaction_id=interaction_id,
                sentence_len_delta=round(sl_delta, 4),
                formality_delta=round(form_delta, 4),
                first_person_delta=round(fp_delta, 4),
                overall_score=round(overall, 4),
            )
            session.add(log)
            session.commit()
            return overall
        except Exception as e:
            session.rollback()
            logger.error("score_response_fidelity a échoué : %s", e)
            return 0.0

    # ══════════════════════════════════════════════════════════════════
    # Résumés pour l'UI
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def get_learning_summary(self) -> dict:
        """Résumé de l'apprentissage pour le dashboard."""
        summary = {
            "total_corrections": 0,
            "style_calibrations": 0,
            "beliefs_confirmed": 0,
            "beliefs_rejected": 0,
            "avg_fidelity_score": 0.0,
            "fidelity_trend": "stable",
            "last_calibration": None,
            "pending_conflicts": 0,
        }
        try:
            summary["total_corrections"] = session.query(StyleCorrection).count()
            summary["style_calibrations"] = session.query(StyleCalibrationLog).count()
            summary["beliefs_confirmed"] = (
                session.query(BeliefStore)
                .filter_by(status="user_confirmed").count()
            )
            summary["beliefs_rejected"] = (
                session.query(BeliefStore)
                .filter_by(status="superseded").count()
            )
            summary["pending_conflicts"] = (
                session.query(BeliefStore)
                .filter_by(status="conflicted").count()
            )

            history = self.get_fidelity_history(self.settings.FIDELITY_SCORE_WINDOW)
            scores = [h["overall_score"] for h in history]
            if scores:
                summary["avg_fidelity_score"] = round(sum(scores) / len(scores), 3)
                summary["fidelity_trend"] = self._trend(scores)

            last_cal = (
                session.query(StyleCalibrationLog)
                .order_by(StyleCalibrationLog.created_at.desc())
                .first()
            )
            if last_cal is not None and last_cal.created_at:
                summary["last_calibration"] = last_cal.created_at.isoformat()
        except Exception as e:
            logger.debug("get_learning_summary a échoué : %s", e)
        return summary

    @uses_db_session
    def get_fidelity_history(self, last_n: int = 30) -> list[dict]:
        """Retourne les N derniers scores de fidélité (ordre chronologique)."""
        try:
            rows = (
                session.query(IdentityFidelityLog)
                .order_by(IdentityFidelityLog.created_at.desc())
                .limit(last_n)
                .all()
            )
            rows = list(reversed(rows))
            return [
                {
                    "overall_score": r.overall_score,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "interaction_id": r.interaction_id,
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("get_fidelity_history a échoué : %s", e)
            return []

    @uses_db_session
    def get_belief_timeline(self, topic: str | None = None) -> list[dict]:
        """Retourne l'historique d'évolution des croyances (filtrable par sujet)."""
        try:
            q = session.query(BeliefTimeline)
            if topic:
                q = q.filter(BeliefTimeline.topic == topic)
            rows = q.order_by(BeliefTimeline.created_at.desc()).all()
            return [r.to_dict() for r in rows]
        except Exception as e:
            logger.debug("get_belief_timeline a échoué : %s", e)
            return []

    # ══════════════════════════════════════════════════════════════════
    # Helpers internes
    # ══════════════════════════════════════════════════════════════════

    @uses_db_session
    def _log_signal(
        self, session, signal_type: str, interaction_id, context: dict, delta: dict,
    ) -> None:
        """Enregistre un feedback_signal. Best-effort, jamais bloquant."""
        try:
            sig = FeedbackSignal(
                interaction_id=interaction_id,
                signal_type=signal_type,
                context_json=json.dumps(context, ensure_ascii=False),
                delta_json=json.dumps(delta, ensure_ascii=False, default=str),
                processed=False,
            )
            session.add(sig)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.debug("_log_signal a échoué : %s", e)

    @uses_db_session
    def _create_belief(
        self, session, topic: str, position: str, confidence: str, status: str,
    ) -> int:
        """Crée une nouvelle croyance avec embedding. Retourne son id (-1 si échec)."""
        try:
            belief = BeliefStore(
                topic=topic,
                position=position,
                confidence=confidence if confidence in _CONFIDENCE_ORDER else "medium",
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            embedding = self._embed(f"{topic}. {position}")
            if embedding:
                belief.set_embedding(embedding)
            session.add(belief)
            session.flush()
            return belief.id
        except Exception as e:
            logger.error("_create_belief a échoué : %s", e)
            return -1

    def _embed(self, text: str) -> list[float]:
        """Embedding sûr — retourne [] en cas d'erreur."""
        try:
            if self.embedder is None:
                return []
            return self.embedder.embed_query(text)
        except Exception:
            return []

    @staticmethod
    def _boost_confidence(conf: str) -> str:
        """👍 : low→medium, medium reste medium, high reste high."""
        return "medium" if conf == "low" else (conf or "medium")

    @staticmethod
    def _decay_confidence(conf: str) -> str:
        """👎 : high→medium, medium→low, low reste low."""
        if conf == "high":
            return "medium"
        if conf == "medium":
            return "low"
        return "low"

    @staticmethod
    def _metric_delta(resp_val, prof_val) -> float:
        """Écart relatif normalisé entre une valeur de réponse et le profil."""
        try:
            if prof_val is None or resp_val is None:
                return 0.0
            if prof_val == 0:
                return 0.0 if resp_val == 0 else 1.0
            d = abs(float(resp_val) - float(prof_val)) / abs(float(prof_val))
            return max(0.0, min(1.0, d))
        except (TypeError, ValueError):
            return 0.0

    def _profile_delta(self, before: dict, after: dict) -> float:
        """Magnitude du changement entre deux profils (0=identique, 1=total)."""
        keys = [
            "avg_sentence_len", "formality_score", "first_person_ratio",
            "vocabulary_richness", "hedging_ratio",
        ]
        deltas = []
        for k in keys:
            b, a = before.get(k), after.get(k)
            if b is None or a is None:
                continue
            deltas.append(self._metric_delta(a, b))
        if not deltas:
            return 0.0
        return max(0.0, min(1.0, sum(deltas) / len(deltas)))

    @staticmethod
    def _numeric_metrics(metrics: dict) -> dict:
        """Extrait un instantané sérialisable du profil pour les logs."""
        keys = [
            "avg_sentence_len", "vocabulary_richness", "formality_score",
            "first_person_ratio", "hedging_ratio", "example_preference",
            "preferred_length", "tone", "source_word_count",
        ]
        return {k: metrics.get(k) for k in keys if k in metrics}

    @staticmethod
    def _trend(scores: list[float]) -> str:
        """Détermine la tendance : improving | stable | declining."""
        if len(scores) < 4:
            return "stable"
        half = len(scores) // 2
        first = sum(scores[:half]) / half
        second = sum(scores[half:]) / (len(scores) - half)
        diff = second - first
        if diff > 0.05:
            return "improving"
        if diff < -0.05:
            return "declining"
        return "stable"


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
