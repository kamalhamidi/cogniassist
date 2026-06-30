"""
pages/chat.py — Page de chat intelligent.

Interface conversationnelle avec streaming, suggestions contextuelles,
feedback utilisateur et affichage des sources.
"""

from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).parent.parent / "assets"
ASSISTANT_AVATAR = str(_ASSETS / "cogniassist_icon.png")
HERO_ART = str(_ASSETS / "cogniassist_icon.png")


@st.cache_resource
def _get_pipeline():
    """Charge le pipeline RAG (cache persistant)."""
    from rag import get_pipeline
    return get_pipeline()


@st.cache_data(ttl=300)
def _get_suggestions(user_id: str) -> list[str]:
    """Récupère les questions suggérées (cache 5 min)."""
    from user import get_recommender
    return get_recommender(user_id).get_suggested_questions()


@st.cache_data(ttl=60)
def _get_docs(user_id: str) -> list[dict]:
    """Récupère la liste des documents (cache 1 min)."""
    from user import get_user_manager
    return get_user_manager(user_id).get_user_documents()


def _handle_thumb(interaction_id: int, feedback: int) -> None:
    """Route le feedback 👍/👎 via le pipeline (historique + boucle de rétroaction)."""
    try:
        pipeline = _get_pipeline()
        if feedback == 1:
            pipeline.on_thumbs_up(interaction_id)
            st.toast("Merci pour votre retour ! 👍")
        else:
            pipeline.on_thumbs_down(interaction_id)
            st.toast("Retour enregistré — j'en tiendrai compte 👎")
    except Exception as e:
        st.error(f"Erreur feedback : {e}")


def _fidelity_caption(score) -> None:
    """Affiche un indicateur discret de fidélité vocale sous une réponse."""
    if score is None:
        return
    try:
        score = float(score)
    except (TypeError, ValueError):
        return
    if score >= 0.8:
        st.caption(f"🎯 Très fidèle à votre voix · {score * 100:.0f}%")
    elif score >= 0.6:
        st.caption(f"📝 Assez fidèle · {score * 100:.0f}%")
    else:
        st.caption(f"⚠️ Moins fidèle — corrigez si besoin · {score * 100:.0f}%")


def show_chat_page() -> None:
    """Affiche la page de chat intelligent."""
    user_id = st.session_state.get("user_id", "default")

    # ═══ Layout : chat (7) + panneau latéral (3) ═══
    col_chat, col_suggestions = st.columns([7, 3], gap="large")

    # ─────────────────────────────────────────────
    # Colonne droite : panneau d'information
    # ─────────────────────────────────────────────
    with col_suggestions:
        from ui import section_title, suggestion_card

        # Indicateur de mode second cerveau (Layer 2 — Identité)
        try:
            _pipe = _get_pipeline()
            if (_pipe.identity_mode_enabled
                    and _pipe.identity_builder is not None
                    and _pipe.identity_builder.is_identity_mode_ready()):
                st.success("🧠 Mode second cerveau actif")
        except Exception:
            pass

        # ─── Carte : Questions suggérées ───
        with st.container(border=True):
            section_title("💡 Questions suggérées")
            try:
                suggestions = _get_suggestions(user_id)
                if suggestions:
                    for i, sug in enumerate(suggestions):
                        if st.button(f"{sug}  ›", key=f"sug_{i}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": sug})
                            st.rerun()
                else:
                    st.caption("Aucune suggestion pour le moment")
            except Exception:
                st.caption("Suggestions indisponibles")

        # ─── Carte : Documents disponibles ───
        with st.container(border=True):
            st.html(
                """
                <div style="display:flex;align-items:center;justify-content:space-between;
                            gap:8px;margin:4px 0 10px;">
                    <span style="display:flex;align-items:center;gap:9px;font-size:0.98rem;
                                 font-weight:700;color:var(--ca-ink);white-space:nowrap;">
                        <span style="width:4px;height:18px;border-radius:4px;
                              background:linear-gradient(180deg,var(--ca-primary),var(--ca-accent));"></span>
                        📊 Documents disponibles
                    </span>
                    <a href="?page=upload" target="_self" style="font-size:0.78rem;font-weight:700;
                       color:var(--ca-primary);text-decoration:none;white-space:nowrap;">Voir tout</a>
                </div>
                """
            )
            try:
                docs = _get_docs(user_id)
                if docs:
                    rows = "".join(
                        f"""<div class="ca-doc-row">
                                <span class="ca-doc-name">📄 {d['file_name']}</span>
                                <span class="ca-doc-meta">{d['chunk_count']} chunks</span>
                            </div>"""
                        for d in docs[:5]
                    )
                    st.html(f"<div>{rows}</div>")
                else:
                    st.caption("Aucun document importé")
                    if st.button("📁 Importer un document", key="goto_upload"):
                        st.session_state.current_page = "📁 Documents"
                        st.rerun()
            except Exception:
                st.caption("Erreur de chargement")

        # ─── Progressive Profiling (carte « Suggestion ») ───
        try:
            from user.progressive import ProgressiveProfilingEngine
            prog = ProgressiveProfilingEngine(user_id)
            suggestion = prog.check_for_prompts()
            if suggestion:
                suggestion_card("Suggestion", suggestion["message"])
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✨ Ajouter à mes intérêts", key="prog_accept", use_container_width=True):
                        prog.accept_prompt(suggestion["id"])
                        st.cache_data.clear()
                        st.rerun()
                with col_no:
                    if st.button("Non merci", key="prog_decline", use_container_width=True):
                        prog.decline_prompt(suggestion["id"])
                        st.rerun()
        except Exception:
            pass  # Progressive profiling est non-critique

    # ─────────────────────────────────────────────
    # Colonne gauche : chat principal
    # ─────────────────────────────────────────────
    with col_chat:
        from ui import page_header, success_banner
        page_header(
            "Chat intelligent",
            "Posez vos questions sur vos documents personnels",
            icon="💬",
            art_image=HERO_ART,
        )

        # Vérifier le pipeline
        pipeline = _get_pipeline()
        if not pipeline.is_ready:
            st.warning(
                "⚠️ Ollama n'est pas démarré. Lancez la commande suivante dans un terminal :"
            )
            st.code("ollama run mistral:7b", language="bash")
            st.stop()

        success_banner(
            "Ollama est démarré et prêt à répondre à vos questions 🚀"
        )

        # Bouton effacer (uniquement si une conversation existe)
        if st.session_state.messages:
            col_sp, col_clear = st.columns([7, 3])
            with col_clear:
                if st.button(
                    "🗑️ Effacer la conversation",
                    key="clear_chat",
                    use_container_width=True,
                ):
                    st.session_state.messages = []
                    pipeline.clear_memory()
                    st.rerun()
        else:
            # ─── Écran d'accueil quand aucune conversation ───
            from ui import empty_state
            empty_state(
                "💬",
                "Démarrez une conversation",
                "Posez une question sur vos documents personnels. CogniAssist "
                "retrouve les passages pertinents et répond dans votre contexte.",
            )
            try:
                welcome_sugs = _get_suggestions(user_id)
            except Exception:
                welcome_sugs = []
            if welcome_sugs:
                st.caption("✨ Pour commencer, essayez :")
                for i, sug in enumerate(welcome_sugs[:3]):
                    if st.button(
                        f"💡 {sug}",
                        key=f"welcome_sug_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.messages.append(
                            {"role": "user", "content": sug}
                        )
                        st.rerun()

        # Afficher l'historique
        for idx, message in enumerate(st.session_state.messages):
            avatar = "👤" if message["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                # Sources et feedback pour les messages assistant
                if message["role"] == "assistant":
                    # Indicateur de fidélité vocale (mode identité)
                    if message.get("identity_mode"):
                        _fidelity_caption(message.get("fidelity_score"))

                    sources = message.get("sources", [])
                    iid = message.get("interaction_id")

                    # ─── Barre d'actions (chips) ───
                    a1, a2, a3, _sp = st.columns([1.7, 1.1, 1.05, 4.15])
                    with a1:
                        if st.button(
                            f"📄 Sources ({len(sources)})", key=f"act_src_{idx}",
                        ):
                            flag = f"show_src_{idx}"
                            st.session_state[flag] = not st.session_state.get(flag, False)
                    with a2:
                        if st.button("📋 Copier", key=f"act_cp_{idx}"):
                            st.toast("Réponse copiée ✓")
                    with a3:
                        if st.button("👍 Utile", key=f"act_up_{idx}"):
                            if iid and iid > 0:
                                _handle_thumb(iid, 1)
                            else:
                                st.toast("Merci pour votre retour ! 👍")

                    if sources and st.session_state.get(f"show_src_{idx}", False):
                        with st.container(border=True):
                            st.caption("📚 Sources utilisées")
                            for s in sources:
                                if isinstance(s, dict):
                                    st.caption(f"📄 {s.get('file_name', 'inconnu')}")
                                else:
                                    st.caption(f"📄 {s}")

                    # ─── Transparence : pourquoi cette réponse ? (mode identité) ───
                    if message.get("identity_mode"):
                        with st.expander("🔍 Pourquoi cette réponse ?"):
                            beliefs_used = message.get("beliefs_used", [])
                            if beliefs_used:
                                st.markdown("**Croyances mobilisées :**")
                                conf_fr = {
                                    "high": "forte", "medium": "modérée",
                                    "low": "faible",
                                }
                                for b in beliefs_used:
                                    conf = conf_fr.get(b.get("confidence", "medium"), "modérée")
                                    date_w = b.get("date_written")
                                    meta = f"conviction {conf}"
                                    if date_w:
                                        meta += f" · écrit le {date_w}"
                                    st.caption(
                                        f"• **{b.get('topic', '')}** — "
                                        f"{b.get('position', '')}"
                                    )
                                    st.caption(
                                        f"<span style='font-size:0.75rem;color:#9690b0'>"
                                        f"&nbsp;&nbsp;&nbsp;{meta}</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption(
                                    "Aucune position spécifique mobilisée sur ce sujet."
                                )

                            frag = message.get("style_fragment")
                            if frag:
                                st.markdown("**Style appliqué :**")
                                st.caption(frag)

                            fs = message.get("fidelity_score")
                            if fs is not None:
                                st.markdown("**Fidélité vocale :**")
                                st.caption(
                                    f"{float(fs) * 100:.0f}% de correspondance "
                                    "avec votre profil de style."
                                )

                    # ─── Correction stylistique (Layer 2 — Identité) ───
                    prev = (
                        st.session_state.messages[idx - 1] if idx > 0 else None
                    )
                    query_text = (
                        prev["content"]
                        if prev and prev.get("role") == "user" else ""
                    )
                    with st.expander("✏️ Ce n'est pas ma façon de dire ça"):
                        corrected = st.text_area(
                            "Reformulez la réponse à votre manière :",
                            value=message["content"],
                            key=f"corr_{idx}",
                            height=150,
                        )
                        if st.button(
                            "💾 Enregistrer ma version", key=f"savecorr_{idx}",
                        ):
                            try:
                                res = pipeline.on_style_correction(
                                    message.get("interaction_id"),
                                    query_text, message["content"], corrected,
                                )
                                if res.get("recalibrated"):
                                    st.toast(
                                        "✨ Style recalibré à partir de vos corrections !"
                                    )
                                else:
                                    st.toast(
                                        "Merci ! Votre style a été pris en compte. ✍️"
                                    )
                            except Exception as e:
                                st.error(f"Erreur : {e}")

        # ─── Ma pensée a évolué (Layer 6 — changement d'avis explicite) ───
        with st.expander("💭 Ma pensée a évolué sur un sujet"):
            mc_topic = st.text_input(
                "Sujet concerné",
                key="mind_change_topic",
                placeholder="Ex : le télétravail, les frameworks JS…",
            )
            mc_position = st.text_area(
                "Ma nouvelle position",
                key="mind_change_position",
                placeholder="Je ne pense plus que… je pense désormais que…",
                height=100,
            )
            if st.button("Enregistrer ma nouvelle position", key="mind_change_submit"):
                if mc_topic.strip() and mc_position.strip():
                    try:
                        new_id = pipeline.on_explicit_mind_change(
                            mc_topic.strip(), mc_position.strip(),
                        )
                        if new_id and new_id > 0:
                            st.success(
                                "Votre nouvelle position a été enregistrée et "
                                "prendra effet dès la prochaine conversation."
                            )
                        else:
                            st.warning("Enregistrement impossible pour le moment.")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                else:
                    st.warning("Renseignez le sujet et votre nouvelle position.")

        # ─── Input utilisateur ───
        prompt = st.chat_input("Posez votre question sur vos documents...")

        if prompt:
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Générer la réponse en streaming
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                try:
                    full_text = st.write_stream(
                        pipeline.ask_stream(prompt, user_id=user_id)
                    )
                except Exception as e:
                    full_text = f"Erreur : {e}"
                    st.error(full_text)

            # Récupérer les métadonnées complètes
            assistant_msg = {
                "role": "assistant",
                "content": full_text,
                "sources": [],
                "interaction_id": None,
                "identity_mode": False,
                "fidelity_score": None,
                "beliefs_used": [],
                "style_fragment": None,
            }
            try:
                result = pipeline.ask(prompt, user_id=user_id)
                assistant_msg["sources"] = result.get("sources", [])
                assistant_msg["interaction_id"] = result.get("interaction_id")
                assistant_msg["identity_mode"] = result.get("identity_mode", False)
                assistant_msg["fidelity_score"] = result.get("fidelity_score")

                # Détails de transparence (mode identité uniquement)
                if result.get("identity_mode"):
                    try:
                        assistant_msg["beliefs_used"] = (
                            pipeline.belief_extractor.get_beliefs_for_topic(prompt)
                        )
                    except Exception:
                        assistant_msg["beliefs_used"] = []
                    try:
                        from user.db import get_session
                        prof = pipeline.style_analyzer.get_profile(get_session())
                        if prof:
                            assistant_msg["style_fragment"] = prof.get(
                                "style_prompt_fragment"
                            )
                    except Exception:
                        pass
            except Exception:
                pass

            st.session_state.messages.append(assistant_msg)
            st.rerun()


if __name__ == "__main__":
    st.session_state.current_page = "💬 Chat"
    st.switch_page("app.py")

