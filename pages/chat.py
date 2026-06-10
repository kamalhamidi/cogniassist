"""
pages/chat.py — Page de chat intelligent.

Interface conversationnelle avec streaming, suggestions contextuelles,
feedback utilisateur et affichage des sources.
"""

import streamlit as st


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


def _save_feedback(interaction_id: int, feedback: int) -> None:
    """Enregistre le feedback utilisateur."""
    try:
        from user import get_interaction_history
        history = get_interaction_history(st.session_state.get("user_id", "default"))
        history.save_feedback(interaction_id, feedback)
        st.toast("Merci pour votre retour ! 👍" if feedback == 1 else "Retour enregistré 👎")
    except Exception as e:
        st.error(f"Erreur feedback : {e}")


def show_chat_page() -> None:
    """Affiche la page de chat intelligent."""
    user_id = st.session_state.get("user_id", "default")

    # ═══ Layout : chat (7) + suggestions (3) ═══
    col_chat, col_suggestions = st.columns([7, 3])

    # ─────────────────────────────────────────────
    # Colonne droite : suggestions
    # ─────────────────────────────────────────────
    with col_suggestions:
        st.subheader("💡 Questions suggérées")

        try:
            suggestions = _get_suggestions(user_id)
            for i, sug in enumerate(suggestions):
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": sug})
                    st.rerun()
        except Exception:
            st.caption("Suggestions indisponibles")

        st.divider()
        st.subheader("📋 Documents disponibles")

        try:
            docs = _get_docs(user_id)
            if docs:
                for doc in docs[:5]:
                    st.caption(f"📄 {doc['file_name']} ({doc['chunk_count']} chunks)")
            else:
                st.caption("Aucun document importé")
                if st.button("📁 Importer un document", key="goto_upload"):
                    st.session_state.current_page = "📁 Documents"
                    st.rerun()
        except Exception:
            st.caption("Erreur de chargement")

        # ─── Progressive Profiling ───
        try:
            from user.progressive import ProgressiveProfilingEngine
            prog = ProgressiveProfilingEngine(user_id)
            suggestion = prog.check_for_prompts()
            if suggestion:
                st.divider()
                st.subheader("🎯 Suggestion")
                st.info(suggestion["message"])
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Accepter", key="prog_accept", use_container_width=True):
                        prog.accept_prompt(suggestion["id"])
                        st.cache_data.clear()
                        st.rerun()
                with col_no:
                    if st.button("❌ Non merci", key="prog_decline", use_container_width=True):
                        prog.decline_prompt(suggestion["id"])
                        st.rerun()
        except Exception:
            pass  # Progressive profiling est non-critique

    # ─────────────────────────────────────────────
    # Colonne gauche : chat principal
    # ─────────────────────────────────────────────
    with col_chat:
        st.title("💬 Chat intelligent")
        st.caption("Posez vos questions sur vos documents personnels")

        # Vérifier le pipeline
        pipeline = _get_pipeline()
        if not pipeline.is_ready:
            st.warning(
                "⚠️ Ollama n'est pas démarré. Lancez la commande suivante dans un terminal :"
            )
            st.code("ollama run mistral:7b", language="bash")
            st.stop()

        # Bouton effacer
        if st.button("🗑️ Effacer la conversation", key="clear_chat"):
            st.session_state.messages = []
            pipeline.clear_memory()
            st.rerun()

        # Afficher l'historique
        for idx, message in enumerate(st.session_state.messages):
            avatar = "👤" if message["role"] == "user" else "🧠"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                # Sources et feedback pour les messages assistant
                if message["role"] == "assistant":
                    sources = message.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources utilisées"):
                            for s in sources:
                                if isinstance(s, dict):
                                    st.caption(f"📄 {s.get('file_name', 'inconnu')}")
                                else:
                                    st.caption(f"📄 {s}")

                    iid = message.get("interaction_id")
                    if iid and iid > 0:
                        c1, c2, c3 = st.columns([1, 1, 8])
                        with c1:
                            st.button("👍", key=f"up_{idx}_{iid}",
                                      on_click=_save_feedback, args=(iid, 1))
                        with c2:
                            st.button("👎", key=f"down_{idx}_{iid}",
                                      on_click=_save_feedback, args=(iid, -1))

        # ─── Input utilisateur ───
        prompt = st.chat_input("Posez votre question sur vos documents...")

        if prompt:
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Générer la réponse en streaming
            with st.chat_message("assistant", avatar="🧠"):
                try:
                    full_text = st.write_stream(
                        pipeline.ask_stream(prompt, user_id=user_id)
                    )
                except Exception as e:
                    full_text = f"Erreur : {e}"
                    st.error(full_text)

            # Récupérer les métadonnées complètes
            try:
                result = pipeline.ask(prompt, user_id=user_id)
                sources = result.get("sources", [])
                interaction_id = result.get("interaction_id")
            except Exception:
                sources = []
                interaction_id = None

            # Sauvegarder le message assistant
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "sources": sources,
                "interaction_id": interaction_id,
            })
            st.rerun()
