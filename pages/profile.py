"""
pages/profile.py — Page de gestion du profil utilisateur.

Affiche et permet de modifier le profil, les préférences,
et prévisualise le contexte RAG personnalisé.
"""

import streamlit as st


def show_profile_page() -> None:
    """Affiche la page de profil utilisateur."""
    user_id = st.session_state.get("user_id", "default")

    st.title("👤 Mon profil")
    st.caption("Personnalisez votre expérience CogniAssist")

    # Charger le profil
    try:
        from user import get_user_manager
        mgr = get_user_manager(user_id)
        profile = mgr.get_profile()
    except Exception as e:
        st.error(f"Erreur chargement profil : {e}")
        return

    # ═══ En-tête profil ═══
    col_avatar, col_info = st.columns([2, 8])

    with col_avatar:
        st.markdown(
            f"<div style='font-size:80px;text-align:center'>"
            f"{profile['avatar']}</div>",
            unsafe_allow_html=True,
        )
        avatar_choice = st.selectbox(
            "Changer l'avatar",
            ["🧠", "👨‍🎓", "👩‍🎓", "🎓", "💡", "🔬", "📚", "🤖"],
            index=0,
            key="avatar_select",
        )
        if st.button("Mettre à jour l'avatar", key="btn_avatar"):
            try:
                mgr.update_profile(avatar=avatar_choice)
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    with col_info:
        st.markdown(f"### {profile['name']}")
        created = profile.get("created_at", "")[:10] if profile.get("created_at") else "—"
        st.caption(f"Membre depuis {created}")

        level_labels = {
            "beginner": "🟢 Débutant",
            "intermediate": "🟡 Intermédiaire",
            "expert": "🔴 Expert",
        }
        st.caption(f"Niveau : {level_labels.get(profile['expertise_level'], profile['expertise_level'])}")

    st.divider()

    # ═══ Onglets ═══
    tab_prefs, tab_context = st.tabs(["⚙️ Préférences", "🎯 Contexte RAG"])

    # ─── Tab 1 : Préférences ───
    with tab_prefs:
        with st.form("preferences_form"):
            name = st.text_input("Nom", value=profile["name"])

            expertise = st.select_slider(
                "Niveau d'expertise",
                options=["beginner", "intermediate", "expert"],
                value=profile["expertise_level"],
                format_func=lambda x: {"beginner": "Débutant", "intermediate": "Intermédiaire", "expert": "Expert"}[x],
            )

            style_options = ["concise", "detailed", "academic", "simple"]
            style_captions = ["Court", "Détaillé", "Académique", "Simple"]
            current_style = profile.get("response_style", "detailed")
            style_index = style_options.index(current_style) if current_style in style_options else 1

            response_style = st.radio(
                "Style de réponse",
                options=style_options,
                index=style_index,
                horizontal=True,
                captions=style_captions,
            )

            lang_options = ["fr", "en", "ar"]
            lang_labels = {"fr": "Français", "en": "Anglais", "ar": "Arabe"}
            current_lang = profile.get("language", "fr")
            lang_index = lang_options.index(current_lang) if current_lang in lang_options else 0

            language = st.selectbox(
                "Langue préférée",
                options=lang_options,
                index=lang_index,
                format_func=lambda x: lang_labels.get(x, x),
            )

            domains_input = st.text_input(
                "Domaines d'intérêt (séparés par des virgules)",
                value=", ".join(profile.get("domain_focus", [])),
            )

            goals = st.text_area(
                "Mes objectifs d'apprentissage",
                value=profile.get("goals", ""),
                height=120,
                placeholder="Ex : Maîtriser le RAG pour mon PFE, comprendre les LLMs...",
            )

            submitted = st.form_submit_button("💾 Sauvegarder", type="primary")
            if submitted:
                try:
                    mgr.update_profile(name=name)
                    mgr.update_preferences(
                        expertise_level=expertise,
                        response_style=response_style,
                        language=language,
                        domain_focus=[d.strip() for d in domains_input.split(",") if d.strip()],
                        goals=goals,
                    )
                    st.success("✅ Profil mis à jour avec succès !")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur sauvegarde : {e}")

    # ─── Tab 2 : Contexte RAG ───
    with tab_context:
        st.subheader("Contexte injecté dans vos prompts RAG")
        st.caption("Voici exactement ce que CogniAssist sait de vous lors de chaque question")

        try:
            context = mgr.get_personalization_context()
            st.code(context, language=None)
        except Exception as e:
            st.error(f"Erreur : {e}")

        st.divider()
        st.subheader("🔬 Paramètres RAG adaptés à votre profil")

        try:
            from user import get_recommender
            rec = get_recommender(user_id)
            params = rec.adapt_rag_parameters()

            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks récupérés (k)", params["k"])
            c2.metric("Température LLM", params["temperature"])
            c3.metric("Niveau", profile["expertise_level"])
        except Exception as e:
            st.error(f"Erreur paramètres : {e}")

        st.divider()
        st.subheader("⚠️ Zone dangereuse")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            if st.button("🗑️ Effacer l'historique", type="secondary", key="btn_clear_hist"):
                try:
                    from user import get_interaction_history
                    get_interaction_history(user_id).clear_history()
                    st.success("Historique effacé")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        with col_d2:
            if st.button("🔄 Réinitialiser le profil", type="secondary", key="btn_reset_profile"):
                try:
                    mgr.update_preferences(
                        expertise_level="intermediate",
                        response_style="detailed",
                        language="fr",
                        domain_focus=[],
                        goals="",
                    )
                    st.success("Profil réinitialisé aux valeurs par défaut")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
