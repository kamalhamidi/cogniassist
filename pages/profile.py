"""
pages/profile.py — Page de gestion du profil utilisateur avec ACPE.

Affiche et permet de modifier le profil, les préférences,
prévisualise le contexte RAG personnalisé, et offre des contrôles
de transparence et de confidentialité ACPE.
"""

import json
import streamlit as st


def show_profile_page() -> None:
    """Affiche la page de profil utilisateur."""
    user_id = st.session_state.get("user_id", "default")

    from ui import page_header
    page_header(
        "Mon profil",
        "Personnalisez votre expérience CogniAssist",
        icon="👤",
    )

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

        # ACPE badges
        user_type = profile.get("user_type", "individual")
        role = profile.get("role", "")
        type_label = "👤 Individuel" if user_type == "individual" else "🏢 Entreprise"
        st.caption(f"Type : {type_label}")
        if role:
            st.caption(f"Rôle : {role}")

    st.divider()

    # ═══ Onglets ═══
    tab_prefs, tab_context, tab_privacy, tab_kmb, tab_identity = st.tabs([
        "⚙️ Préférences", "🎯 Contexte RAG", "🔒 Données & Confidentialité",
        "🧠 Know Me Better", "🧠 Mon identité",
    ])

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

            style_options = ["concise", "detailed", "step_by_step", "educational"]
            style_captions = ["Court", "Détaillé", "Étape par étape", "Pédagogique"]
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

            # Explications de transparence
            st.divider()
            st.subheader("💡 Pourquoi ces paramètres ?")

            reasons = []
            user_type = profile.get("user_type", "individual")
            level = profile.get("expertise_level", "intermediate")

            if level == "beginner":
                reasons.append(
                    f"**k={params['k']}** — Plus de chunks récupérés pour "
                    f"fournir un contexte riche adapté à votre niveau débutant."
                )
                reasons.append(
                    "**Température basse** — Réponses plus déterministes et fiables."
                )
            elif level == "expert":
                reasons.append(
                    f"**k={params['k']}** — Moins de chunks mais plus ciblés "
                    f"pour des réponses concises adaptées à votre expertise."
                )
            if user_type == "enterprise":
                reasons.append(
                    "**Mode entreprise** — Priorité aux citations exactes et "
                    "au registre professionnel."
                )

            for r in reasons:
                st.markdown(f"• {r}")

        except Exception as e:
            st.error(f"Erreur paramètres : {e}")

    # ─── Tab 3 : Données & Confidentialité ───
    with tab_privacy:
        st.subheader("🔒 Gestion des données personnelles")
        st.caption(
            "Toutes vos données sont stockées localement. "
            "Aucune information n'est envoyée à des services externes."
        )

        # ── Profil de connaissances ──
        st.markdown("---")
        st.markdown("#### 🧠 Profil de connaissances")

        try:
            from user.knowledge_engine import KnowledgeProfileEngine
            ke = KnowledgeProfileEngine(user_id)
            knowledge = ke.get_knowledge_profile()

            if knowledge:
                import pandas as pd
                df = pd.DataFrame(knowledge)
                st.dataframe(
                    df[["domain", "mastery_score", "confidence", "interaction_count"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Aucun profil de connaissances encore construit.")
        except Exception:
            st.caption("Profil de connaissances non disponible.")

        # ── Insights ──
        st.markdown("---")
        st.markdown("#### 💡 Insights détectés")

        try:
            from user.profile_evolution import ProfileEvolutionEngine
            evo = ProfileEvolutionEngine(user_id)
            insights = evo.generate_insights()
            if insights:
                for insight in insights:
                    st.markdown(f"• {insight}")

                # Explications de transparence
                top_domains = ke.get_top_domains(limit=3)
                if top_domains:
                    for domain in top_domains:
                        kp_data = next(
                            (k for k in knowledge if k["domain"] == domain), None,
                        )
                        if kp_data:
                            pct = kp_data["interaction_count"]
                            st.caption(
                                f"💬 Vous voyez \"{domain}\" car {pct} de vos "
                                f"interactions récentes concernent ce domaine."
                            )
            else:
                st.caption("Les insights apparaîtront après quelques interactions.")
        except Exception:
            st.caption("Insights non disponibles.")

        # ── Contrôles ──
        st.markdown("---")
        st.markdown("#### ⚙️ Contrôles")

        # Toggle apprentissage adaptatif
        adaptive = profile.get("adaptive_learning_enabled", True)
        new_adaptive = st.toggle(
            "Apprentissage adaptatif activé",
            value=adaptive,
            help="Désactivez pour arrêter l'analyse automatique de vos interactions.",
            key="toggle_adaptive",
        )
        if new_adaptive != adaptive:
            mgr.update_profile(adaptive_learning_enabled=new_adaptive)
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # ── Boutons d'action ──
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            if st.button("📥 Exporter mes données", key="btn_export"):
                try:
                    export = mgr.export_profile_data()
                    json_str = json.dumps(export, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Télécharger JSON",
                        data=json_str,
                        file_name=f"cogniassist_profile_{user_id}.json",
                        mime="application/json",
                        key="btn_download",
                    )
                except Exception as e:
                    st.error(f"Erreur export : {e}")

        with col_d2:
            if st.button("🔄 Réinitialiser le profil ACPE", key="btn_reset_acpe"):
                try:
                    mgr.reset_acpe_data()
                    st.success("Données ACPE réinitialisées")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        with col_d3:
            if st.button("🔁 Relancer l'onboarding", key="btn_redo_onboarding"):
                try:
                    from user.profile import UserProfile
                    from user.db import get_session
                    session = get_session()
                    up = session.query(UserProfile).filter_by(
                        user_id=user_id,
                    ).first()
                    if up:
                        up.onboarding_completed = False
                        session.commit()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        st.markdown("---")
        st.markdown("#### ⚠️ Zone dangereuse")

        col_z1, col_z2 = st.columns(2)

        with col_z1:
            if st.button("🗑️ Effacer l'historique", type="secondary", key="btn_clear_hist"):
                try:
                    from user import get_interaction_history
                    get_interaction_history(user_id).clear_history()
                    st.success("Historique effacé")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        with col_z2:
            if st.button("🔄 Réinitialiser tout le profil", type="secondary", key="btn_reset_profile"):
                try:
                    mgr.update_preferences(
                        expertise_level="intermediate",
                        response_style="detailed",
                        language="fr",
                        domain_focus=[],
                        goals="",
                    )
                    mgr.reset_acpe_data()
                    st.success("Profil réinitialisé aux valeurs par défaut")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        st.markdown("---")
        st.markdown("##### ⚙️ Réinitialisation complète du système")
        st.warning(
            "⚠️ **Attention :** Cette opération est totalement destructive et irréversible. "
            "Elle supprimera définitivement :\n"
            "- L'intégralité de votre profil et de vos préférences\n"
            "- L'historique complet de vos conversations\n"
            "- Tous les documents importés dans la base de connaissances\n"
            "- Tous les index de recherche (ChromaDB et BM25)"
        )

        confirm_reset = st.checkbox(
            "Je confirme vouloir réinitialiser complètement le système et supprimer toutes les données.",
            key="chk_confirm_factory_reset",
        )

        if st.button(
            "💥 Réinitialiser complètement CogniAssist",
            type="primary",
            disabled=not confirm_reset,
            key="btn_factory_reset",
            use_container_width=True,
        ):
            try:
                from user import reset_system
                reset_system()
                st.success("🎉 Le système a été entièrement réinitialisé !")
                
                # Vider le session_state de Streamlit
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Effacer tous les caches de Streamlit
                st.cache_data.clear()
                st.cache_resource.clear()
                
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la réinitialisation : {e}")

    # ─── Tab 4 : Know Me Better ───
    with tab_kmb:
        try:
            from pages.know_me_better import show_kmb_page
            show_kmb_page(is_onboarding=False)
        except Exception as e:
            st.error(f"Erreur d'affichage Know Me Better : {e}")

    # ─── Tab 5 : Mon identité (Layer 2 — Second cerveau) ───
    with tab_identity:
        _show_identity_tab(user_id)


def _show_identity_tab(user_id: str) -> None:
    """Affiche l'onglet identité : style, croyances, mode second cerveau."""
    from rag import get_pipeline

    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Pipeline indisponible : {e}")
        return

    # ── Section A : Style d'écriture ──
    st.subheader("✍️ Style d'écriture")
    try:
        from user.db import get_session
        session = get_session()
        style = pipeline.style_analyzer.get_profile(session)
    except Exception:
        style = None

    if style and style.get("source_word_count", 0) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mots/phrase", f"{style['avg_sentence_len']:.0f}")
        c2.metric("Formalité", f"{style['formality_score'] * 100:.0f}%")
        c3.metric("Ton", style["tone"].capitalize())
        c4, c5, c6 = st.columns(3)
        c4.metric("Longueur préférée", style["preferred_length"].capitalize())
        c5.metric("Richesse lexicale", f"{style['vocabulary_richness'] * 100:.0f}%")
        c6.metric("Mots analysés", style["source_word_count"])

        if style.get("style_prompt_fragment"):
            st.info(f"🗣️ **Votre voix :** {style['style_prompt_fragment']}")
    else:
        st.info(
            "Aucun profil de style calculé. Importez vos écrits personnels "
            "depuis la page **Documents → Alimenter mon identité**."
        )

    st.divider()

    # ── Section B : Croyances extraites ──
    st.subheader("Ce que je crois (extrait de mes écrits)")
    try:
        beliefs = pipeline.belief_extractor.get_all_beliefs()
    except Exception:
        beliefs = []

    active = [b for b in beliefs if b["status"] in ("active", "user_confirmed")]
    conflicts = [b for b in beliefs if b["status"] == "conflicted"]

    st.caption(
        f"{len(active)} croyance(s) active(s), "
        f"{len(conflicts)} conflit(s) à résoudre."
    )

    conf_colors = {"high": "success", "medium": "primary", "low": "muted"}

    # Conflits d'abord (à résoudre)
    if conflicts:
        # Regrouper les conflits par paires (même topic le plus proche)
        st.markdown("#### ⚠️ Conflits à résoudre")
        # On présente chaque croyance conflictuelle avec un bouton de confirmation
        for b in conflicts:
            with st.container(border=True):
                st.warning(
                    f"**{b['topic']}** — {b['position']}\n\n"
                    f"_Confiance : {b['confidence']}_"
                )
                # Trouver les autres croyances conflictuelles sur un sujet proche
                others = [
                    o for o in conflicts
                    if o["id"] != b["id"]
                ]
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(
                        "C'est ma vision actuelle",
                        key=f"keep_{b['id']}",
                        use_container_width=True,
                    ):
                        # Marquer celle-ci confirmée et les autres conflits supersédées
                        for o in others:
                            pipeline.belief_extractor.resolve_conflict(b["id"], o["id"])
                        if not others:
                            pipeline.belief_extractor.resolve_conflict(b["id"], b["id"])
                        st.rerun()
                with col_b:
                    st.caption(f"Statut : {b['status']}")

    # Croyances actives
    if active:
        from ui import stat_badge
        st.markdown("#### Mes croyances actives")
        for b in active:
            badge = stat_badge(
                b["confidence"], conf_colors.get(b["confidence"], "primary"),
            )
            st.html(
                f"<div style='margin-bottom:8px'>"
                f"<b>{b['topic']}</b> {badge}<br>"
                f"<span style='color:#6B6880'>{b['position']}</span></div>"
            )
    elif not conflicts:
        st.caption("Aucune croyance extraite pour le moment.")

    st.divider()

    # ── Section C : Mode identité ──
    st.subheader("🧠 Mode second cerveau")
    ready = False
    try:
        ready = pipeline.identity_builder.is_identity_mode_ready()
    except Exception:
        ready = False

    if not ready:
        st.warning(
            "Importez d'abord vos écrits personnels pour activer ce mode."
        )

    enabled = st.toggle(
        "Activer le mode second cerveau",
        value=pipeline.identity_mode_enabled,
        disabled=not ready,
        key="toggle_identity_mode",
        help="CogniAssist répondra en imitant votre style et vos positions.",
    )
    if enabled != pipeline.identity_mode_enabled:
        pipeline.enable_identity_mode(enabled)
        st.toast(
            "🧠 Mode second cerveau activé." if enabled
            else "Mode second cerveau désactivé."
        )
        st.rerun()


if __name__ == "__main__":
    st.session_state.current_page = "👤 Profil"
    st.switch_page("app.py")

