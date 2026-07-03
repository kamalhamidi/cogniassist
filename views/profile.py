"""
pages/profile.py — Page de gestion du profil utilisateur avec ACPE.

Affiche et permet de modifier le profil, les préférences,
prévisualise le contexte RAG personnalisé, et offre des contrôles
de transparence et de confidentialité ACPE.
"""

import json
from pathlib import Path

import streamlit as st

_ICON_PATH = str(Path(__file__).parent.parent / "assets" / "cogniassist_icon.png")
_AVATAR_NAMES = {
    "🧠": "Cerveau", "👨‍🎓": "Étudiant", "👩‍🎓": "Étudiante", "🎓": "Diplômé",
    "💡": "Idée", "🔬": "Science", "📚": "Études", "🤖": "Robot",
}
_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
    "août", "septembre", "octobre", "novembre", "décembre",
]


def _format_member_date(created_at: str) -> str:
    """Formate une date ISO en « le D mois AAAA » (français)."""
    if not created_at:
        return "récemment"
    try:
        d = created_at[:10].split("-")
        return f"le {int(d[2])} {_MONTHS_FR[int(d[1]) - 1]} {d[0]}"
    except Exception:
        return created_at[:10]


def _render_tips_card() -> None:
    """Affiche la carte « Astuces » de la colonne droite des préférences."""
    from ui import img_data_uri
    uri = img_data_uri(_ICON_PATH)
    st.html(
        f"""
        <div class="ca-tips">
            <img class="ca-tips-art" src="{uri}" alt="" />
            <h4>💡 Astuces</h4>
            <div class="ca-tips-lead">Personnalisez votre profil pour obtenir des
                réponses plus pertinentes et adaptées à vos besoins.</div>
            <div class="ca-tip-row">
                <div class="ca-tip-ic" style="background:rgba(108,92,231,.14);">✏️</div>
                <div class="ca-tip-txt"><b>Un niveau d'expertise précis</b>
                    permet d'adapter la complexité.</div>
            </div>
            <div class="ca-tip-row">
                <div class="ca-tip-ic" style="background:rgba(22,163,74,.14);">🎯</div>
                <div class="ca-tip-txt"><b>Définissez vos objectifs</b>
                    pour des suggestions ciblées.</div>
            </div>
            <div class="ca-tip-row">
                <div class="ca-tip-ic" style="background:rgba(217,119,6,.16);">⭐</div>
                <div class="ca-tip-txt"><b>Vos domaines d'intérêt</b>
                    améliorent la qualité des réponses.</div>
            </div>
        </div>
        """
    )


def show_profile_page() -> None:
    """Affiche la page de profil utilisateur."""
    user_id = st.session_state.get("user_id", "default")

    # Charger le profil
    try:
        from user import get_user_manager
        mgr = get_user_manager(user_id)
        profile = mgr.get_profile()
    except Exception as e:
        st.error(f"Erreur chargement profil : {e}")
        return

    level_fr = {
        "beginner": "Débutant", "intermediate": "Intermédiaire", "expert": "Expert",
    }
    user_type = profile.get("user_type", "individual")
    type_label = "Individuel" if user_type == "individual" else "Entreprise"
    role = profile.get("role") or "—"
    member_since = _format_member_date(profile.get("created_at", ""))
    avatar_glyph = profile.get("avatar", "🧠")

    # ═══ En-tête profil (héro dégradé) ═══
    with st.container(key="profile_hero"):
        c_av, c_main, c_btn = st.columns([2.3, 6.2, 2.2], vertical_alignment="center")

        with c_av:
            st.html(
                f"""
                <div class="ca-pf-avatar">
                    <span style="font-size:62px;line-height:1;">{avatar_glyph}</span>
                    <span class="ca-pf-pencil">✏️</span>
                </div>
                """
            )
            avatar_choice = st.selectbox(
                "Changer l'avatar",
                list(_AVATAR_NAMES.keys()),
                index=0,
                key="pf_avatar_sel",
                format_func=lambda e: f"{e}  {_AVATAR_NAMES.get(e, '')}",
            )
            with st.container(key="pf_avatar_btn"):
                if st.button("⬆️ Mettre à jour l'avatar", key="btn_avatar",
                             use_container_width=True):
                    try:
                        mgr.update_profile(avatar=avatar_choice)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        with c_main:
            st.html(
                f"""
                <h1 class="ca-pf-name">{profile['name']}</h1>
                <div class="ca-pf-sub">Membre depuis {member_since}</div>
                <div class="ca-pf-badges">
                    <div class="ca-pf-badge">
                        <span class="ic">⭐</span>
                        <div><div class="lbl">Niveau</div>
                            <div class="val">{level_fr.get(profile['expertise_level'],
                                                            profile['expertise_level'])}</div></div>
                    </div>
                    <div class="ca-pf-badge">
                        <span class="ic">👤</span>
                        <div><div class="lbl">Type</div>
                            <div class="val">{type_label}</div></div>
                    </div>
                    <div class="ca-pf-badge">
                        <span class="ic">🛡️</span>
                        <div><div class="lbl">Rôle</div>
                            <div class="val">{role}</div></div>
                    </div>
                </div>
                """
            )

        with c_btn:
            with st.container(key="pf_edit_btn"):
                if st.button("✏️ Modifier le profil", key="btn_edit_profile",
                             use_container_width=True):
                    st.toast("Modifiez vos informations dans l'onglet Préférences ci-dessous ✏️")

    # ═══ Navigation par onglet (rendu paresseux — un seul panneau à la fois) ═══
    _TAB_OPTIONS = [
        "⚙️ Préférences", "🎯 Contexte RAG", "🔒 Données & Confidentialité",
        "🧠 Know Me Better", "🧠 Mon identité",
    ]
    active_tab = st.segmented_control(
        "Section profil",
        options=_TAB_OPTIONS,
        default=st.session_state.get("profile_tab", _TAB_OPTIONS[0]),
        label_visibility="collapsed",
        key="profile_tab_nav",
    )
    if active_tab:
        st.session_state.profile_tab = active_tab
    else:
        active_tab = st.session_state.get("profile_tab", _TAB_OPTIONS[0])

    # ─── Préférences ───
    if active_tab == "⚙️ Préférences":
        col_form, col_tips = st.columns([7, 3], gap="large")

        with col_form:
            with st.form("preferences_form"):
                name = st.text_input("Nom", value=profile["name"])

                expertise = st.select_slider(
                    "Niveau d'expertise",
                    options=["beginner", "intermediate", "expert"],
                    value=profile["expertise_level"],
                    format_func=lambda x: level_fr[x],
                )

                style_options = ["concise", "detailed", "step_by_step", "educational"]
                style_labels = {
                    "concise": "Concis", "detailed": "Détaillé",
                    "step_by_step": "Step by step", "educational": "Éducational",
                }
                style_captions = ["Court", "Détaillé", "Étape par étape", "Pédagogique"]
                current_style = profile.get("response_style", "detailed")
                style_index = style_options.index(current_style) if current_style in style_options else 1

                response_style = st.radio(
                    "Style de réponse",
                    options=style_options,
                    index=style_index,
                    horizontal=True,
                    captions=style_captions,
                    format_func=lambda x: style_labels.get(x, x),
                )

                lang_options = ["fr", "en", "ar"]
                lang_labels = {"fr": "Français", "en": "Anglais", "ar": "Arabe"}
                current_lang = profile.get("language", "fr")
                lang_index = lang_options.index(current_lang) if current_lang in lang_options else 0

                col_lang, col_dom = st.columns(2)
                with col_lang:
                    language = st.selectbox(
                        "Langue préférée",
                        options=lang_options,
                        index=lang_index,
                        format_func=lambda x: lang_labels.get(x, x),
                    )
                with col_dom:
                    domains_input = st.text_input(
                        "Domaines d'intérêt (séparés par des virgules)",
                        value=", ".join(profile.get("domain_focus", [])),
                    )

                goals = st.text_area(
                    "Mes objectifs d'apprentissage",
                    value=profile.get("goals", ""),
                    height=120,
                    max_chars=500,
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

        with col_tips:
            _render_tips_card()

    # ─── Contexte RAG ───
    elif active_tab == "🎯 Contexte RAG":
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

    # ─── Données & Confidentialité ───
    elif active_tab == "🔒 Données & Confidentialité":
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
                    from user.db import db_session
                    with db_session() as session:
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

    # ─── Know Me Better ───
    elif active_tab == "🧠 Know Me Better":
        try:
            from views.know_me_better import show_kmb_page
            show_kmb_page(is_onboarding=False)
        except Exception as e:
            st.error(f"Erreur d'affichage Know Me Better : {e}")

    # ─── Mon identité ───
    elif active_tab == "🧠 Mon identité":
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
        from user.db import db_session
        with db_session() as session:
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

    # Conflits d'abord (à résoudre) — carte claire à deux versions
    if conflicts:
        st.markdown("#### ⚠️ Conflits à résoudre")

        # Regrouper les croyances conflictuelles par sujet
        by_topic: dict[str, list] = {}
        for b in conflicts:
            by_topic.setdefault(b["topic"], []).append(b)

        for topic, group in by_topic.items():
            with st.container(border=True):
                st.markdown(f"**⚠️ Conflit détecté : {topic}**")
                # Trier par date pour présenter « Version A / Version B »
                group_sorted = sorted(group, key=lambda x: x.get("created_at", ""))
                for i, b in enumerate(group_sorted):
                    label = chr(ord("A") + i)
                    date_str = (b.get("created_at") or "")[:10]
                    st.markdown(
                        f"**Version {label}**"
                        + (f" ({date_str})" if date_str else "")
                    )
                    st.caption(b["position"])
                    col_keep, col_drop = st.columns(2)
                    with col_keep:
                        if st.button(
                            "✓ C'est ce que je pense",
                            key=f"conf_keep_{b['id']}",
                            use_container_width=True,
                        ):
                            pipeline.on_belief_confirmed(b["id"])
                            # Archiver les autres versions du même sujet
                            for other in group_sorted:
                                if other["id"] != b["id"]:
                                    pipeline.on_belief_rejected(other["id"])
                            st.rerun()
                    with col_drop:
                        if st.button(
                            "✗ Archiver",
                            key=f"conf_drop_{b['id']}",
                            use_container_width=True,
                        ):
                            pipeline.on_belief_rejected(b["id"])
                            st.rerun()

    # Croyances actives — éditables
    if active:
        from ui import stat_badge
        st.markdown("#### Mes croyances actives")
        for b in active:
            badge = stat_badge(
                b["confidence"], conf_colors.get(b["confidence"], "primary"),
            )
            col_txt, col_edit = st.columns([9, 1])
            with col_txt:
                st.html(
                    f"<div style='margin-bottom:2px'>"
                    f"<b>{b['topic']}</b> {badge}<br>"
                    f"<span style='color:#6B6880'>{b['position']}</span></div>"
                )
            with col_edit:
                edit_key = f"editing_belief_{b['id']}"
                if st.button("✏️", key=f"editbtn_{b['id']}", help="Modifier"):
                    st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                    st.rerun()

            if st.session_state.get(f"editing_belief_{b['id']}", False):
                with st.container(border=True):
                    new_pos = st.text_area(
                        "Ma position",
                        value=b["position"],
                        key=f"editpos_{b['id']}",
                        height=100,
                    )
                    new_conf = st.select_slider(
                        "Niveau de conviction",
                        options=["low", "medium", "high"],
                        value=b["confidence"] if b["confidence"] in ("low", "medium", "high") else "medium",
                        format_func=lambda x: {"low": "Faible", "medium": "Modérée", "high": "Forte"}[x],
                        key=f"editconf_{b['id']}",
                    )
                    c_save, c_cancel = st.columns(2)
                    with c_save:
                        if st.button(
                            "💾 Enregistrer", key=f"savebelief_{b['id']}",
                            use_container_width=True,
                        ):
                            pipeline.on_belief_updated(b["id"], new_pos, new_conf)
                            st.session_state[f"editing_belief_{b['id']}"] = False
                            st.toast("Croyance mise à jour ✍️")
                            st.rerun()
                    with c_cancel:
                        if st.button(
                            "Annuler", key=f"cancelbelief_{b['id']}",
                            use_container_width=True,
                        ):
                            st.session_state[f"editing_belief_{b['id']}"] = False
                            st.rerun()
    elif not conflicts:
        st.caption("Aucune croyance extraite pour le moment.")

    st.divider()

    # ── Section D : Historique d'évolution ──
    st.subheader("🔄 Historique d'évolution")
    st.caption("Comment votre pensée a évolué")
    try:
        timeline = pipeline.feedback_engine.get_belief_timeline()
    except Exception:
        timeline = []

    if timeline:
        # Regrouper par sujet
        tl_by_topic: dict[str, list] = {}
        for entry in timeline:
            tl_by_topic.setdefault(entry["topic"], []).append(entry)

        for topic, entries in tl_by_topic.items():
            with st.expander(f"📌 {topic} ({len(entries)} changement(s))"):
                for e in entries:
                    date_str = (e.get("created_at") or "")[:10]
                    reason_fr = {
                        "user_correction": "correction",
                        "conflict_resolved": "conflit résolu",
                        "thumbs_feedback": "retour 👎",
                        "explicit_update": "changement d'avis",
                    }.get(e.get("change_reason"), e.get("change_reason", ""))
                    if e.get("old_position"):
                        st.markdown(
                            f"_{date_str}_ · **{reason_fr}**  \n"
                            f"❌ {e['old_position']}  \n"
                            f"✅ {e.get('new_position') or '(archivée)'}"
                        )
                    else:
                        st.markdown(
                            f"_{date_str}_ · **{reason_fr}**  \n"
                            f"✅ {e.get('new_position') or ''}"
                        )
                    st.divider()
    else:
        st.caption(
            "Vos évolutions de pensée apparaîtront ici au fil de vos corrections."
        )

    st.divider()

    # ── Section E : Fidélité vocale ──
    st.subheader("📈 Fidélité vocale")
    try:
        history = pipeline.feedback_engine.get_fidelity_history(
            last_n=30,
        )
    except Exception:
        history = []

    if history:
        try:
            import pandas as pd
            import plotly.graph_objects as go

            df = pd.DataFrame(history)
            df["index"] = range(1, len(df) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["index"],
                y=df["overall_score"],
                mode="lines+markers",
                line=dict(color="#6C5CE7", width=3),
                marker=dict(size=6),
                name="Fidélité",
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 1], title="Score"),
                xaxis=dict(title="Réponses (récentes →)"),
                margin=dict(l=40, r=20, t=20, b=40),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            import pandas as pd
            df = pd.DataFrame(history)
            st.line_chart(df.set_index("index")["overall_score"] if "index" in df else df["overall_score"])

        try:
            summary = pipeline.get_learning_summary()
            trend = summary.get("fidelity_trend", "stable")
            trend_fr = {
                "improving": "📈 En amélioration",
                "stable": "➡️ Stable",
                "declining": "📉 En baisse",
            }.get(trend, "➡️ Stable")
            st.caption(f"Tendance : **{trend_fr}**")
        except Exception:
            pass

        st.caption(
            "Cette courbe mesure à quel point CogniAssist répond dans votre "
            "voix. Elle s'améliore à chaque correction que vous faites."
        )
    else:
        st.caption(
            "Le score de fidélité apparaîtra après vos premières réponses "
            "en mode second cerveau."
        )

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
    from ui.layout import PAGE_FILES
    st.switch_page(PAGE_FILES["profile"])

