"""
CogniAssist — Point d'entrée principal de l'application Streamlit.

Lance l'interface multi-pages avec navigation sidebar :
- 💬 Chat : Conversation intelligente avec le RAG
- 📁 Documents : Import et gestion de documents
- 📊 Dashboard : Statistiques et visualisations
- 👤 Profil : Gestion du profil utilisateur
"""

import streamlit as st

st.set_page_config(
    page_title="CogniAssist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "user_id": "default",
        "messages": [],
        "current_page": "💬 Chat",
        "pipeline_ready": False,
        "last_interaction_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def _get_cached_pipeline():
    """Charge le pipeline RAG une seule fois."""
    from rag import get_pipeline
    return get_pipeline()


def main() -> None:
    """Point d'entrée principal de l'application."""
    init_session_state()

    # ═══ Thème global ═══
    from ui import apply_theme
    apply_theme()

    # ═══ Sidebar ═══
    with st.sidebar:
        st.html(
            """
            <div style="display:flex;align-items:center;gap:12px;
                        padding:6px 2px 14px;">
                <div style="font-size:1.7rem;width:48px;height:48px;border-radius:14px;
                            display:flex;align-items:center;justify-content:center;
                            background:linear-gradient(135deg,#6C5CE7,#A855F7);
                            box-shadow:0 8px 18px rgba(108,92,231,.35);">🧠</div>
                <div>
                    <div style="font-size:1.25rem;font-weight:800;color:#1E1B2E;
                                line-height:1;">CogniAssist</div>
                    <div style="font-size:.8rem;color:#6B6880;margin-top:2px;">
                        Assistant cognitif intelligent</div>
                </div>
            </div>
            """
        )
        st.divider()

        # Statut du pipeline
        try:
            pipeline = _get_cached_pipeline()
            if pipeline.is_ready:
                st.success("✅ Mistral:7b prêt")
                st.session_state.pipeline_ready = True
            else:
                st.error("❌ Ollama non démarré")
                st.session_state.pipeline_ready = False

            status = pipeline.get_pipeline_status()
            st.caption(f"🤖 Modèle : {status['llm_model']}")
            st.caption(f"📚 Documents indexés : {status['total_documents_indexed']}")
        except Exception:
            st.error("❌ Pipeline indisponible")
            st.session_state.pipeline_ready = False

        st.divider()
        st.caption("Master SID 2025–2026")
        st.caption("HAMIDI Kamal")

    # ═══ Onboarding Gate ═══
    try:
        from user import get_user_manager
        _mgr = get_user_manager(st.session_state.user_id)
        _profile = _mgr.get_profile()
        if not _profile.get("onboarding_completed", False):
            from pages.onboarding import show_onboarding_page
            show_onboarding_page()
            st.stop()
    except Exception:
        pass  # Si erreur, continuer normalement

    # ═══ Top Nav Bar ═══
    if "query_page_checked" not in st.session_state:
        query_page = st.query_params.get("page")
        page_mapping = {
            "chat": "💬 Chat",
            "upload": "📁 Documents",
            "dashboard": "📊 Dashboard",
            "profile": "👤 Profil"
        }
        if query_page in page_mapping:
            st.session_state.current_page = page_mapping[query_page]
        elif st.session_state.get("current_page") in page_mapping.values():
            page_reverse_mapping = {
                "💬 Chat": "chat",
                "📁 Documents": "upload",
                "📊 Dashboard": "dashboard",
                "👤 Profil": "profile"
            }
            st.query_params["page"] = page_reverse_mapping[st.session_state.current_page]
        else:
            st.session_state.current_page = "💬 Chat"
            st.query_params["page"] = "chat"
        st.session_state.query_page_checked = True

    nav_pages = ["💬 Chat", "📁 Documents", "📊 Dashboard", "👤 Profil"]

    selected_page = st.segmented_control(
        "Navigation",
        options=nav_pages,
        default=st.session_state.current_page,
        label_visibility="collapsed",
        key="top_navbar"
    )
    
    if selected_page and selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        page_reverse_mapping = {
            "💬 Chat": "chat",
            "📁 Documents": "upload",
            "📊 Dashboard": "dashboard",
            "👤 Profil": "profile"
        }
        st.query_params["page"] = page_reverse_mapping[selected_page]
        st.rerun()
    elif not selected_page:
        st.rerun()

    page = st.session_state.current_page
    st.divider()

    # ═══ Know Me Better Gate ═══
    try:
        from user import get_kmb_manager
        _kmb_mgr = get_kmb_manager(st.session_state.user_id)
        if not _kmb_mgr.has_seen_kmb_onboarding() and not st.session_state.get("kmb_skipped", False):
            if page == "💬 Chat":
                from pages.know_me_better import show_kmb_page
                show_kmb_page(is_onboarding=True)
                st.stop()
            else:
                # Si l'utilisateur clique sur une autre page de la sidebar, on considère qu'il passe/reporte l'onboarding KMB
                st.session_state.kmb_skipped = True
    except Exception:
        pass  # Si erreur, continuer normalement

    # ═══ Routage des pages ═══
    if page == "💬 Chat":
        from pages.chat import show_chat_page
        show_chat_page()
    elif page == "📁 Documents":
        from pages.upload import show_upload_page
        show_upload_page()
    elif page == "📊 Dashboard":
        from pages.dashboard import show_dashboard_page
        show_dashboard_page()
    elif page == "👤 Profil":
        from pages.profile import show_profile_page
        show_profile_page()


if __name__ == "__main__":
    main()
