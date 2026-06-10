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

    # ═══ Sidebar ═══
    with st.sidebar:
        st.title("🧠 CogniAssist")
        st.caption("Assistant cognitif intelligent")
        st.divider()

        page = st.radio(
            "Navigation",
            options=["💬 Chat", "📁 Documents", "📊 Dashboard", "👤 Profil"],
            index=["💬 Chat", "📁 Documents", "📊 Dashboard", "👤 Profil"].index(
                st.session_state.current_page
            ),
            label_visibility="collapsed",
        )
        st.session_state.current_page = page

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

    # ═══ Know Me Better Gate ═══
    try:
        from user import get_kmb_manager
        _kmb_mgr = get_kmb_manager(st.session_state.user_id)
        if not _kmb_mgr.has_seen_kmb_onboarding() and not st.session_state.get("kmb_skipped", False):
            from pages.know_me_better import show_kmb_page
            show_kmb_page(is_onboarding=True)
            st.stop()
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
