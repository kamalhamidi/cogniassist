"""
CogniAssist — Point d'entrée principal de l'application Streamlit.

Lance l'interface multi-pages avec navigation sidebar :
- 💬 Chat : Conversation intelligente avec le RAG
- 📄 Upload : Import de documents (PDF, DOCX, TXT)
- 📊 Dashboard : Statistiques et visualisations
- 👤 Profil : Gestion du profil utilisateur
"""

import streamlit as st
from config import settings


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "messages": [],
        "user_id": "default",
        "documents_loaded": False,
        "current_page": "Accueil",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    """Point d'entrée principal de l'application."""

    # Configuration de la page
    st.set_page_config(
        page_title=settings.APP_NAME,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialisation de la session
    init_session_state()

    # === Sidebar — Navigation ===
    with st.sidebar:
        st.image("assets/logo.png", width=80) if False else st.markdown("# 🧠")
        st.title(settings.APP_NAME)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=["🏠 Accueil", "💬 Chat", "📄 Upload", "📊 Dashboard", "👤 Profil"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(f"v0.1.0 • {settings.ollama_model}")

    # === Routage des pages ===
    if page == "🏠 Accueil":
        show_home()
    elif page == "💬 Chat":
        from pages.chat import render_chat_page
        render_chat_page()
    elif page == "📄 Upload":
        from pages.upload import render_upload_page
        render_upload_page()
    elif page == "📊 Dashboard":
        from pages.dashboard import render_dashboard_page
        render_dashboard_page()
    elif page == "👤 Profil":
        from pages.profile import render_profile_page
        render_profile_page()


def show_home() -> None:
    """Affiche la page d'accueil avec un message de bienvenue."""
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🧠 CogniAssist</h1>
            <p style="font-size: 1.3rem; color: #888; margin-bottom: 2rem;">
                Votre assistant cognitif intelligent personnalisé
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Importez vos documents")
        st.markdown(
            "Chargez vos fichiers PDF, DOCX ou TXT. "
            "CogniAssist les analyse et les indexe automatiquement."
        )

    with col2:
        st.markdown("### 💬 Posez vos questions")
        st.markdown(
            "Interrogez vos documents en langage naturel. "
            "Le système retrouve les passages pertinents et génère une réponse précise."
        )

    with col3:
        st.markdown("### 🎯 Réponses personnalisées")
        st.markdown(
            "CogniAssist s'adapte à votre profil et à vos préférences "
            "pour des réponses toujours plus pertinentes."
        )

    st.markdown("---")

    # Statut rapide
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("Documents chargés", "0" if not st.session_state.documents_loaded else "✓")
    with status_col2:
        st.metric("Conversations", len(st.session_state.messages))
    with status_col3:
        st.metric("Modèle", settings.ollama_model)


if __name__ == "__main__":
    main()
