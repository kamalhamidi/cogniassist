"""
pages/dashboard.py — Page de statistiques et visualisations.

Affiche des métriques clés, des graphiques d'utilisation et
des statistiques sur les interactions et la base de connaissances.
"""

import streamlit as st


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "messages": [],
        "user_id": "default",
        "documents_loaded": False,
        "current_page": "Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_dashboard_page() -> None:
    """Affiche la page de dashboard avec statistiques."""
    init_session_state()
    st.header("📊 Dashboard")
    st.markdown("Vue d'ensemble de votre activité CogniAssist.")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            from vectorstore.store import VectorStore
            store = VectorStore()
            st.metric("📄 Chunks indexés", store.count())
        except Exception:
            st.metric("📄 Chunks indexés", 0)

    with col2:
        st.metric("💬 Messages", len(st.session_state.get("messages", [])))

    with col3:
        st.metric("👤 Utilisateur", st.session_state.get("user_id", "default"))

    with col4:
        from config import settings
        st.metric("🤖 Modèle", settings.ollama_model)

    st.markdown("---")

    # Graphiques (placeholder)
    st.subheader("📈 Activité récente")
    st.info(
        "Les graphiques d'activité seront disponibles après quelques interactions. "
        "Commencez par importer des documents et poser des questions !"
    )

    # Section historique
    st.subheader("🕐 Dernières interactions")
    messages = st.session_state.get("messages", [])
    if messages:
        for msg in reversed(messages[-10:]):
            role = "🧑 Vous" if msg["role"] == "user" else "🤖 CogniAssist"
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            st.markdown(f"**{role}** : {content}")
    else:
        st.caption("Aucune interaction pour le moment.")

    st.markdown("---")

    # Actions de maintenance
    st.subheader("🔧 Maintenance")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🗑️ Réinitialiser la base vectorielle", type="secondary"):
            try:
                from vectorstore.store import VectorStore
                store = VectorStore()
                store.reset()
                st.session_state.documents_loaded = False
                st.success("✅ Base vectorielle réinitialisée.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

    with col_b:
        if st.button("🗑️ Vider l'historique de chat", type="secondary"):
            st.session_state.messages = []
            st.success("✅ Historique vidé.")
            st.rerun()


render_dashboard_page()
