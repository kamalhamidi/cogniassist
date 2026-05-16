"""
pages/chat.py — Page de chat intelligent.

Interface conversationnelle Streamlit pour interagir avec le
pipeline RAG. Affiche les messages, gère les entrées utilisateur
et affiche les sources utilisées pour chaque réponse.
"""

import streamlit as st


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "messages": [],
        "user_id": "default",
        "documents_loaded": False,
        "current_page": "Chat",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat_page() -> None:
    """Affiche la page de chat intelligent."""
    init_session_state()
    st.header("💬 Chat Intelligent")
    st.markdown("Posez vos questions sur vos documents. CogniAssist retrouve les informations pertinentes et génère une réponse.")

    # Vérifier que des documents sont chargés
    if not st.session_state.get("documents_loaded", False):
        st.info("📄 Aucun document chargé. Rendez-vous sur la page **Upload** pour importer vos fichiers.")

    st.markdown("---")

    # Afficher l'historique des messages
    for message in st.session_state.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

            # Afficher les sources si disponibles
            if role == "assistant" and "sources" in message:
                with st.expander("📚 Sources utilisées"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** — Score: {source.get('score', 'N/A'):.2f}")
                        st.caption(source.get("content", ""))

    # Zone de saisie
    if prompt := st.chat_input("Posez votre question..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche en cours..."):
                try:
                    from rag.pipeline import RAGPipeline

                    pipeline = RAGPipeline()
                    result = pipeline.query(prompt)

                    answer = result["answer"]
                    st.markdown(answer)

                    # Afficher les sources
                    if result.get("sources"):
                        with st.expander("📚 Sources utilisées"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**Source {i}** — Score: {source.get('score', 0):.2f}")
                                st.caption(source.get("content", ""))

                    # Sauvegarder dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result.get("sources", []),
                    })

                except Exception as e:
                    error_msg = f"❌ Erreur lors de la génération : {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

    # Bouton pour vider la conversation
    if st.session_state.get("messages"):
        if st.sidebar.button("🗑️ Vider la conversation"):
            st.session_state.messages = []
            st.rerun()


render_chat_page()
