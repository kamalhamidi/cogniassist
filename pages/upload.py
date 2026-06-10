"""
pages/upload.py — Page de gestion des documents.

Import de fichiers (PDF, DOCX, TXT, MD), ingestion, indexation
dans ChromaDB, et gestion de la bibliothèque de documents.
"""

import streamlit as st


@st.cache_data(ttl=30)
def _get_all_docs(user_id: str) -> list[dict]:
    """Récupère tous les documents de l'utilisateur."""
    from user import get_user_manager
    return get_user_manager(user_id).get_user_documents()


def show_upload_page() -> None:
    """Affiche la page de gestion des documents."""
    user_id = st.session_state.get("user_id", "default")

    st.title("📁 Gestion des documents")
    st.caption("Importez vos fichiers PDF, DOCX ou TXT")

    tab_upload, tab_library = st.tabs(["⬆️ Importer", "📚 Bibliothèque"])

    # ═══════════════════════════════════════════
    # Tab 1 : Import
    # ═══════════════════════════════════════════
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Glissez vos fichiers ici",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            help="Formats supportés : PDF, Word, Texte, Markdown",
        )

        if uploaded_files:
            for file in uploaded_files:
                with st.expander(f"📄 {file.name} ({file.size / 1024:.1f} Ko)"):
                    st.caption(f"Type : {file.type or file.name.split('.')[-1]}")

            if st.button("🚀 Lancer l'ingestion", type="primary", key="btn_ingest"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                total = len(uploaded_files)

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Traitement de {file.name}...")
                    progress_bar.progress((i + 1) / total)

                    try:
                        # 1. Ingestion
                        from ingestion import ingest_file
                        file_bytes = file.read()
                        chunks, stats = ingest_file(file_bytes, file.name)

                        # 2. Indexation vectorielle
                        from vectorstore import add_to_vectorstore
                        add_to_vectorstore(chunks)

                        # 3. Enregistrement profil
                        from user import get_user_manager
                        mgr = get_user_manager(user_id)
                        mgr.register_document(
                            file_name=file.name,
                            file_type=file.name.rsplit(".", 1)[-1],
                            chunk_count=stats["total_chunks"],
                        )

                        # 4. Résumé automatique
                        try:
                            from rag import get_pipeline
                            pipeline = get_pipeline()
                            if pipeline.is_ready:
                                with st.spinner(f"Résumé de {file.name}..."):
                                    summary = pipeline.summarize_document(file.name)
                                    mgr.store_document_summary(file.name, summary)
                        except Exception:
                            pass  # Le résumé est optionnel

                        with results_container:
                            st.success(
                                f"✅ {file.name} — {stats['total_chunks']} chunks créés"
                            )
                            with st.expander(f"📊 Statistiques — {file.name}"):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Chunks", stats["total_chunks"])
                                c2.metric("Moy. mots/chunk", f"{stats['avg_word_count']:.0f}")
                                c3.metric("Taille moy.", f"{stats['avg_chunk_size']:.0f} chars")

                    except Exception as e:
                        st.error(f"❌ Erreur sur {file.name} : {str(e)}")

                progress_bar.progress(1.0)
                status_text.text("✅ Ingestion terminée !")
                st.cache_data.clear()

    # ═══════════════════════════════════════════
    # Tab 2 : Bibliothèque
    # ═══════════════════════════════════════════
    with tab_library:
        try:
            docs = _get_all_docs(user_id)
        except Exception as e:
            st.error(f"Erreur chargement documents : {e}")
            docs = []

        if not docs:
            st.info("📭 Aucun document importé. Utilisez l'onglet Importer.")
        else:
            # Métriques résumées
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents", len(docs))
            c2.metric("Total chunks", sum(d["chunk_count"] for d in docs))
            c3.metric("Avec résumé", sum(1 for d in docs if d["has_summary"]))

            st.divider()

            # Liste des documents
            for doc in docs:
                with st.container(border=True):
                    col_info, col_actions = st.columns([8, 2])

                    with col_info:
                        st.markdown(f"**📄 {doc['file_name']}**")
                        st.caption(
                            f"Importé le {doc['upload_date'][:10] if doc['upload_date'] else '—'} · "
                            f"{doc['chunk_count']} chunks · "
                            f"Consulté {doc['access_count']} fois"
                        )
                        if doc["has_summary"]:
                            with st.expander("📝 Voir le résumé"):
                                try:
                                    from user import get_user_manager
                                    m = get_user_manager(user_id)
                                    # Récupérer le résumé depuis la base
                                    from user.db import get_session
                                    from user.profile import DocumentAccess
                                    session = get_session()
                                    da = (
                                        session.query(DocumentAccess)
                                        .filter_by(user_id=user_id, file_name=doc["file_name"])
                                        .first()
                                    )
                                    if da and da.summary:
                                        st.markdown(da.summary)
                                    else:
                                        st.caption("Résumé non disponible")
                                except Exception:
                                    st.caption("Erreur lors du chargement du résumé")

                    with col_actions:
                        if st.button(
                            "🗑️",
                            key=f"del_{doc['file_name']}",
                            help="Supprimer ce document",
                        ):
                            try:
                                from user import get_user_manager
                                get_user_manager(user_id).delete_document(doc["file_name"])
                                st.cache_data.clear()
                                st.success(f"{doc['file_name']} supprimé")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur suppression : {e}")


if __name__ == "__main__":
    st.session_state.current_page = "📁 Documents"
    st.switch_page("app.py")
