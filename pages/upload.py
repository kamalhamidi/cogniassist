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

    from ui import page_header
    page_header(
        "Gestion des documents",
        "Importez vos fichiers PDF, DOCX ou TXT",
        icon="📁",
    )

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
            from ui import empty_state
            empty_state(
                "📭",
                "Votre bibliothèque est vide",
                "Importez vos premiers fichiers depuis l'onglet « Importer » "
                "pour commencer à interroger vos documents.",
            )
        else:
            # Métriques résumées
            from ui import metric_card
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Documents", len(docs), icon="📄")
            with c2:
                metric_card(
                    "Total chunks",
                    sum(d["chunk_count"] for d in docs),
                    icon="🧩",
                )
            with c3:
                metric_card(
                    "Avec résumé",
                    sum(1 for d in docs if d["has_summary"]),
                    icon="📝",
                )

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

    # ═══════════════════════════════════════════
    # Section : Écrits personnels (Layer 2 — Identité)
    # ═══════════════════════════════════════════
    st.divider()
    _show_personal_writing_section(user_id)


def _show_personal_writing_section(user_id: str) -> None:
    """Section d'import des écrits personnels pour alimenter l'identité."""
    from ui import section_title
    section_title("🧠 Alimenter mon identité")
    st.caption(
        "Importez vos écrits personnels (journal, essais, messages) pour que "
        "CogniAssist apprenne votre voix et vos opinions."
    )

    pw_files = st.file_uploader(
        "Mes journaux, essais, notes personnelles",
        type=["txt", "md"],
        accept_multiple_files=True,
        key="pw_uploader",
        help="Formats supportés : TXT, Markdown",
    )

    from datetime import date as _date
    pw_date = st.date_input(
        "Date approximative de ces écrits",
        value=None,
        key="pw_date",
        help="Utilisée pour le suivi temporel de vos positions (optionnel).",
    )

    if pw_files and st.button(
        "🧠 Analyser mes écrits", type="primary", key="btn_ingest_pw",
    ):
        date_written = pw_date if isinstance(pw_date, _date) else None
        _ingest_personal_writing(user_id, pw_files, date_written)


def _ingest_personal_writing(user_id: str, files: list, date_written) -> None:
    """Charge, indexe, analyse le style et extrait les croyances des écrits."""
    from ingestion import ingest_file
    from vectorstore.store import VectorStore
    from rag import get_pipeline

    all_chunks = []           # Documents LangChain pour ChromaDB
    style_texts: list[str] = []  # textes bruts pour l'analyse de style
    belief_chunks: list[dict] = []  # {text, chunk_id} pour l'extraction

    progress = st.progress(0.0, text="Lecture des fichiers…")
    total = len(files)

    for i, file in enumerate(files):
        try:
            file_bytes = file.read()
            chunks, _stats = ingest_file(file_bytes, file.name)
            for c in chunks:
                # Marquer comme écrit personnel
                c.metadata["source_type"] = "personal_writing"
                all_chunks.append(c)
                style_texts.append(c.page_content)
                belief_chunks.append({
                    "text": c.page_content,
                    "chunk_id": c.metadata.get("chunk_id", f"pw_{file.name}_{i}"),
                })
        except Exception as e:
            st.error(f"❌ Erreur sur {file.name} : {e}")
        progress.progress((i + 1) / total, text=f"Lecture… ({i + 1}/{total})")

    progress.empty()

    if not all_chunks:
        st.warning("Aucun contenu exploitable dans les fichiers fournis.")
        return

    # 1. Indexer dans la collection personal_writing
    try:
        store = VectorStore()
        added = store.add_personal_writing(all_chunks)
        st.caption(f"📥 {added} extrait(s) indexé(s) dans vos écrits personnels.")
    except Exception as e:
        st.error(f"Erreur indexation : {e}")

    pipeline = get_pipeline()

    # 2. Analyse du style (rapide, non bloquante)
    try:
        from user.db import get_session
        with st.spinner("Analyse de votre style d'écriture…"):
            session = get_session()
            metrics = pipeline.style_analyzer.analyze(style_texts)
            pipeline.style_analyzer.save_profile(metrics, session)
        st.caption("✍️ Profil de style mis à jour.")
    except Exception as e:
        st.warning(f"Analyse du style indisponible : {e}")

    # 3. Extraction des croyances via le LLM (dégradation gracieuse)
    beliefs_count = 0
    if not pipeline.is_ready:
        st.toast("⚠️ Ollama non démarré — extraction des croyances ignorée.")
    else:
        try:
            beliefs_count = pipeline.belief_extractor.extract_from_chunks(
                belief_chunks, date_written=date_written,
            )
        except Exception as e:
            st.toast(f"⚠️ Extraction des croyances échouée : {e}")

    # 4. Récapitulatif
    st.success(
        f"{beliefs_count} croyances extraites. "
        f"Votre profil d'identité a été mis à jour."
    )

    # 5. Aperçu des croyances extraites
    try:
        beliefs = pipeline.belief_extractor.get_all_beliefs()
        if beliefs:
            with st.expander("🔍 Aperçu des croyances extraites", expanded=True):
                for b in beliefs[:10]:
                    st.markdown(f"**{b['topic']}** — {b['position']}")
                    st.caption(f"Confiance : {b['confidence']} · Statut : {b['status']}")
    except Exception:
        pass

    st.cache_data.clear()


if __name__ == "__main__":
    st.session_state.current_page = "📁 Documents"
    st.switch_page("app.py")
