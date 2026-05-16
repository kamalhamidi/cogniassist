"""
pages/upload.py — Page d'import de documents.

Interface Streamlit pour uploader des fichiers PDF, DOCX et TXT,
les traiter (nettoyage, chunking) et les indexer dans la base
vectorielle ChromaDB.
"""

import streamlit as st
from pathlib import Path
import uuid


def render_upload_page() -> None:
    """Affiche la page d'import de documents."""
    st.header("📄 Import de Documents")
    st.markdown("Chargez vos fichiers pour les indexer dans la base de connaissances.")

    # Zone d'upload
    uploaded_files = st.file_uploader(
        "Glissez-déposez vos fichiers ici",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Formats supportés : PDF, DOCX, TXT",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} fichier(s) sélectionné(s)**")

        # Paramètres de chunking
        with st.expander("⚙️ Paramètres de découpage", expanded=False):
            from config import settings

            chunk_size = st.slider(
                "Taille des chunks (caractères)",
                min_value=100,
                max_value=2000,
                value=settings.CHUNK_SIZE,
                step=50,
            )
            chunk_overlap = st.slider(
                "Chevauchement (caractères)",
                min_value=0,
                max_value=200,
                value=settings.CHUNK_OVERLAP,
                step=10,
            )

        # Bouton de traitement
        if st.button("🚀 Traiter et indexer les documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                from config import settings
                from ingestion.loader import DocumentLoader
                from ingestion.cleaner import TextCleaner
                from ingestion.chunker import TextChunker
                from vectorstore.embedder import Embedder
                from vectorstore.store import VectorStore

                loader = DocumentLoader()
                cleaner = TextCleaner()
                chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                embedder = Embedder()
                store = VectorStore()

                total_chunks = 0

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.markdown(f"📖 Traitement de **{uploaded_file.name}**...")
                    progress_bar.progress((idx) / len(uploaded_files))

                    # Sauvegarder le fichier uploadé
                    save_path = settings.upload_dir_path / uploaded_file.name
                    save_path.write_bytes(uploaded_file.getbuffer())

                    # Charger et nettoyer
                    doc = loader.load(save_path)
                    doc.content = cleaner.clean(doc.content)

                    # Découper en chunks
                    chunks = chunker.split(doc.content, metadata={"filename": doc.filename})

                    if chunks:
                        # Générer les embeddings
                        texts = [c.content for c in chunks]
                        embeddings = embedder.embed_texts(texts)

                        # Stocker dans ChromaDB
                        ids = [f"{doc.filename}_{uuid.uuid4().hex[:8]}" for _ in chunks]
                        metadatas = [c.metadata for c in chunks]
                        store.add_documents(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

                        total_chunks += len(chunks)

                progress_bar.progress(1.0)
                st.session_state.documents_loaded = True

                st.success(
                    f"✅ **{len(uploaded_files)} document(s)** traité(s) avec succès ! "
                    f"({total_chunks} chunks indexés)"
                )

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement : {str(e)}")

    # Statistiques de la base
    st.markdown("---")
    st.subheader("📊 État de la base de connaissances")

    try:
        from vectorstore.store import VectorStore
        store = VectorStore()
        count = store.count()
        st.metric("Chunks indexés", count)
    except Exception:
        st.info("Base vectorielle non initialisée.")
