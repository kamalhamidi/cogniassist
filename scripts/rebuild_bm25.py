#!/usr/bin/env python3
"""
scripts/rebuild_bm25.py — Reconstruction de l'index BM25.

Script one-shot pour reconstruire l'index BM25 à partir des documents
déjà indexés dans ChromaDB. Utile après une migration ou si l'index
pickle est corrompu / manquant.

Usage :
    python scripts/rebuild_bm25.py
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.schema import Document
from config import settings, BASE_DIR
from vectorstore.store import VectorStore
from vectorstore.bm25_index import BM25Index


def main() -> None:
    """Reconstruit l'index BM25 depuis les chunks ChromaDB existants."""
    print("=" * 60)
    print("🔄 Reconstruction de l'index BM25")
    print("=" * 60)

    # 1. Charger le VectorStore (ChromaDB)
    print("\n📦 Chargement de ChromaDB…")
    store = VectorStore()
    collection = store.collection
    total_chunks = collection.count()

    if total_chunks == 0:
        print("⚠️  Aucun chunk dans ChromaDB. Rien à indexer.")
        return

    print(f"   ✅ {total_chunks} chunk(s) trouvé(s) dans ChromaDB.")

    # 2. Récupérer tous les chunks
    print("\n📥 Récupération de tous les documents et métadonnées…")
    all_data = collection.get(include=["documents", "metadatas"])

    documents_list: list[Document] = []
    raw_docs = all_data.get("documents", [])
    raw_metas = all_data.get("metadatas", [])

    for i, doc_text in enumerate(raw_docs):
        metadata = raw_metas[i] if i < len(raw_metas) and raw_metas[i] else {}
        documents_list.append(
            Document(page_content=doc_text, metadata=metadata)
        )

    print(f"   ✅ {len(documents_list)} document(s) récupéré(s).")

    # 3. Construire l'index BM25
    bm25_path = str((BASE_DIR / settings.BM25_INDEX_PATH).resolve())
    print(f"\n🔨 Construction de l'index BM25…")
    print(f"   📍 Chemin : {bm25_path}")

    bm25_index = BM25Index(persist_path=bm25_path)
    bm25_index.build(documents_list)

    print(f"\n✅ Index BM25 reconstruit avec succès !")
    print(f"   📊 {len(documents_list)} chunk(s) indexés")
    print(f"   💾 Sauvegardé dans : {bm25_path}")

    # 4. Test rapide
    print("\n🧪 Test rapide de l'index…")
    test_results = bm25_index.retrieve("test", k=3)
    print(f"   Résultats pour 'test' : {len(test_results)} chunk(s)")
    if test_results:
        top = test_results[0]
        print(f"   Top-1 score : {top['score']:.4f}")
        source = top["document"].metadata.get("file_name", "inconnu")
        print(f"   Top-1 source : {source}")

    print("\n" + "=" * 60)
    print("✅ Terminé !")
    print("=" * 60)


if __name__ == "__main__":
    main()
