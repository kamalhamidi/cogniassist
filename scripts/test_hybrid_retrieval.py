#!/usr/bin/env python3
"""
scripts/test_hybrid_retrieval.py — Test complet du hybrid retrieval.

Vérifie toutes les couches : BM25Index, RRF, HybridRetriever,
synchronisation VectorStore, et intégration pipeline.

Usage :
    python scripts/test_hybrid_retrieval.py
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.schema import Document

# ══════════════════════════════════════════════════════════════
# Couleurs terminal
# ══════════════════════════════════════════════════════════════
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
errors: list[str] = []


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}✅ PASS{RESET} — {msg}")


def fail(msg: str, detail: str = "") -> None:
    global failed
    failed += 1
    errors.append(msg)
    print(f"  {RED}❌ FAIL{RESET} — {msg}")
    if detail:
        print(f"         {detail}")


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}{RESET}")


# ══════════════════════════════════════════════════════════════
# DOCUMENTS DE TEST
# ══════════════════════════════════════════════════════════════
SAMPLE_DOCS = [
    Document(
        page_content="Le machine learning est une branche de l'intelligence artificielle "
                     "qui permet aux systèmes d'apprendre à partir de données.",
        metadata={"file_name": "cours_ml.pdf", "chunk_id": "ml_001", "chunk_index": 0},
    ),
    Document(
        page_content="Les réseaux de neurones convolutifs (CNN) sont utilisés pour la "
                     "reconnaissance d'images et le traitement visuel.",
        metadata={"file_name": "cours_ml.pdf", "chunk_id": "ml_002", "chunk_index": 1},
    ),
    Document(
        page_content="Python est un langage de programmation interprété, multi-paradigme, "
                     "très utilisé en data science et en intelligence artificielle.",
        metadata={"file_name": "guide_python.pdf", "chunk_id": "py_001", "chunk_index": 0},
    ),
    Document(
        page_content="Les frameworks Django et Flask permettent de créer des applications "
                     "web en Python rapidement et efficacement.",
        metadata={"file_name": "guide_python.pdf", "chunk_id": "py_002", "chunk_index": 1},
    ),
    Document(
        page_content="Le traitement du langage naturel (NLP) utilise des techniques de "
                     "deep learning comme les transformers et BERT pour comprendre le texte.",
        metadata={"file_name": "cours_nlp.pdf", "chunk_id": "nlp_001", "chunk_index": 0},
    ),
    Document(
        page_content="L'algorithme de gradient descent est fondamental en apprentissage "
                     "automatique pour optimiser les fonctions de perte.",
        metadata={"file_name": "cours_ml.pdf", "chunk_id": "ml_003", "chunk_index": 2},
    ),
    Document(
        page_content="Les bases de données vectorielles comme ChromaDB et Pinecone "
                     "permettent la recherche sémantique par similarité.",
        metadata={"file_name": "cours_rag.pdf", "chunk_id": "rag_001", "chunk_index": 0},
    ),
    Document(
        page_content="Le RAG (Retrieval-Augmented Generation) combine la recherche "
                     "documentaire avec la génération de texte par LLM.",
        metadata={"file_name": "cours_rag.pdf", "chunk_id": "rag_002", "chunk_index": 1},
    ),
]


# ══════════════════════════════════════════════════════════════
# TEST 1 : BM25Index — Fonctionnalités de base
# ══════════════════════════════════════════════════════════════
def test_bm25_index():
    section("Test 1 : BM25Index — Fonctionnalités de base")
    from vectorstore.bm25_index import BM25Index

    tmp_path = tempfile.mktemp(suffix=".pkl")

    try:
        # 1.1 — Construction de l'index
        idx = BM25Index(persist_path=tmp_path)
        idx.build(SAMPLE_DOCS)
        if len(idx.documents) == len(SAMPLE_DOCS):
            ok(f"build() : {len(idx.documents)} documents indexés")
        else:
            fail(f"build() : attendu {len(SAMPLE_DOCS)}, obtenu {len(idx.documents)}")

        # 1.2 — Recherche pertinente
        results = idx.retrieve("machine learning intelligence artificielle", k=3)
        if len(results) > 0:
            ok(f"retrieve() : {len(results)} résultat(s) trouvé(s)")
            top_id = results[0]["document"].metadata["chunk_id"]
            if top_id == "ml_001":
                ok(f"retrieve() : top-1 correct (chunk_id={top_id})")
            else:
                fail(f"retrieve() : top-1 inattendu (chunk_id={top_id}, attendu: ml_001)")
        else:
            fail("retrieve() : aucun résultat")

        # 1.3 — Scores non-nuls
        all_positive = all(r["score"] > 0 for r in results)
        if all_positive:
            ok("retrieve() : tous les scores > 0")
        else:
            fail("retrieve() : certains scores sont nuls")

        # 1.4 — Recherche sans résultat (termes absents)
        no_results = idx.retrieve("xyzqwertasdfg", k=5)
        if len(no_results) == 0:
            ok("retrieve() : 0 résultat pour une requête absurde")
        else:
            fail(f"retrieve() : {len(no_results)} résultat(s) pour une requête absurde")

        # 1.5 — Persistance (save/load)
        idx2 = BM25Index(persist_path=tmp_path)
        if len(idx2.documents) == len(SAMPLE_DOCS):
            ok(f"Persistance : {len(idx2.documents)} documents rechargés depuis pickle")
        else:
            fail(f"Persistance : attendu {len(SAMPLE_DOCS)}, obtenu {len(idx2.documents)}")

        # Vérifier que la recherche fonctionne après reload
        results2 = idx2.retrieve("machine learning", k=2)
        if len(results2) > 0:
            ok("Persistance : retrieve() fonctionne après rechargement")
        else:
            fail("Persistance : retrieve() échoue après rechargement")

        # 1.6 — add_documents()
        new_doc = Document(
            page_content="Les arbres de décision sont des algorithmes de classification supervisée.",
            metadata={"file_name": "cours_ml.pdf", "chunk_id": "ml_004", "chunk_index": 3},
        )
        idx.add_documents([new_doc])
        if len(idx.documents) == len(SAMPLE_DOCS) + 1:
            ok(f"add_documents() : {len(idx.documents)} documents (ajout OK)")
        else:
            fail(f"add_documents() : attendu {len(SAMPLE_DOCS) + 1}, obtenu {len(idx.documents)}")

        # 1.7 — remove_document()
        idx.remove_document("guide_python.pdf")
        remaining = [d.metadata["file_name"] for d in idx.documents]
        if "guide_python.pdf" not in remaining:
            ok(f"remove_document() : guide_python.pdf supprimé, reste {len(idx.documents)} docs")
        else:
            fail("remove_document() : guide_python.pdf toujours présent")

        # 1.8 — Index vide
        empty_idx = BM25Index(persist_path=tempfile.mktemp(suffix=".pkl"))
        empty_results = empty_idx.retrieve("test", k=5)
        if empty_results == []:
            ok("Index vide : retrieve() retourne []")
        else:
            fail(f"Index vide : attendu [], obtenu {len(empty_results)} résultat(s)")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# TEST 2 : Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════
def test_rrf():
    section("Test 2 : Reciprocal Rank Fusion (RRF)")
    from vectorstore.retriever import reciprocal_rank_fusion

    # Listes avec des documents communs et uniques
    dense = [
        Document(page_content="A dense", metadata={"chunk_id": "A"}),
        Document(page_content="B dense", metadata={"chunk_id": "B"}),
        Document(page_content="C dense", metadata={"chunk_id": "C"}),
    ]
    sparse = [
        Document(page_content="B sparse", metadata={"chunk_id": "B"}),
        Document(page_content="D sparse", metadata={"chunk_id": "D"}),
        Document(page_content="A sparse", metadata={"chunk_id": "A"}),
    ]

    # 2.1 — Nombre de résultats (déduplication)
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    unique_ids = [d.metadata["chunk_id"] for d in fused]
    if len(unique_ids) == 4:
        ok(f"Déduplication : 4 documents uniques (6 en entrée)")
    else:
        fail(f"Déduplication : attendu 4, obtenu {len(unique_ids)}")

    # 2.2 — B devrait être en premier (présent dans les 2 listes, rang élevé)
    if unique_ids[0] == "B":
        ok("Classement : B en premier (apparaît dans dense ET sparse)")
    else:
        fail(f"Classement : premier est {unique_ids[0]}, attendu B")

    # 2.3 — A devrait être en deuxième (dense rang 0 + sparse rang 2)
    if unique_ids[1] == "A":
        ok("Classement : A en deuxième (dense rang 0 + sparse rang 2)")
    else:
        fail(f"Classement : deuxième est {unique_ids[1]}, attendu A")

    # 2.4 — Pas de doublons
    if len(unique_ids) == len(set(unique_ids)):
        ok("Pas de doublons dans les résultats fusionnés")
    else:
        fail("Doublons détectés dans les résultats fusionnés")

    # 2.5 — Listes vides
    empty_fused = reciprocal_rank_fusion([], [], k=60)
    if empty_fused == []:
        ok("Listes vides : retourne []")
    else:
        fail(f"Listes vides : attendu [], obtenu {len(empty_fused)} résultat(s)")

    # 2.6 — Pondération : dense_weight > sparse_weight favorise les résultats dense
    dense_only = [
        Document(page_content="X", metadata={"chunk_id": "X"}),
    ]
    sparse_only = [
        Document(page_content="Y", metadata={"chunk_id": "Y"}),
    ]
    fused_weighted = reciprocal_rank_fusion(
        dense_only, sparse_only, k=60, dense_weight=0.9, sparse_weight=0.1,
    )
    if fused_weighted[0].metadata["chunk_id"] == "X":
        ok("Pondération : dense_weight=0.9 > sparse_weight=0.1 → X en premier")
    else:
        fail("Pondération : le document sparse ne devrait pas être premier")


# ══════════════════════════════════════════════════════════════
# TEST 3 : HybridRetriever
# ══════════════════════════════════════════════════════════════
def test_hybrid_retriever():
    section("Test 3 : HybridRetriever")
    from vectorstore.bm25_index import BM25Index
    from vectorstore.retriever import SmartRetriever, HybridRetriever

    # Créer un BM25 avec les documents de test
    tmp_path = tempfile.mktemp(suffix=".pkl")
    bm25 = BM25Index(persist_path=tmp_path)
    bm25.build(SAMPLE_DOCS)

    try:
        # Vérifier que HybridRetriever s'instancie sans erreur
        # (SmartRetriever nécessite un VectorStore qui nécessite Ollama,
        #  donc on teste la construction avec un mock ou on vérifie juste la classe)
        print(f"\n  {YELLOW}ℹ️  Les tests du HybridRetriever avec ChromaDB nécessitent")
        print(f"     qu'Ollama soit lancé. Test de la classe uniquement.{RESET}\n")

        # 3.1 — La classe existe et a les bonnes méthodes
        methods = ["retrieve", "retrieve_with_context_window"]
        for method in methods:
            if hasattr(HybridRetriever, method):
                ok(f"HybridRetriever.{method}() existe")
            else:
                fail(f"HybridRetriever.{method}() manquant")

        # 3.2 — Vérifier les attributs du constructeur
        import inspect
        sig = inspect.signature(HybridRetriever.__init__)
        params = list(sig.parameters.keys())
        expected = ["self", "smart_retriever", "bm25_index", "dense_weight", "sparse_weight"]
        if params == expected:
            ok(f"HybridRetriever.__init__ signature correcte")
        else:
            fail(f"HybridRetriever.__init__ signature inattendue : {params}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# TEST 4 : Config — Nouvelles variables
# ══════════════════════════════════════════════════════════════
def test_config():
    section("Test 4 : Configuration — Nouvelles variables")
    from config import settings

    # 4.1 — BM25_INDEX_PATH
    if hasattr(settings, "BM25_INDEX_PATH"):
        ok(f"BM25_INDEX_PATH = '{settings.BM25_INDEX_PATH}'")
    else:
        fail("BM25_INDEX_PATH manquant dans settings")

    # 4.2 — HYBRID_DENSE_WEIGHT
    if hasattr(settings, "HYBRID_DENSE_WEIGHT"):
        if 0.0 <= settings.HYBRID_DENSE_WEIGHT <= 1.0:
            ok(f"HYBRID_DENSE_WEIGHT = {settings.HYBRID_DENSE_WEIGHT}")
        else:
            fail(f"HYBRID_DENSE_WEIGHT hors bornes : {settings.HYBRID_DENSE_WEIGHT}")
    else:
        fail("HYBRID_DENSE_WEIGHT manquant dans settings")

    # 4.3 — HYBRID_SPARSE_WEIGHT
    if hasattr(settings, "HYBRID_SPARSE_WEIGHT"):
        if 0.0 <= settings.HYBRID_SPARSE_WEIGHT <= 1.0:
            ok(f"HYBRID_SPARSE_WEIGHT = {settings.HYBRID_SPARSE_WEIGHT}")
        else:
            fail(f"HYBRID_SPARSE_WEIGHT hors bornes : {settings.HYBRID_SPARSE_WEIGHT}")
    else:
        fail("HYBRID_SPARSE_WEIGHT manquant dans settings")


# ══════════════════════════════════════════════════════════════
# TEST 5 : Synchronisation VectorStore ↔ BM25
# ══════════════════════════════════════════════════════════════
def test_store_sync():
    section("Test 5 : Synchronisation VectorStore ↔ BM25")

    # Vérifier que store.py importe et instancie BM25Index
    from vectorstore.store import VectorStore
    import inspect

    source = inspect.getsource(VectorStore.__init__)

    if "bm25_index" in source or "BM25Index" in source:
        ok("VectorStore.__init__ instancie BM25Index")
    else:
        fail("VectorStore.__init__ n'instancie pas BM25Index")

    add_source = inspect.getsource(VectorStore.add_documents)
    if "bm25_index" in add_source:
        ok("VectorStore.add_documents() synchronise le BM25")
    else:
        fail("VectorStore.add_documents() ne synchronise pas le BM25")

    del_source = inspect.getsource(VectorStore.delete_document)
    if "bm25_index" in del_source:
        ok("VectorStore.delete_document() synchronise le BM25")
    else:
        fail("VectorStore.delete_document() ne synchronise pas le BM25")


# ══════════════════════════════════════════════════════════════
# TEST 6 : Pipeline — Intégration HybridRetriever
# ══════════════════════════════════════════════════════════════
def test_pipeline_integration():
    section("Test 6 : Pipeline — Intégration HybridRetriever")
    import inspect

    from rag.pipeline import RAGPipeline
    source = inspect.getsource(RAGPipeline.__init__)

    if "HybridRetriever" in source:
        ok("RAGPipeline.__init__ utilise HybridRetriever")
    else:
        fail("RAGPipeline.__init__ n'utilise pas HybridRetriever")

    if "HYBRID_DENSE_WEIGHT" in source or "dense_weight" in source:
        ok("RAGPipeline utilise les poids de config")
    else:
        fail("RAGPipeline n'utilise pas les poids de config")

    # Vérifier les imports au niveau du module
    from rag import pipeline as pipeline_module
    module_source = inspect.getsource(pipeline_module)
    if "from vectorstore.retriever import" in module_source and "HybridRetriever" in module_source:
        ok("pipeline.py importe HybridRetriever")
    else:
        fail("pipeline.py n'importe pas HybridRetriever")


# ══════════════════════════════════════════════════════════════
# TEST 7 : Test end-to-end BM25 → RRF (sans Ollama)
# ══════════════════════════════════════════════════════════════
def test_end_to_end_without_ollama():
    section("Test 7 : End-to-End BM25 + RRF (sans Ollama)")
    from vectorstore.bm25_index import BM25Index
    from vectorstore.retriever import reciprocal_rank_fusion

    tmp_path = tempfile.mktemp(suffix=".pkl")
    bm25 = BM25Index(persist_path=tmp_path)
    bm25.build(SAMPLE_DOCS)

    try:
        # Simuler une recherche dense (on prend les docs dans un ordre fixe)
        # En réalité, ce serait ChromaDB. Ici on simule.
        query = "réseaux de neurones deep learning"

        # Simuler dense results (comme si ChromaDB retournait ces résultats)
        dense_results = [
            SAMPLE_DOCS[1],  # CNN - rank 0 dense
            SAMPLE_DOCS[4],  # NLP/transformers - rank 1 dense
            SAMPLE_DOCS[0],  # ML général - rank 2 dense
        ]

        # Sparse results (BM25 réel)
        bm25_raw = bm25.retrieve(query, k=5)
        sparse_results = [r["document"] for r in bm25_raw]

        print(f"\n  {YELLOW}Résultats Dense (simulés) :{RESET}")
        for i, d in enumerate(dense_results):
            print(f"    rang {i}: {d.metadata['chunk_id']} — {d.page_content[:60]}…")

        print(f"\n  {YELLOW}Résultats Sparse (BM25 réel) :{RESET}")
        for i, r in enumerate(bm25_raw):
            d = r["document"]
            print(f"    rang {i}: {d.metadata['chunk_id']} (score={r['score']:.3f}) — {d.page_content[:60]}…")

        # Fusion RRF
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        print(f"\n  {YELLOW}Résultats Fusionnés (RRF) :{RESET}")
        for i, d in enumerate(fused[:5]):
            print(f"    rang {i}: {d.metadata['chunk_id']} — {d.page_content[:60]}…")

        # Vérifications
        if len(fused) > 0:
            ok(f"E2E : {len(fused)} résultats fusionnés")
        else:
            fail("E2E : aucun résultat fusionné")

        # Les documents communs aux deux listes devraient être en haut
        fused_ids = [d.metadata["chunk_id"] for d in fused]
        if len(fused_ids) == len(set(fused_ids)):
            ok("E2E : pas de doublons après fusion")
        else:
            fail("E2E : doublons détectés")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# TEST 8 : Test avec Ollama (si disponible)
# ══════════════════════════════════════════════════════════════
def test_with_ollama():
    section("Test 8 : Intégration complète avec Ollama (optionnel)")

    try:
        import requests
        resp = requests.get("http://localhost:11434/api/version", timeout=3)
        if resp.status_code != 200:
            raise ConnectionError()
    except Exception:
        print(f"\n  {YELLOW}⏭️  Ollama non détecté — test ignoré.{RESET}")
        print(f"  {YELLOW}   Lancez 'ollama run mistral:7b' pour activer ce test.{RESET}\n")
        return

    print(f"\n  {GREEN}🟢 Ollama détecté — test d'intégration complète{RESET}\n")

    try:
        from vectorstore.store import VectorStore
        from vectorstore.retriever import SmartRetriever, HybridRetriever
        from config import settings, BASE_DIR

        # Utiliser un répertoire temporaire pour ne pas polluer les vraies données
        tmp_dir = tempfile.mkdtemp(prefix="cogniassist_test_")
        chroma_dir = os.path.join(tmp_dir, "chroma_test")
        bm25_path = os.path.join(tmp_dir, "bm25_test.pkl")

        try:
            # Créer un VectorStore temporaire
            store = VectorStore(persist_directory=chroma_dir)

            # Remplacer le bm25_index par un temporaire
            from vectorstore.bm25_index import BM25Index
            store.bm25_index = BM25Index(persist_path=bm25_path)

            # Ajouter des documents
            added = store.add_documents(SAMPLE_DOCS)
            if added > 0:
                ok(f"VectorStore : {added} chunks ajoutés à ChromaDB + BM25")
            else:
                fail("VectorStore : aucun chunk ajouté")

            # Vérifier la sync BM25
            bm25_count = len(store.bm25_index.documents)
            if bm25_count > 0:
                ok(f"BM25 synchronisé : {bm25_count} documents")
            else:
                fail("BM25 non synchronisé après add_documents")

            # Créer le HybridRetriever
            smart = SmartRetriever(vector_store=store)
            hybrid = HybridRetriever(
                smart_retriever=smart,
                bm25_index=store.bm25_index,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
            )

            # Recherche hybride
            results = hybrid.retrieve("machine learning intelligence artificielle", k=3)
            if len(results) > 0:
                ok(f"HybridRetriever.retrieve() : {len(results)} résultat(s)")
                for i, doc in enumerate(results):
                    cid = doc.metadata.get("chunk_id", "?")
                    print(f"    rang {i}: {cid} — {doc.page_content[:60]}…")
            else:
                fail("HybridRetriever.retrieve() : aucun résultat")

            # Contexte formaté
            context = hybrid.retrieve_with_context_window("deep learning NLP", k=2)
            if "Extrait" in context and len(context) > 50:
                ok(f"retrieve_with_context_window() : contexte de {len(context)} chars")
            else:
                fail("retrieve_with_context_window() : contexte invalide")

            # Test suppression et sync
            deleted = store.delete_document("guide_python.pdf")
            bm25_after_del = len(store.bm25_index.documents)
            if deleted > 0 and bm25_after_del < bm25_count:
                ok(f"delete_document() : {deleted} chunks supprimés, BM25 sync ({bm25_after_del} restants)")
            else:
                fail("delete_document() : synchronisation BM25 échouée")

        finally:
            # Nettoyage
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        fail(f"Test Ollama : {e}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"\n{BOLD}{'🧪' * 3} CogniAssist — Test Hybrid Retrieval {'🧪' * 3}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")

    test_bm25_index()
    test_rrf()
    test_hybrid_retriever()
    test_config()
    test_store_sync()
    test_pipeline_integration()
    test_end_to_end_without_ollama()
    test_with_ollama()

    # Résumé
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  RÉSUMÉ")
    print(f"{'═' * 60}{RESET}")
    total = passed + failed
    print(f"\n  Total : {total} test(s)")
    print(f"  {GREEN}✅ Réussis : {passed}{RESET}")
    print(f"  {RED}❌ Échoués : {failed}{RESET}")

    if errors:
        print(f"\n  {RED}Erreurs :{RESET}")
        for err in errors:
            print(f"    • {err}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}🎉 TOUS LES TESTS PASSENT !{RESET}\n")
    else:
        print(f"\n  {RED}{BOLD}⚠️  {failed} test(s) échoué(s) — voir ci-dessus{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
