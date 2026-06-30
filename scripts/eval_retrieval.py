#!/usr/bin/env python3
"""
scripts/eval_retrieval.py — Evaluation reelle du retrieval (Dense vs BM25 vs Hybride).

Methodologie : known-item passage retrieval avec questions generees par LLM.
1. Reconstruit le corpus reel depuis ChromaDB (chunks des documents indexes).
2. Reindexe proprement dans un VectorStore temporaire (vrais embeddings Ollama) + BM25.
3. Genere une question en francais par chunk (via mistral:7b, temperature 0) dont la
   reponse se trouve dans ce chunk -> le chunk source est l'unique passage pertinent (gold).
4. Pour chaque methode (Dense / BM25 / Hybride), recupere top-k et calcule :
   Precision@k, Recall@k, F1@k, MRR, Hit@1/3/5 et la latence moyenne par requete.

Usage : python scripts/eval_retrieval.py
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.schema import Document

CHROMA_SQLITE = PROJECT_ROOT / "data" / "chroma_db" / "chroma.sqlite3"
TARGET_FILES = {
    "Kamal_HAMIDI_CV_fran.pdf",
    "kamal_hamidi_portfolio.txt",
    "lettre de motivation.pdf",
}
K = 5
MAX_QUESTIONS = 40          # plafond du nombre de questions generees
MIN_CHUNK_CHARS = 120       # ignorer les chunks trop courts (peu informatifs)
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "retrieval_eval_results.json"


def extract_corpus() -> list[dict]:
    """Reconstruit les chunks uniques (texte, chunk_id, file_name) depuis chroma.sqlite3."""
    con = sqlite3.connect(str(CHROMA_SQLITE))
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT id, key, string_value, int_value
        FROM embedding_metadata
        WHERE key IN ('chroma:document', 'chunk_id', 'file_name', 'page_number', 'chunk_index')
        """
    ).fetchall()
    con.close()

    grouped: dict[int, dict] = {}
    for emb_id, key, sval, ival in rows:
        d = grouped.setdefault(emb_id, {})
        if key == "chroma:document":
            d["text"] = sval
        elif key == "chunk_id":
            d["chunk_id"] = sval
        elif key == "file_name":
            d["file_name"] = sval
        elif key == "page_number":
            d["page_number"] = ival if ival is not None else sval
        elif key == "chunk_index":
            d["chunk_index"] = ival

    chunks: list[dict] = []
    seen_text: set[str] = set()
    seen_cid: set[str] = set()
    for d in grouped.values():
        fn = d.get("file_name")
        text = d.get("text")
        cid = d.get("chunk_id")
        if not fn or not text or fn not in TARGET_FILES:
            continue
        norm = text.strip()
        if norm in seen_text:
            continue
        if cid and cid in seen_cid:
            continue
        seen_text.add(norm)
        if cid:
            seen_cid.add(cid)
        chunks.append({
            "chunk_id": cid or f"auto_{len(chunks)}",
            "file_name": fn,
            "page_number": d.get("page_number", "-"),
            "chunk_index": d.get("chunk_index", 0),
            "text": text,
        })
    return chunks


def build_indexes(chunks: list[dict]):
    """Construit un VectorStore (Ollama embeddings) + BM25 temporaires sur le corpus."""
    from vectorstore.store import VectorStore
    from vectorstore.bm25_index import BM25Index
    from vectorstore.retriever import SmartRetriever, HybridRetriever
    from config import settings

    tmp_dir = tempfile.mkdtemp(prefix="cogni_eval_")
    chroma_dir = os.path.join(tmp_dir, "chroma")
    bm25_path = os.path.join(tmp_dir, "bm25.pkl")

    store = VectorStore(persist_directory=chroma_dir)
    store.bm25_index = BM25Index(persist_path=bm25_path)

    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "file_name": c["file_name"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
            },
        )
        for c in chunks
    ]
    added = store.add_documents(docs)
    print(f"  -> {added} chunks reindexes (ChromaDB + BM25).")

    smart = SmartRetriever(vector_store=store)
    smart.min_similarity_score = 0.0  # pas de filtrage pour une eval equitable du rappel
    hybrid = HybridRetriever(
        smart_retriever=smart,
        bm25_index=store.bm25_index,
        dense_weight=settings.HYBRID_DENSE_WEIGHT,
        sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
    )
    return store, smart, hybrid, tmp_dir


def generate_questions(chunks: list[dict]) -> list[dict]:
    """Genere une question par chunk via mistral:7b (temperature 0, reproductible)."""
    from langchain_ollama import ChatOllama
    from config import settings

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_predict=80,
    )

    prompt_tpl = (
        "Tu generes UNE seule question d'evaluation a partir d'un extrait de document.\n"
        "Contraintes : la question doit avoir sa reponse UNIQUEMENT dans cet extrait, "
        "etre concise (une phrase), en francais, sans citer le mot 'extrait'.\n"
        "Reponds STRICTEMENT par la question, rien d'autre.\n\n"
        "EXTRAIT :\n{chunk}\n\nQUESTION :"
    )

    qset: list[dict] = []
    for i, c in enumerate(chunks):
        try:
            resp = llm.invoke(prompt_tpl.format(chunk=c["text"][:1200]))
            q = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            q = q.split("\n")[0].strip().strip('"').strip()
        except Exception as e:
            print(f"  ! question {i} echouee : {e}")
            continue
        if len(q) < 8:
            continue
        qset.append({"question": q, "gold_chunk_id": c["chunk_id"], "file_name": c["file_name"]})
        print(f"  [{len(qset):>2}] {c['file_name'][:22]:22} | {q[:70]}")
    return qset


def dense_retrieve(smart, query, k):
    return [d.metadata.get("chunk_id") for d in smart.retrieve(query, k=k)]


def bm25_retrieve(bm25, query, k):
    return [r["document"].metadata.get("chunk_id") for r in bm25.retrieve(query, k=k)]


def hybrid_retrieve(hybrid, query, k):
    return [d.metadata.get("chunk_id") for d in hybrid.retrieve(query, k=k)]


def evaluate(qset, smart, bm25, hybrid, k=K):
    methods = {
        "Dense": lambda q: dense_retrieve(smart, q, k),
        "BM25": lambda q: bm25_retrieve(bm25, q, k),
        "Hybrid": lambda q: hybrid_retrieve(hybrid, q, k),
    }
    agg = {m: {"p": 0.0, "r": 0.0, "mrr": 0.0, "hit1": 0, "hit3": 0, "hit5": 0,
               "time": 0.0, "n": 0} for m in methods}

    for item in qset:
        gold = item["gold_chunk_id"]
        q = item["question"]
        for m, fn in methods.items():
            t0 = time.perf_counter()
            ranked = fn(q)
            dt = (time.perf_counter() - t0) * 1000.0
            topk = ranked[:k]
            hit = 1 if gold in topk else 0
            rank = (topk.index(gold) + 1) if hit else 0
            a = agg[m]
            a["n"] += 1
            a["time"] += dt
            a["p"] += hit / k                 # 1 seul gold -> precision@k = hit/k
            a["r"] += hit                     # recall@k = hit (gold unique)
            a["mrr"] += (1.0 / rank) if rank else 0.0
            a["hit1"] += 1 if gold in ranked[:1] else 0
            a["hit3"] += 1 if gold in ranked[:3] else 0
            a["hit5"] += 1 if gold in ranked[:5] else 0

    summary = {}
    for m, a in agg.items():
        n = max(a["n"], 1)
        P = a["p"] / n
        R = a["r"] / n
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        summary[m] = {
            "precision@5": round(P, 3),
            "recall@5": round(R, 3),
            "f1@5": round(F1, 3),
            "mrr": round(a["mrr"] / n, 3),
            "hit@1": round(a["hit1"] / n, 3),
            "hit@3": round(a["hit3"] / n, 3),
            "hit@5": round(a["hit5"] / n, 3),
            "avg_time_ms": round(a["time"] / n, 1),
            "n_queries": a["n"],
        }
    return summary


def main():
    print("=" * 70)
    print("  EVALUATION RETRIEVAL — CogniAssist (corpus reel)")
    print("=" * 70)

    print("\n[1/4] Extraction du corpus reel depuis ChromaDB...")
    chunks = extract_corpus()
    by_file: dict[str, int] = {}
    for c in chunks:
        by_file[c["file_name"]] = by_file.get(c["file_name"], 0) + 1
    print(f"  -> {len(chunks)} chunks uniques : {by_file}")

    # Filtrer chunks trop courts + plafonner
    usable = [c for c in chunks if len(c["text"].strip()) >= MIN_CHUNK_CHARS]
    usable = usable[:MAX_QUESTIONS]
    print(f"  -> {len(usable)} chunks utilisables (>= {MIN_CHUNK_CHARS} chars, cap {MAX_QUESTIONS}).")

    print("\n[2/4] Reindexation propre (Ollama embeddings + BM25)...")
    store, smart, hybrid, tmp_dir = build_indexes(chunks)

    print("\n[3/4] Generation des questions (mistral:7b, temperature 0)...")
    qset = generate_questions(usable)
    print(f"  -> {len(qset)} questions generees.")

    print("\n[4/4] Evaluation Dense / BM25 / Hybride...")
    summary = evaluate(qset, smart, store.bm25_index, hybrid, k=K)

    print("\n" + "=" * 70)
    print(f"  RESULTATS (k={K}, {len(qset)} questions, corpus = {len(chunks)} chunks)")
    print("=" * 70)
    header = f"{'Methode':<10} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'MRR':>6} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'Temps(ms)':>10}"
    print(header)
    print("-" * len(header))
    for m in ["Hybrid", "Dense", "BM25"]:
        s = summary[m]
        print(f"{m:<10} {s['precision@5']:>6} {s['recall@5']:>6} {s['f1@5']:>6} "
              f"{s['mrr']:>6} {s['hit@1']:>6} {s['hit@3']:>6} {s['hit@5']:>6} {s['avg_time_ms']:>10}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "k": K,
            "n_chunks": len(chunks),
            "chunks_per_file": by_file,
            "n_questions": len(qset),
            "summary": summary,
            "questions": qset,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Resultats detailles -> {RESULTS_PATH}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
