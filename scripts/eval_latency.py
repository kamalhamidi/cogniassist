#!/usr/bin/env python3
"""
scripts/eval_latency.py — Mesure de latence (retrieval vs generation vs total).

Reconstruit le corpus reel, mesure pour un jeu de requetes representatives :
- le temps de RECUPERATION (HybridRetriever),
- le temps de GENERATION (ChatOllama mistral:7b avec contexte, num_predict=1024),
- le temps TOTAL par requete.
"""

import os
import sys
import time
import sqlite3
import tempfile
import shutil
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

QUERIES = [
    "Quelles sont les competences techniques de Kamal Hamidi ?",
    "Quels projets de deep learning a realise Kamal ?",
    "Quel est le parcours academique de Kamal Hamidi ?",
    "Quelles experiences professionnelles a-t-il eues en developpement web ?",
    "Quelles certifications en data science possede Kamal ?",
]


def extract_corpus() -> list[dict]:
    con = sqlite3.connect(str(CHROMA_SQLITE))
    rows = con.execute(
        """SELECT id, key, string_value, int_value FROM embedding_metadata
           WHERE key IN ('chroma:document','chunk_id','file_name','page_number','chunk_index')"""
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
    chunks, seen = [], set()
    for d in grouped.values():
        fn, text, cid = d.get("file_name"), d.get("text"), d.get("chunk_id")
        if not fn or not text or fn not in TARGET_FILES:
            continue
        if text.strip() in seen or (cid and cid in seen):
            continue
        seen.add(text.strip())
        if cid:
            seen.add(cid)
        chunks.append({"chunk_id": cid or f"a_{len(chunks)}", "file_name": fn,
                       "page_number": d.get("page_number", "-"),
                       "chunk_index": d.get("chunk_index", 0), "text": text})
    return chunks


def main():
    print("=" * 64)
    print("  MESURE DE LATENCE — CogniAssist")
    print("=" * 64)

    from vectorstore.store import VectorStore
    from vectorstore.bm25_index import BM25Index
    from vectorstore.retriever import SmartRetriever, HybridRetriever
    from langchain_ollama import ChatOllama
    from config import settings

    chunks = extract_corpus()
    print(f"\nCorpus : {len(chunks)} chunks.")

    tmp_dir = tempfile.mkdtemp(prefix="cogni_lat_")
    store = VectorStore(persist_directory=os.path.join(tmp_dir, "chroma"))
    store.bm25_index = BM25Index(persist_path=os.path.join(tmp_dir, "bm25.pkl"))
    store.add_documents([
        Document(page_content=c["text"], metadata={
            "chunk_id": c["chunk_id"], "file_name": c["file_name"],
            "page_number": c["page_number"], "chunk_index": c["chunk_index"]})
        for c in chunks
    ])

    smart = SmartRetriever(vector_store=store)
    hybrid = HybridRetriever(smart, store.bm25_index,
                             dense_weight=settings.HYBRID_DENSE_WEIGHT,
                             sparse_weight=settings.HYBRID_SPARSE_WEIGHT)
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                     temperature=0.3, num_predict=1024, top_k=40, top_p=0.9)

    # Warm-up (charge le modele en VRAM, hors mesure)
    print("Warm-up du modele...")
    _ = llm.invoke("Bonjour")

    retr_times, gen_times, tot_times = [], [], []
    print(f"\n{'#':>2} {'Retr(ms)':>9} {'Gen(ms)':>9} {'Total(ms)':>10}  Question")
    print("-" * 64)
    for i, q in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        context = hybrid.retrieve_with_context_window(q, k=K)
        t1 = time.perf_counter()
        prompt = (
            "Tu es un assistant. Reponds a la question en te basant sur le contexte.\n\n"
            f"CONTEXTE :\n{context}\n\nQUESTION : {q}\n\nREPONSE :"
        )
        resp = llm.invoke(prompt)
        t2 = time.perf_counter()
        rt = (t1 - t0) * 1000
        gt = (t2 - t1) * 1000
        tt = (t2 - t0) * 1000
        retr_times.append(rt); gen_times.append(gt); tot_times.append(tt)
        print(f"{i:>2} {rt:>9.1f} {gt:>9.1f} {tt:>10.1f}  {q[:32]}")

    n = len(tot_times)
    avg = lambda xs: sum(xs) / n
    print("\n" + "=" * 64)
    print("  MOYENNES")
    print("=" * 64)
    print(f"  Retrieval        : {avg(retr_times):>10.1f} ms")
    print(f"  Generation       : {avg(gen_times):>10.1f} ms  ({avg(gen_times)/1000:.1f} s)")
    print(f"  Total par requete: {avg(tot_times):>10.1f} ms  ({avg(tot_times)/1000:.1f} s)")
    print(f"\n  Part generation  : {100*avg(gen_times)/avg(tot_times):.1f} %")
    print(f"  Part retrieval   : {100*avg(retr_times)/avg(tot_times):.2f} %")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
