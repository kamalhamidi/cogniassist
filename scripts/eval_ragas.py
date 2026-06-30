#!/usr/bin/env python3
"""
scripts/eval_ragas.py — Evaluation RAGAS de bout en bout (corpus reel, augmente).

Etapes :
1. Reconstruit le corpus reel depuis ChromaDB et le reindexe (Ollama + BM25).
2. Genere un jeu de test AUGMENTE : (question, ground_truth) par chunk via mistral.
3. Execute le pipeline : recuperation hybride (contexts) + generation (answer).
4. Evalue avec la VRAIE librairie RAGAS (juge = mistral:7b, embeddings = nomic-embed-text)
   sur 4 metriques : Faithfulness, Answer Relevancy, Context Precision, Context Recall.
5. Calcule en parallele des proxies lexicaux (garde-fou, toujours disponibles).

Usage : python scripts/eval_ragas.py [n_questions]
"""

import json
import os
import re
import sys
import sqlite3
import tempfile
import shutil
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
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_CHUNK_CHARS = 150
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ragas_eval_results.json"


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


def gen_qa(chunks, llm, n):
    """Genere n paires (question, ground_truth) ancrees dans les chunks."""
    usable = [c for c in chunks if len(c["text"].strip()) >= MIN_CHUNK_CHARS][:n]
    tpl = (
        "A partir de l'extrait suivant, genere UNE question factuelle et sa reponse courte.\n"
        "Reponds STRICTEMENT en JSON : {{\"question\": \"...\", \"reponse\": \"...\"}}\n"
        "La reponse doit etre concise et entierement contenue dans l'extrait, en francais.\n\n"
        "EXTRAIT :\n{chunk}\n\nJSON :"
    )
    qa = []
    for i, c in enumerate(usable, 1):
        try:
            out = llm.invoke(tpl.format(chunk=c["text"][:1200]))
            txt = out.content if hasattr(out, "content") else str(out)
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
            q = (obj.get("question") or "").strip()
            gt = (obj.get("reponse") or obj.get("answer") or "").strip()
        except Exception as e:
            print(f"  ! QA {i} echoue : {e}")
            continue
        if len(q) < 8 or len(gt) < 2:
            continue
        qa.append({"question": q, "ground_truth": gt, "gold_chunk_id": c["chunk_id"],
                   "file_name": c["file_name"]})
        print(f"  [{len(qa):>2}] Q: {q[:60]}")
    return qa


def lexical_proxies(question, answer, contexts, ground_truth):
    """Proxies lexicaux (Jaccard) — garde-fou toujours disponible."""
    sw = set("le la les de du des un une et en a au aux ce qui que par sur dans est sont "
             "avec pour pas ne se sa son the of in is are and to for with on it this that".split())
    def toks(t):
        return {w.strip("?!.,;:'\"()[]{}-") for w in t.lower().split()
                if w.strip("?!.,;:'\"()[]{}-") and w.strip("?!.,;:'\"()[]{}-") not in sw}
    ctx = " ".join(contexts)
    def jac(a, b):
        ta, tb = toks(a), toks(b)
        return len(ta & tb) / len(ta | tb) if (ta and tb) else 0.0
    def rec(ref, cand):
        tr, tc = toks(ref), toks(cand)
        return len(tr & tc) / len(tr) if tr else 0.0
    return {
        "faithfulness": round(min(jac(answer, ctx) * 2, 1.0), 3),
        "answer_relevancy": round(min(jac(answer, question) * 2, 1.0), 3),
        "context_precision": round(min(jac(ctx, question) * 2, 1.0), 3),
        "context_recall": round(min(rec(ground_truth, ctx) * 2, 1.0), 3),
    }


def main():
    print("=" * 66)
    print("  EVALUATION RAGAS — CogniAssist (corpus reel, augmente)")
    print("=" * 66)

    from vectorstore.store import VectorStore
    from vectorstore.bm25_index import BM25Index
    from vectorstore.retriever import SmartRetriever, HybridRetriever
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from config import settings

    chunks = extract_corpus()
    print(f"\n[1/5] Corpus : {len(chunks)} chunks.")

    tmp_dir = tempfile.mkdtemp(prefix="cogni_ragas_")
    store = VectorStore(persist_directory=os.path.join(tmp_dir, "chroma"))
    store.bm25_index = BM25Index(persist_path=os.path.join(tmp_dir, "bm25.pkl"))
    store.add_documents([
        Document(page_content=c["text"], metadata={
            "chunk_id": c["chunk_id"], "file_name": c["file_name"]}) for c in chunks])
    smart = SmartRetriever(vector_store=store)
    hybrid = HybridRetriever(smart, store.bm25_index,
                             dense_weight=settings.HYBRID_DENSE_WEIGHT,
                             sparse_weight=settings.HYBRID_SPARSE_WEIGHT)

    gen_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                         temperature=0.3, num_predict=512)
    qa_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                        temperature=0.0, num_predict=200)

    print(f"\n[2/5] Generation du jeu augmente ({N_QUESTIONS} questions cibles)...")
    qa = gen_qa(chunks, qa_llm, N_QUESTIONS)
    print(f"  -> {len(qa)} paires (question, ground_truth).")

    print(f"\n[3/5] Execution du pipeline (retrieval hybride + generation)...")
    samples = []
    for i, item in enumerate(qa, 1):
        q = item["question"]
        docs = hybrid.retrieve(q, k=K)
        contexts = [d.page_content for d in docs]
        ctx_block = "\n\n".join(f"[{j+1}] {c}" for j, c in enumerate(contexts))
        prompt = ("Reponds a la question uniquement avec le contexte. Si l'info est absente, "
                  f"dis-le.\n\nCONTEXTE :\n{ctx_block}\n\nQUESTION : {q}\n\nREPONSE :")
        ans = gen_llm.invoke(prompt)
        answer = (ans.content if hasattr(ans, "content") else str(ans)).strip()
        item["answer"] = answer
        item["contexts"] = contexts
        samples.append(item)
        print(f"  [{i:>2}/{len(qa)}] genere ({len(answer)} chars)")

    # Proxies lexicaux
    for s in samples:
        s["lexical"] = lexical_proxies(s["question"], s["answer"], s["contexts"], s["ground_truth"])

    print(f"\n[4/5] Evaluation RAGAS (juge = {settings.ollama_model})...")
    ragas_summary, ragas_per = {}, []
    try:
        from ragas import evaluate, RunConfig
        from ragas.dataset_schema import SingleTurnSample
        from ragas import EvaluationDataset
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        judge = LangchainLLMWrapper(ChatOllama(
            model=settings.ollama_model, base_url=settings.ollama_base_url,
            temperature=0.0, num_predict=512))
        emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url))

        ds = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["question"], response=s["answer"],
                retrieved_contexts=s["contexts"], reference=s["ground_truth"],
            ) for s in samples
        ])
        rc = RunConfig(timeout=180, max_retries=1, max_workers=2)
        result = evaluate(
            dataset=ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge, embeddings=emb, run_config=rc, raise_exceptions=False,
        )
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c in
                       ("faithfulness", "answer_relevancy", "context_precision", "context_recall")]
        for c in metric_cols:
            vals = [v for v in df[c].tolist() if v is not None and v == v]  # drop NaN
            ragas_summary[c] = round(sum(vals) / len(vals), 3) if vals else None
            ragas_summary[c + "_valid_n"] = len(vals)
        ragas_per = df.to_dict(orient="records")
        print("  -> RAGAS OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ! RAGAS a echoue : {e}")

    # Moyennes lexicales
    lex_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    lex_summary = {k: round(sum(s["lexical"][k] for s in samples) / len(samples), 3)
                   for k in lex_keys} if samples else {}

    print("\n" + "=" * 66)
    print(f"  RESULTATS ({len(samples)} questions, corpus = {len(chunks)} chunks, k={K})")
    print("=" * 66)
    print(f"\n  {'Metrique':<22} {'RAGAS (juge LLM)':>18} {'Proxy lexical':>16}")
    print("  " + "-" * 56)
    labels = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
              "context_precision": "Context Precision", "context_recall": "Context Recall"}
    for k in lex_keys:
        rv = ragas_summary.get(k)
        rv_s = f"{rv:.3f}" if isinstance(rv, float) else "n/a"
        print(f"  {labels[k]:<22} {rv_s:>18} {lex_summary.get(k, 0):>16.3f}")
    if any(k + "_valid_n" in ragas_summary for k in lex_keys):
        print(f"\n  (RAGAS : nb d'echantillons valides par metrique = "
              f"{ {labels[k]: ragas_summary.get(k+'_valid_n') for k in lex_keys} })")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_chunks": len(chunks), "n_questions": len(samples), "k": K,
                   "ragas_summary": ragas_summary, "lexical_summary": lex_summary,
                   "samples": [{kk: vv for kk, vv in s.items() if kk != "contexts"} | 
                               {"n_contexts": len(s["contexts"])} for s in samples]},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  Resultats detailles -> {RESULTS_PATH}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
