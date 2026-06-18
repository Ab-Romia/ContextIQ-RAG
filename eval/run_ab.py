"""Run the retrieval ablation.

Four arms over the same chunks and the same golden questions, changing only the retriever:

  A  TF-IDF baseline (properly fit)
  B  dense semantic only (bge-small)
  C  hybrid: dense + BM25 fused with reciprocal rank fusion
  D  hybrid + cross-encoder reranking

The only variable is retrieval, so the table reads as a clean before/after. Relevance is
resolved at run time by locating each question's gold answer span inside the current
chunks, so the metrics stay correct even if the chunk size changes. Unanswerable
questions carry no relevant chunk and are excluded from the retrieval metrics; they exist
to test abstention, which the optional judge measures.

Run from the repository root inside an environment that has the application dependencies:

    python -m eval.run_ab            # retrieval metrics only, no API key needed
    python -m eval.run_ab --judge --api-key sk-or-...   # adds the answer-quality judge

Retrieval metrics are deterministic and free. The judge is opt-in because it calls the
model provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.augment import augment
from app.chunking import chunk_document
from app.config import settings
from app.fusion import fuse
from app.models import Chunk
from app.rerank import rerank
from app.retrieval import RetrievalStore

from . import metrics

_DATASET = Path(__file__).parent / "dataset"
_RESULTS = Path(__file__).parent / "results.json"
_K_PRIMARY = 5


def load_chunks() -> list[Chunk]:
    """Chunk the target handbook plus every distractor document in corpus/.

    The questions only target handbook.md, but retrieval runs over the whole corpus so
    the right chunk has to be found among many similar-looking policy chunks from other
    fictional companies. Each document is chunked and augmented on its own so its
    contextual headers reference its own title and sections.
    """
    chunks: list[Chunk] = []
    documents = [_DATASET / "handbook.md", *sorted((_DATASET / "corpus").glob("*.md"))]
    for path in documents:
        doc_chunks = chunk_document(
            path.read_text(),
            path.name,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        chunks.extend(augment(doc_chunks))
    return chunks


def load_questions() -> list[dict]:
    rows = []
    for line in (_DATASET / "golden.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def relevant_ids(span: str, chunks: list[Chunk]) -> set[str]:
    """Chunks that contain the gold span. Falls back to a prefix match if chunk
    boundaries split the span, and refuses to silently return nothing."""
    hits = {c.id for c in chunks if span in c.text}
    if not hits:
        head = span[:60]
        hits = {c.id for c in chunks if head in c.text}
    if not hits:
        raise ValueError(f"Gold span did not resolve to any chunk: {span!r}")
    return hits


def score_arm(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "hit@3": metrics.hit_rate_at_k(ranked, relevant, 3),
        "hit@5": metrics.hit_rate_at_k(ranked, relevant, 5),
        "recall@5": metrics.recall_at_k(ranked, relevant, 5),
        "mrr": metrics.mrr(ranked, relevant),
        "ndcg@5": metrics.ndcg_at_k(ranked, relevant, 5),
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    keys = ["hit@3", "hit@5", "recall@5", "mrr", "ndcg@5"]
    return {key: round(metrics.mean([row[key] for row in per_query]), 4) for key in keys}


def build_rankings(query: str, store: RetrievalStore, tfidf) -> dict[str, list[str]]:
    k = settings.retrieve_k
    dense = store.dense_search(query, k)
    sparse = store.sparse_search(query, k)
    fused = fuse(dense, sparse, k=settings.rrf_k)
    reranked = rerank(query, fused[:k]) if fused else []
    return {
        "A": tfidf.rank(query, k),
        "B": [c.id for c, _ in dense],
        "C": [cand.chunk.id for cand in fused],
        "D": [cand.chunk.id for cand in reranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="Also run the answer-quality judge on arms A and D.")
    parser.add_argument("--api-key", help="Model provider key, required with --judge.")
    parser.add_argument("--judge-n", type=int, default=8, help="Number of answerable questions to judge.")
    args = parser.parse_args()

    from .baseline_tfidf import TfidfRetriever

    chunks = load_chunks()
    questions = load_questions()
    answerable = [q for q in questions if q["gold_answer_span"]]

    store = RetrievalStore()
    store.index(chunks)
    tfidf = TfidfRetriever(chunks)

    per_arm: dict[str, list[dict[str, float]]] = {"A": [], "B": [], "C": [], "D": []}
    for q in answerable:
        relevant = relevant_ids(q["gold_answer_span"], chunks)
        rankings = build_rankings(q["question"], store, tfidf)
        for arm, ranked in rankings.items():
            per_arm[arm].append(score_arm(ranked, relevant))

    results = {
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "chunk_tokens": settings.chunk_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
            "retrieve_k": settings.retrieve_k,
            "rrf_k": settings.rrf_k,
            "n_answerable": len(answerable),
            "n_total": len(questions),
            "chunks": len(chunks),
        },
        "arms": {
            "A": {"label": "TF-IDF baseline", **aggregate(per_arm["A"])},
            "B": {"label": "Dense only", **aggregate(per_arm["B"])},
            "C": {"label": "Hybrid (dense + BM25, RRF)", **aggregate(per_arm["C"])},
            "D": {"label": "Hybrid + rerank", **aggregate(per_arm["D"])},
        },
        "judge": None,
    }

    if args.judge:
        from . import judge

        if not args.api_key:
            parser.error("--judge requires --api-key")
        results["judge"] = judge.run(answerable[: args.judge_n], chunks, store, tfidf, api_key=args.api_key)

    payload = json.dumps(results, indent=2) + "\n"
    print(render_table(results))
    try:
        _RESULTS.write_text(payload)
        print(f"\nWrote {_RESULTS}")
    except OSError:
        # In a read-only mount the caller captures the JSON from stdout instead.
        print("\n<<<RESULTS_JSON>>>")
        print(payload)
        print("<<<END_RESULTS_JSON>>>")


def render_table(results: dict) -> str:
    cfg = results["config"]
    lines = [
        f"Retrieval ablation on {cfg['n_answerable']} answerable questions "
        f"({cfg['n_total']} total, {cfg['chunks']} chunks)",
        f"Embeddings {cfg['embedding_model']}, reranker {cfg['reranker_model']}, "
        f"chunk {cfg['chunk_tokens']} tokens, retrieve {cfg['retrieve_k']}",
        "",
        "| Arm | Retriever | hit@3 | hit@5 | recall@5 | MRR | nDCG@5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, data in results["arms"].items():
        lines.append(
            f"| {arm} | {data['label']} | {data['hit@3']:.2f} | {data['hit@5']:.2f} | "
            f"{data['recall@5']:.2f} | {data['mrr']:.2f} | {data['ndcg@5']:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
