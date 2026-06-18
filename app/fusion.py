"""Reciprocal Rank Fusion.

Dense cosine similarity and BM25 produce scores on incompatible scales, so we fuse by
rank, not by raw score. Each retriever contributes 1 / (k + rank) to a chunk's fused
score; a chunk that ranks well in either list rises, and a chunk that ranks well in
both rises further. k dampens the weight of the very top ranks so a single retriever
cannot dominate. This is the whole reason hybrid search beats either retriever alone,
and it needs no tuning beyond the constant k.
"""

from __future__ import annotations

from .models import Candidate, Chunk


def fuse(
    dense_ranked: list[tuple[Chunk, float]],
    sparse_ranked: list[tuple[Chunk, float]],
    *,
    k: int,
) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}

    def candidate_for(chunk: Chunk) -> Candidate:
        existing = candidates.get(chunk.id)
        if existing is None:
            existing = Candidate(chunk=chunk)
            candidates[chunk.id] = existing
        return existing

    for rank, (chunk, score) in enumerate(dense_ranked, start=1):
        cand = candidate_for(chunk)
        cand.dense_rank = rank
        cand.dense_score = score
        cand.rrf_score += 1.0 / (k + rank)

    for rank, (chunk, score) in enumerate(sparse_ranked, start=1):
        cand = candidate_for(chunk)
        cand.sparse_rank = rank
        cand.sparse_score = score
        cand.rrf_score += 1.0 / (k + rank)

    return sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
