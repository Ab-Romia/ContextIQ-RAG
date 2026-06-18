"""Retrieval metrics.

Pure functions over a ranked list of chunk ids and the set of relevant ids for a query.
No dependencies, so they run anywhere and are easy to unit test. All four answer a
slightly different question:

- hit_rate@k: did at least one relevant chunk make the top k? (can the model see it at all)
- recall@k:   what fraction of the relevant chunks made the top k?
- MRR:        how high did the first relevant chunk rank? (rewards putting it first)
- nDCG@k:     a rank-discounted score, relevance weighted by position.

Relevance here is binary: a chunk is relevant if it contains the gold answer span.
"""

from __future__ import annotations

import math


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for cid in ranked[:k] if cid in relevant)
    return found / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
