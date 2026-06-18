"""Cross-encoder reranking.

Retrieval scores a query and a chunk independently, then compares vectors. A
cross-encoder instead reads the query and the chunk together and scores their actual
relevance, which is far more accurate but far too slow to run over a whole corpus. So we
use it where it pays off: reranking the fused top candidates down to the handful that go
to the model. This is the single biggest quality lever in the pipeline, and the reason
the candidate pool is deep. Reranking three results changes nothing; reranking fifty
finds the two that retrieval ranked seventh and ninth.

MiniLM-L-6 is the default because it reranks a fifty-chunk pool on CPU in a fraction of a
second. The model loads lazily and is reused for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings
from .models import Candidate


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.reranker_model)


def rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    scores = list(_model().rerank(query, [c.chunk.text for c in candidates]))
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


def warm() -> None:
    """Force the model to load. Called at build time so it is baked into the image."""
    rerank("warmup", [])
    list(_model().rerank("warmup", ["warmup passage"]))
