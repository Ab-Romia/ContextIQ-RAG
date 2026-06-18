"""The pipeline.

One object owns the retrieval store and runs the two flows: indexing a document, and
answering a query. The query flow returns the citations the model may use and a trace
that records how each candidate got there. Generation itself is streamed by the API layer
using the citations this returns, so the reader sees retrieval finish, then the answer
arrive token by token.
"""

from __future__ import annotations

from . import rerank as rerank_module
from .augment import augment
from .chunking import chunk_document
from .config import settings
from .fusion import fuse
from .models import Candidate, Citation
from .retrieval import RetrievalStore
from .schemas import CandidateTrace, CitationOut, PipelineTrace

_TRACE_LIMIT = 25


class Pipeline:
    def __init__(self) -> None:
        self.store = RetrievalStore()

    def index_text(self, text: str, source: str) -> int:
        chunks = chunk_document(
            text,
            source,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        augment(chunks)
        return self.store.index(chunks)

    def clear(self) -> None:
        self.store.clear()

    def count(self) -> int:
        return self.store.count()

    def retrieve(self, query: str, *, rerank_enabled: bool) -> tuple[list[Citation], PipelineTrace]:
        dense = self.store.dense_search(query, settings.retrieve_k)
        sparse = self.store.sparse_search(query, settings.retrieve_k)
        fused = fuse(dense, sparse, k=settings.rrf_k)

        use_rerank = rerank_enabled and settings.reranker_enabled and bool(fused)
        if use_rerank:
            ranked = rerank_module.rerank(query, fused[: settings.retrieve_k])
        else:
            ranked = fused

        selected = ranked[: settings.final_k]
        selected_ids = {c.chunk.id for c in selected}

        citations = [
            Citation(
                marker=i + 1,
                text=c.chunk.text,
                source=c.chunk.source,
                ordinal=c.chunk.ordinal,
                heading_path=c.chunk.heading_path,
            )
            for i, c in enumerate(selected)
        ]

        trace = PipelineTrace(
            query=query,
            rerank_enabled=use_rerank,
            dense_count=len(dense),
            sparse_count=len(sparse),
            fused_count=len(fused),
            candidates=[self._candidate_trace(c, selected_ids) for c in ranked[:_TRACE_LIMIT]],
            citations=[
                CitationOut(
                    marker=c.marker,
                    source=c.source,
                    ordinal=c.ordinal,
                    heading_path=c.heading_path,
                    text=c.text,
                )
                for c in citations
            ],
        )
        return citations, trace

    @staticmethod
    def _candidate_trace(candidate: Candidate, selected_ids: set[str]) -> CandidateTrace:
        chunk = candidate.chunk
        preview = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
        return CandidateTrace(
            id=chunk.id,
            source=chunk.source,
            ordinal=chunk.ordinal,
            preview=preview,
            dense_rank=candidate.dense_rank,
            sparse_rank=candidate.sparse_rank,
            rrf_score=round(candidate.rrf_score, 6),
            rerank_score=None if candidate.rerank_score is None else round(candidate.rerank_score, 4),
            selected=chunk.id in selected_ids,
        )
