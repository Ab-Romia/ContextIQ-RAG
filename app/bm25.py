"""Lexical retrieval via BM25.

Dense retrieval is strong on meaning but can miss exact terms: an identifier, a product
name, a negation. BM25 is the opposite, and that complementarity is the point of running
both. bm25s implements BM25 over scipy sparse matrices with no C extension to build, so
it stays inside the toolchain-free image. We tokenize without a stemmer to keep the
build dependency-free; the guide notes stemming as a deliberate, documented omission.
"""

from __future__ import annotations

import bm25s

from .models import Chunk


class BM25Index:
    def __init__(self) -> None:
        self._engine: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if not chunks:
            self._engine = None
            return
        corpus = [chunk.text for chunk in chunks]
        tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
        engine = bm25s.BM25()
        engine.index(tokens, show_progress=False)
        self._engine = engine

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._engine is None or not self._chunks:
            return []
        k = min(k, len(self._chunks))
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        indices, scores = self._engine.retrieve(query_tokens, k=k, show_progress=False)
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            results.append((self._chunks[int(idx)], float(score)))
        return results
