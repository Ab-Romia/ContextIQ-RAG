"""The hybrid retrieval store.

Holds the indexed chunks once and exposes the two retrievers that read from them: dense
vectors in Chroma and lexical BM25. Chroma is configured with no embedding function of
its own because we embed with fastembed and hand it precomputed vectors; letting Chroma
embed would pull in a second model and a second opinion about what the vectors mean.

The store is in memory and additive. Indexing a document adds its chunks; indexing the
same source again replaces just that source. There is no implicit wipe of the whole
store on every upload, which was the original design's defining bug. State lives for the
process lifetime only, which on a free Space means until it sleeps. That is a deliberate,
documented limit, not a persistence layer pretending to be durable.
"""

from __future__ import annotations

import chromadb

from . import embeddings
from .bm25 import BM25Index
from .models import Chunk

_COLLECTION = "contextiq"


class RetrievalStore:
    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = BM25Index()
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        for source in {chunk.source for chunk in chunks}:
            self._remove_source(source)

        vectors = embeddings.embed_documents([chunk.embedding_text() for chunk in chunks])
        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source, "ordinal": chunk.ordinal} for chunk in chunks],
        )
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

        self._bm25.build(list(self._chunks.values()))
        return len(chunks)

    def dense_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        result = self._collection.query(
            query_embeddings=[embeddings.embed_query(query)],
            n_results=min(k, len(self._chunks)),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        hits: list[tuple[Chunk, float]] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = self._chunks.get(chunk_id)
            if chunk is not None:
                hits.append((chunk, 1.0 - float(distance)))  # cosine distance -> similarity
        return hits

    def sparse_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        return self._bm25.search(query, k)

    def clear(self) -> None:
        if self._chunks:
            self._collection.delete(ids=list(self._chunks))
        self._chunks.clear()
        self._bm25.build([])

    def count(self) -> int:
        return len(self._chunks)

    def _remove_source(self, source: str) -> None:
        stale = [cid for cid, chunk in self._chunks.items() if chunk.source == source]
        if not stale:
            return
        self._collection.delete(ids=stale)
        for cid in stale:
            del self._chunks[cid]
