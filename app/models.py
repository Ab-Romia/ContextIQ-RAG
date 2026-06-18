"""Internal data types that flow through the retrieval pipeline.

These are plain dataclasses rather than Pydantic models because they never cross the
HTTP boundary. Keeping them in one place lets every stage (chunking, retrieval,
fusion, reranking, assembly) speak the same vocabulary, and makes the pipeline trace
that the UI renders a direct projection of these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A unit of retrievable text and where it came from.

    `text` is shown to the reader and cited. `augmented_text` is what actually gets
    embedded: the same text prefixed with a short contextual header (document title
    and heading path) so an isolated chunk still carries the context a human would
    infer from its position in the document.
    """

    id: str
    text: str
    source: str
    ordinal: int
    heading_path: list[str] = field(default_factory=list)
    char_span: tuple[int, int] = (0, 0)
    augmented_text: str = ""

    def embedding_text(self) -> str:
        return self.augmented_text or self.text


@dataclass
class Candidate:
    """A chunk as it moves through retrieval, fusion, and reranking.

    Ranks and scores are filled in progressively so the final object records the whole
    journey: where each retriever placed it, what fusion did, and how the cross-encoder
    scored it. The UI reads exactly these fields to explain why a chunk won.
    """

    chunk: Chunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class Citation:
    """A source the answer is allowed to cite, identified by a stable marker like [1]."""

    marker: int
    text: str
    source: str
    ordinal: int
    heading_path: list[str] = field(default_factory=list)
