"""HTTP request and response models.

These Pydantic models are the only types that cross the API boundary. The retrieval
internals (Chunk, Candidate) stay in models.py; here we expose just what a client
needs to drive the pipeline and to render the trace.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to index.")
    source: str = Field("pasted-text", description="A label for where the text came from.")


class IndexResponse(BaseModel):
    source: str
    chunks_added: int
    chunks_total: int


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1, description="OpenAI compatible key, used for this request only.")
    model: str | None = Field(None, description="Override the default generation model.")
    rerank: bool | None = Field(None, description="Override whether the cross-encoder reranker runs.")


class CitationOut(BaseModel):
    marker: int
    source: str
    ordinal: int
    heading_path: list[str]
    text: str


class CandidateTrace(BaseModel):
    """One chunk's path through retrieval, flattened for the UI."""

    id: str
    source: str
    ordinal: int
    preview: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    rerank_score: float | None
    selected: bool


class PipelineTrace(BaseModel):
    """Everything the teaching UI needs to explain a single answer."""

    query: str
    rerank_enabled: bool
    dense_count: int
    sparse_count: int
    fused_count: int
    candidates: list[CandidateTrace]
    citations: list[CitationOut]
