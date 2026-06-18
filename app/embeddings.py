"""Dense embeddings via fastembed.

fastembed runs the bge-small model through ONNX Runtime, so there is no PyTorch in the
image and the model is small enough to load on a CPU Space in a couple of seconds. The
model is loaded lazily on first use and reused for the process lifetime; loading it at
import time would slow every cold start even for requests that never embed anything.

bge models were trained with an asymmetric setup: documents are embedded as-is, but a
query needs a short instruction prefix. fastembed's `query_embed` applies that prefix,
so we keep the two paths distinct rather than embedding queries like documents.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_model().query_embed(text))).tolist()


def warm() -> None:
    """Force the model to load. Called at build time so it is baked into the image."""
    embed_query("warmup")
