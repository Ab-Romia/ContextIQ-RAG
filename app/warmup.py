"""Build-time model warmup.

Run as `python -m app.warmup` in the Docker build so the embedding and reranker ONNX
models download into the image layer. Without this, the first request after every cold
start would re-download the models, which on a Space that sleeps after inactivity means a
slow and unreliable first impression.
"""

from __future__ import annotations

from . import embeddings, rerank


def main() -> None:
    embeddings.warm()
    rerank.warm()


if __name__ == "__main__":
    main()
