"""Runtime configuration.

Every retrieval and generation parameter that affects answer quality lives here so
the pipeline reads as a set of explicit, tunable decisions rather than magic numbers
scattered across modules. Values can be overridden through environment variables
(see `Settings`), which is how the Hugging Face Space and the evaluation harness
exercise different configurations against the same code.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONTEXTIQ_", env_file=".env", extra="ignore")

    # Generation provider. The key is supplied per request by the caller and is never
    # stored server side. base_url points at an OpenAI compatible endpoint; OpenRouter
    # is the default because its free tier lets the live demo run without a credit card.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "deepseek/deepseek-r1-0528:free"
    max_output_tokens: int = 1024
    temperature: float = 0.2

    # Embeddings. bge-small-en-v1.5 is 384-dimensional and ships as a ~67 MB ONNX
    # model, which loads and runs comfortably on the Space's 2 vCPUs.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Cross-encoder reranker. MiniLM-L-6 is the default because it is ~80 MB and fast
    # enough to rerank a 50-candidate pool on CPU without blowing the request budget.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # Chunking. Token counts, not characters, so chunk size is meaningful to the model.
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # Retrieval depth. We pull a deliberately deep pool from each retriever before
    # fusion and reranking. A shallow pool is the most common reason a reranker looks
    # worthless in benchmarks: there is nothing for it to reorder.
    retrieve_k: int = 50
    rrf_k: int = 60
    final_k: int = 5

    # Context assembly. Hard ceiling on characters handed to the model so a large
    # document cannot push the prompt past the model's window or stall generation.
    max_context_chars: int = 8000


settings = Settings()
