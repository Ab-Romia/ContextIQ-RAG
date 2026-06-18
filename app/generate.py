"""Answer generation.

The model is given only the reranked context and is told to ground every claim in it and
to cite sources by their marker. If the context does not answer the question, it is told
to say so rather than fall back on training knowledge. That instruction is deliberate: a
RAG system that quietly answers from memory when retrieval fails is indistinguishable
from a plain chatbot and is the harder failure to catch. Generation streams token by
token so the reader sees progress immediately, which also keeps the request well under
the Space's proxy timeout.

The provider is any OpenAI-compatible endpoint; OpenRouter is the default. The API key
arrives with the request and is used to construct a client for that request only. It is
never stored, logged, or written to disk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import settings
from .models import Citation

SYSTEM_PROMPT = (
    "You are a careful assistant that answers strictly from the provided sources. "
    "Use only the information in the numbered sources below. Cite every claim with its "
    "source marker in square brackets, for example [1] or [2][3]. If the sources do not "
    "contain the answer, say that the provided context does not cover it and stop. Do "
    "not use outside knowledge and do not guess."
)


def build_messages(query: str, citations: list[Citation]) -> list[dict[str, str]]:
    blocks = [f"[{c.marker}] {c.text}" for c in citations]
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    user = f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer using only the sources above and cite them."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def stream_answer(
    query: str,
    citations: list[Citation],
    *,
    api_key: str,
    model: str | None = None,
) -> AsyncIterator[str]:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        # OpenRouter uses these to attribute and route requests; they are optional but
        # recommended and can help free-tier routing.
        default_headers={
            "HTTP-Referer": "https://github.com/Ab-Romia/ContextIQ-RAG",
            "X-Title": "ContextIQ",
        },
    )
    stream = await client.chat.completions.create(
        model=model or settings.default_model,
        messages=build_messages(query, citations),
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
