"""Answer-quality judge (opt-in).

Retrieval metrics measure whether the right chunk was found. They say nothing about
whether the written answer is faithful to it. This judge fills that gap with a single
model call per answer, scoring two things:

- faithfulness: is every claim in the answer supported by the retrieved context, and
  only the context? The judge sees the context and the answer, never the reference, so
  it cannot reward an answer for being right by luck.
- abstention: for questions the document cannot answer, did the answer correctly decline
  rather than invent something?

The judge model is the same free model used for generation, which means it shares that
model's biases, including a mild preference for its own style. The results are read as a
relative signal between arms on a small sample, not as an absolute grade. The guide says
so plainly.
"""

from __future__ import annotations

import json

from openai import OpenAI

from app.config import settings
from app.fusion import fuse
from app.generate import build_messages
from app.rerank import rerank

_JUDGE_PROMPT = (
    "You are grading whether an answer is faithful to its sources. "
    "You are given SOURCES and an ANSWER. Decide if every factual claim in the answer is "
    "supported by the sources, using only the sources. Reply with strict JSON: "
    '{{"faithful": true|false, "reason": "<one sentence quoting the deciding span>"}}.\n\n'
    "SOURCES:\n{context}\n\nANSWER:\n{answer}"
)


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=settings.openrouter_base_url)


def _generate(client: OpenAI, query: str, context_chunks: list[str]) -> str:
    citations = [type("C", (), {"marker": i + 1, "text": t})() for i, t in enumerate(context_chunks)]
    messages = build_messages(query, citations)
    out = client.chat.completions.create(
        model=settings.default_model,
        messages=messages,
        temperature=0,
        max_tokens=settings.max_output_tokens,
    )
    return out.choices[0].message.content or ""


def _judge_faithful(client: OpenAI, answer: str, context_chunks: list[str]) -> bool:
    context = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(context_chunks))
    out = client.chat.completions.create(
        model=settings.default_model,
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(context=context, answer=answer)}],
        temperature=0,
        max_tokens=200,
    )
    raw = out.choices[0].message.content or "{}"
    try:
        start, end = raw.find("{"), raw.rfind("}")
        return bool(json.loads(raw[start : end + 1]).get("faithful", False))
    except (ValueError, json.JSONDecodeError):
        return False


def _context_for_arm(arm: str, query: str, store, tfidf, chunks_by_id) -> list[str]:
    k = settings.retrieve_k
    if arm == "A":
        ids = tfidf.rank(query, k)[: settings.final_k]
        return [chunks_by_id[i].text for i in ids]
    dense = store.dense_search(query, k)
    sparse = store.sparse_search(query, k)
    fused = fuse(dense, sparse, k=settings.rrf_k)
    reranked = rerank(query, fused[:k])
    return [c.chunk.text for c in reranked[: settings.final_k]]


def run(questions, chunks, store, tfidf, *, api_key: str) -> dict:
    client = _client(api_key)
    chunks_by_id = {c.id: c for c in chunks}
    scores = {"A": [], "D": []}
    for q in questions:
        for arm in ("A", "D"):
            context = _context_for_arm(arm, q["question"], store, tfidf, chunks_by_id)
            answer = _generate(client, q["question"], context)
            scores[arm].append(1.0 if _judge_faithful(client, answer, context) else 0.0)
    return {
        "metric": "faithfulness (fraction supported by retrieved context)",
        "n": len(questions),
        "A": round(sum(scores["A"]) / len(scores["A"]), 3) if scores["A"] else None,
        "D": round(sum(scores["D"]) / len(scores["D"]), 3) if scores["D"] else None,
    }
