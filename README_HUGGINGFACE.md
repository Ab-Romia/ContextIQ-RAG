---
title: ContextIQ
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: docker
pinned: true
license: mit
app_port: 7860
models:
  - BAAI/bge-small-en-v1.5
  - Xenova/ms-marco-MiniLM-L-6-v2
---

# ContextIQ

A hybrid-retrieval RAG worked example that runs on CPU.

Index a document, then ask a question. ContextIQ retrieves the passages that answer it and writes a grounded, cited answer. The interface shows the retrieval trace, so you can see how each candidate ranked under dense search, BM25, fusion, and reranking, and which passages were sent to the model.

## Using it

1. Paste text or upload a file, then index it. Retrieval reads only from what you index.
2. Enter an OpenRouter API key. It stays in your browser and is used only to write the answer; the default model is free. Retrieval and the trace work without a key.
3. Ask a question. The answer streams in with citation markers that link back to their source passages.

## How it works

Dense semantic search (`bge-small`, run through ONNX) and lexical BM25 retrieve in parallel. Their rankings are combined with reciprocal rank fusion, then a cross-encoder reranker reorders the top candidates by reading each one against the query. The model answers only from the reranked passages and is told to abstain when they do not contain the answer.

The whole pipeline runs on CPU with no GPU and no PyTorch. The retrieval design, an evaluation harness, and the reasoning behind each stage are in the source repository: [github.com/Ab-Romia/ContextIQ-RAG](https://github.com/Ab-Romia/ContextIQ-RAG).

## Notes

The index is held in memory and is cleared when the Space restarts after sleeping. This is a single-session demo, not durable storage.
