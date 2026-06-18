"""HTTP API.

Indexing and retrieval are CPU-bound and synchronous, so they run in a worker thread to
keep the event loop free. Generation is streamed over Server-Sent Events: the client
first receives a `trace` event describing what retrieval did, then a sequence of `token`
events as the answer is written, then `done`. Errors during generation arrive as an
`error` event rather than a broken stream, and the API key is never echoed back.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from . import generate, ingest
from .schemas import GenerateRequest, IndexResponse, IndexTextRequest

router = APIRouter()


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/index", response_model=IndexResponse)
async def index_text(request: Request, body: IndexTextRequest) -> IndexResponse:
    pipeline = _pipeline(request)
    added = await run_in_threadpool(pipeline.index_text, body.text, body.source)
    return IndexResponse(source=body.source, chunks_added=added, chunks_total=pipeline.count())


@router.post("/api/index-file", response_model=IndexResponse)
async def index_file(request: Request, file: UploadFile = File(...)) -> JSONResponse | IndexResponse:
    content = await file.read()
    try:
        text = ingest.extract(file.filename or "upload", content)
    except ingest.UnsupportedFile as exc:
        return JSONResponse(status_code=415, content={"detail": str(exc)})
    except ingest.ExtractionError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    pipeline = _pipeline(request)
    source = file.filename or "upload"
    added = await run_in_threadpool(pipeline.index_text, text, source)
    return IndexResponse(source=source, chunks_added=added, chunks_total=pipeline.count())


@router.post("/api/clear")
async def clear(request: Request) -> dict[str, str]:
    _pipeline(request).clear()
    return {"status": "cleared"}


@router.post("/api/generate")
async def generate_answer(request: Request, body: GenerateRequest) -> EventSourceResponse:
    pipeline = _pipeline(request)
    rerank_enabled = True if body.rerank is None else body.rerank

    citations, trace = await run_in_threadpool(
        pipeline.retrieve, body.query, rerank_enabled=rerank_enabled
    )

    async def events():
        yield {"event": "trace", "data": json.dumps(trace.model_dump())}
        try:
            async for delta in generate.stream_answer(
                body.query, citations, api_key=body.api_key, model=body.model
            ):
                yield {"event": "token", "data": json.dumps({"text": delta})}
        except Exception as exc:  # surface provider/auth errors without leaking the key
            yield {"event": "error", "data": json.dumps({"detail": _safe_error(str(exc))})}
            return
        yield {"event": "done", "data": "{}"}

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return EventSourceResponse(events(), headers=headers)


def _safe_error(message: str) -> str:
    if "401" in message or "Unauthorized" in message or "invalid_api_key" in message:
        return "The model provider rejected the API key. Check the key and try again."
    if "429" in message:
        return "The model provider is rate limiting this key. Wait a moment and retry."
    return "The model provider could not complete the request. Try again or switch models."
