"""Application entrypoint.

Builds the FastAPI app, mounts the static assets and template with paths resolved
relative to this package (so it runs the same locally and in the container), and creates
the single Pipeline instance the request handlers share. Models are loaded at build time
and warmed again on startup so the first real request does not pay the load cost.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import router
from .pipeline import Pipeline

_BASE = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_BASE / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline()
    yield


app = FastAPI(title="ContextIQ", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})
