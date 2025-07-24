from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app import schemas
from app import services

# Application setup
app = FastAPI(
    title="ContextIQ API with RAG",
    description="A backend for a RAG-powered context-aware assistant."
)

# Determine base directory and mount static/templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# --- RAG API Endpoints ---

@app.post("/api/v1/index", response_model=schemas.IndexResponse)
async def index_context(document_request: schemas.DocumentRequest):
    """
    Receives text, chunks it, and stores its embeddings in the vector DB.
    """
    try:
        docs_added = services.index_document(document_request)
        return schemas.IndexResponse(
            message="Context has been successfully indexed.",
            documents_added=docs_added
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@app.post("/api/v1/generate", response_model=schemas.ChatResponse)
async def generate_response(chat_request: schemas.ChatRequest):
    """
    Receives a prompt, retrieves relevant context from the vector DB,
    and returns an AI-generated response.
    """
    try:
        ai_message = await services.get_rag_response(chat_request)
        return schemas.ChatResponse(response=ai_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Frontend Route ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)