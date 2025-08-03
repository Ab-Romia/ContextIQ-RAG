from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import os
import schemas
import services
import config
from typing import Optional

# Get the base directory (works both locally and on Hugging Face)
if os.path.exists("/app"):  # Hugging Face environment
    BASE_DIR = Path("/app")
else:  # Local environment
    BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="ContextIQ RAG - Intelligent Context-Aware Assistant",
    description="A sophisticated RAG-powered backend using FastAPI and OpenRouter.",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")


def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Extract API key from header or use a server default."""
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()

    # Fall back to server default if available
    if config.settings.OPENROUTER_API_KEY:
        return config.settings.OPENROUTER_API_KEY

    raise HTTPException(
        status_code=400,
        detail="No API key provided. Please provide your OpenRouter API key via the X-API-Key header."
    )


@app.get("/debug")
async def debug_config():
    """Debug endpoint to check configuration."""
    return {
        "api_key_configured": bool(config.settings.OPENROUTER_API_KEY),
        "api_key_length": len(config.settings.OPENROUTER_API_KEY) if config.settings.OPENROUTER_API_KEY else 0,
        "model_name": config.settings.MODEL_NAME,
        "server_has_fallback_key": bool(config.settings.OPENROUTER_API_KEY),
        "require_user_api_key": config.settings.REQUIRE_USER_API_KEY
    }


@app.post("/api/v1/index", response_model=schemas.IndexResponse)
async def index_context(document_request: schemas.DocumentRequest, api_key: str = Depends(get_api_key)):
    """
    Receives text, chunks the new text, and stores its embeddings in the vector DB.
    """
    try:
        docs_added = services.index_document(document_request, api_key)
        return schemas.IndexResponse(
            message=f"Successfully indexed {docs_added} document chunks.",
            chunks_added=docs_added
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=schemas.ChatResponse)
async def chat_with_context(chat_request: schemas.ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Retrieves context from the vector DB and generates a response.
    """
    try:
        response_text = await services.get_rag_response(chat_request, api_key)
        return schemas.ChatResponse(response=response_text)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/task")
async def perform_task(task_request: schemas.TaskRequest, api_key: str = Depends(get_api_key)):
    """
    Performs a specific task on the indexed documents (e.g., summarization).
    """
    try:
        response = await services.perform_task_on_documents(task_request, api_key)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Serves the main HTML page for the frontend.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/v1/clear-index", response_model=schemas.IndexResponse)
async def clear_index():
    """
    Clears all indexed documents from the vector database.
    """
    services.clear_index()
    return schemas.IndexResponse(message="Successfully cleared all documents from the index.", chunks_added=0)


@app.post("/api/v1/test_api_key", response_model=schemas.ApiKeyTestResponse)
async def test_api_key_endpoint(request: schemas.ApiKeyRequest):
    """
    Tests an OpenRouter API key provided by the user.
    """
    response = await services.test_api_key(request.api_key)
    return schemas.ApiKeyTestResponse(**response)