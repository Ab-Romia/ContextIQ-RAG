from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile
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
    version="2.2.0"  # Version updated for new feature
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")


def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Extract API key from header or use default."""
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
        "accepts_user_keys": True
    }


@app.post("/api/v1/test-api-key", response_model=schemas.ApiKeyTestResponse)
async def test_api_key_endpoint(api_key_request: schemas.ApiKeyRequest):
    """
    Test if the provided API key is valid.
    """
    try:
        result = await services.test_api_key(api_key_request.api_key)
        return schemas.ApiKeyTestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/task", response_model=schemas.TaskResponse)
async def execute_task(
        task_request: schemas.TaskRequest,
        x_api_key: Optional[str] = Header(None)
):
    """
    Executes a specific task (e.g., summarize, plan) based on the provided context
    using the provided or default API key.
    """
    try:
        api_key = get_api_key(x_api_key)
        result = await services.execute_task(task_request, api_key)
        return schemas.TaskResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html page from the templates directory.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for Hugging Face."""
    return {"status": "healthy", "message": "ContextIQ RAG is running!"}


@app.post("/api/v1/generate", response_model=schemas.ChatResponse)
async def generate_response(
        chat_request: schemas.ChatRequest,
        x_api_key: Optional[str] = Header(None)
):
    """
    Receives a prompt, retrieves relevant context from the vector DB,
    and returns an AI-generated response using the provided or default API key.
    """
    try:
        api_key = get_api_key(x_api_key)
        ai_message = await services.get_rag_response(chat_request, api_key)
        return schemas.ChatResponse(response=ai_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/clear_index", response_model=schemas.GeneralResponse)
async def clear_context_index(x_api_key: Optional[str] = Header(None)):
    """
    Clears all data from the vector database index.
    """
    try:
        services.clear_index()
        return schemas.GeneralResponse(message="Knowledge base has been successfully cleared.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e}")


@app.post("/api/v1/index", response_model=schemas.IndexResponse)
async def index_context(
        document_request: schemas.DocumentRequest,
        x_api_key: Optional[str] = Header(None)
):
    """
    Receives text, clears the old index, chunks the new text,
    and stores its embeddings in the vector DB.
    """
    try:
        # Validate API key access (but indexing doesn't require API calls)
        get_api_key(x_api_key)

        docs_added = services.index_document(document_request)
        return schemas.IndexResponse(
            message="Context has been successfully indexed.",
            documents_added=docs_added,
            extracted_text=document_request.context  # Return the provided text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")


# ✨ UPDATED: File Upload Endpoint now returns the extracted text
@app.post("/api/v1/index-file", response_model=schemas.IndexResponse)
async def index_file(
        x_api_key: Optional[str] = Header(None),
        file: UploadFile = File(...)
):
    """
    Receives a file (.txt, .pdf), extracts text, and indexes it.
    """
    try:
        # API key validation is still important
        get_api_key(x_api_key)

        # The service layer will handle the file processing
        docs_added, extracted_text = await services.process_and_index_file(file)

        return schemas.IndexResponse(
            message=f"Successfully indexed content from file: {file.filename}",
            documents_added=docs_added,
            extracted_text=extracted_text
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions to return proper status codes
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

