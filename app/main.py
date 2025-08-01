from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import schemas
import services
import config

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Context Aware AI - Refactored",
    description="A simplified and efficient RAG-powered backend using FastAPI and OpenRouter.",
    version="2.0.0"
)


app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "static"), name="static")
# Setting up Jinja2 templates to serve the main HTML page
templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")


@app.post("/api/v1/index", response_model=schemas.IndexResponse)
async def index_context(document_request: schemas.DocumentRequest):
    """
    Receives text, clears the old index, chunks the new text,
    and stores its embeddings in the vector DB.
    """
    try:
        docs_added = services.index_document(document_request)
        return schemas.IndexResponse(
            message="Context has been successfully indexed.",
            documents_added=docs_added
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

@app.post("/api/v1/clear_index", response_model=schemas.GeneralResponse)
async def clear_context_index():
    """
    Clears all data from the vector database index.
    """
    try:
        services.clear_index()
        return schemas.GeneralResponse(message="Knowledge base has been successfully cleared.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e}")


@app.post("/api/v1/generate", response_model=schemas.ChatResponse)
async def generate_response(chat_request: schemas.ChatRequest):
    """
    Receives a prompt, retrieves relevant context from the vector DB,
    and returns an AI-generated response.
    """
    if not config.settings.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key is not configured on the server."
        )
    try:
        ai_message = await services.get_rag_response(chat_request)
        return schemas.ChatResponse(response=ai_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html page from the templates directory.
    """
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
