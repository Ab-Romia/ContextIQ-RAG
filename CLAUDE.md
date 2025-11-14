# CLAUDE.md - AI Assistant Guide for ContextIQ RAG

## Project Overview

**ContextIQ** is a sophisticated Retrieval-Augmented Generation (RAG) application that enables intelligent, context-aware interactions with documents. The application supports dual AI provider integration (OpenAI and OpenRouter), processes 11+ file formats, and uses ChromaDB with custom TF-IDF embeddings for fast vector search.

**Version:** 2.2.0
**License:** MIT
**Author:** Abdelrahman Abouroumia (Ab-Romia)
**Python Version:** 3.8+
**Framework:** FastAPI

---

## Repository Structure

```
ContextIQ-RAG/
├── app/                        # Main application package
│   ├── __init__.py            # Package initialization
│   ├── main.py                # FastAPI application and API endpoints
│   ├── config.py              # Configuration management with Pydantic
│   ├── schemas.py             # Pydantic data models for API requests/responses
│   ├── services.py            # Business logic and file processing
│   └── rag_setup.py           # RAG implementation, ChromaDB, and LLM clients
├── static/                     # Static frontend assets
│   └── app.js                 # Frontend JavaScript (36KB)
├── templates/                  # HTML templates
│   └── index.html             # Main web interface (19KB)
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── .gitignore                 # Git ignore patterns
├── LICENSE                     # MIT License
├── README.md                   # User-facing documentation
└── README_HUGGINGFACE.md      # Hugging Face deployment guide
```

---

## Architecture Overview

### Technology Stack

**Backend:**
- **FastAPI** (0.116+) - Modern async web framework
- **Uvicorn** (0.35+) - ASGI server
- **Pydantic** (2.11+) - Data validation and settings management
- **ChromaDB** (1.0+) - In-memory vector database
- **Scikit-learn** (1.7+) - Custom TF-IDF embeddings
- **OpenAI SDK** (1.62+) - GPT models integration
- **Requests** (2.32+) - HTTP client for OpenRouter API

**File Processing:**
- **PyMuPDF (fitz)** (1.25+) - PDF processing
- **python-docx** (1.1+) - Word documents
- **python-pptx** (1.0+) - PowerPoint files
- **Pandas** (2.2+) & **OpenPyXL** (3.1+) - Excel/CSV
- **BeautifulSoup** (4.12+) - HTML parsing
- **striprtf** (0.0.26) - RTF file support
- **Markdown** (3.7) - Markdown processing

**Frontend:**
- **Vanilla JavaScript** - No framework dependencies
- **Tailwind CSS** - Utility-first styling
- **Marked.js** - Markdown rendering in responses

### System Architecture

```
┌────────────────────────────────────────────┐
│     Frontend (HTML/JS/Tailwind)            │
│  • Provider Selection                      │
│  • File Upload & Text Input                │
│  • Real-time Chat Interface                │
│  • LocalStorage API Key Management         │
└─────────────┬──────────────────────────────┘
              │ REST API (JSON)
┌─────────────▼──────────────────────────────┐
│        FastAPI Backend (Python)            │
│  • Request Validation (Pydantic)           │
│  • Multi-Provider LLM Support              │
│  • File Processing Pipeline                │
│  • Response Caching (10 min)               │
└──────┬──────────────────┬──────────────────┘
       │                  │
┌──────▼────────┐  ┌──────▼─────────────────┐
│   ChromaDB    │  │   LLM Providers        │
│ Vector Store  │  │  • OpenAI API          │
│ (TF-IDF)      │  │  • OpenRouter API      │
│ In-Memory     │  │    (200+ models)       │
└───────────────┘  └────────────────────────┘
```

---

## Core Components

### 1. Configuration (`app/config.py`)

**Key Settings:**
- `Settings` class using Pydantic for environment variable management
- Dual provider configuration (OpenAI and OpenRouter)
- Token limits per task type:
  - `MAX_TOKENS_CHAT`: 4000 (Q&A)
  - `MAX_TOKENS_SUMMARIZE`: 3000 (Summaries)
  - `MAX_TOKENS_PLAN`: 5000 (Action plans)
  - `MAX_TOKENS_CREATIVE`: 6000 (Creative writing)
  - `MAX_TOKENS_TEST`: 50 (API key validation)
- Context limits:
  - `MAX_CONTEXT_LENGTH_CHAT`: 12000
  - `MAX_CONTEXT_LENGTH_TASK`: 16000
  - `MAX_CHUNKS_RETRIEVE`: 5
  - `CHUNK_SIZE`: 500 chars
  - `CHUNK_OVERLAP`: 100 chars
- Timeout configuration:
  - `REQUEST_TIMEOUT_BASE`: 120 seconds
  - `REQUEST_TIMEOUT_PER_1K_TOKENS`: 4 seconds per 1000 tokens

**Important Functions:**
- `validate_api_key()` - Validates OpenRouter/OpenAI API key format
- `detect_provider_from_key()` - Auto-detects provider from key prefix
- `get_max_tokens_for_task()` - Returns appropriate token limit for task type
- `get_timeout_for_tokens()` - Calculates dynamic timeout based on token count

**Environment Variables:**
```bash
OPENROUTER_API_KEY=""        # Optional server default
OPENAI_API_KEY=""             # Optional server default
OPENROUTER_MODEL="deepseek/deepseek-r1-0528:free"
OPENAI_MODEL="gpt-4o-mini"
REQUIRE_USER_API_KEY=True     # Force user to provide API key
```

### 2. API Routes (`app/main.py`)

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve main interface (index.html) |
| GET | `/health` | Health check for deployment monitoring |
| GET | `/debug` | Debug endpoint for configuration status |
| POST | `/api/v1/test-api-key` | Validate API key for OpenRouter or OpenAI |
| POST | `/api/v1/index` | Index text context into vector DB |
| POST | `/api/v1/index-file` | Upload and index file (11+ formats) |
| POST | `/api/v1/generate` | Generate RAG response with conversation history |
| POST | `/api/v1/task` | Execute specialized tasks (summarize/plan/creative) |
| POST | `/api/v1/clear_index` | Clear all documents from vector DB |

**API Key Handling:**
- User API keys sent via `X-API-Key` header
- Falls back to server default if available
- Validates key format before API calls
- Auto-detects provider (OpenRouter vs OpenAI)

### 3. Business Logic (`app/services.py`)

**File Processing Pipeline:**

Supported formats with dedicated processors:
- `.txt` - Multi-encoding support (UTF-8, Latin-1, CP1252)
- `.pdf` - Multi-page extraction with PyMuPDF
- `.docx` - Paragraphs and tables extraction
- `.pptx` - Slide-by-slide text extraction
- `.xlsx`/`.xls` - Structured data with Pandas (limit 100 rows per sheet)
- `.csv` - Multi-encoding support (limit 200 rows)
- `.json` - Intelligent parsing and formatting
- `.xml` - Element tree parsing with attributes
- `.html`/`.htm` - Clean text extraction (removes scripts/styles)
- `.md`/`.markdown` - Markdown to HTML conversion
- `.rtf` - RTF to plain text conversion

**Key Functions:**

```python
async def process_and_index_file(file: UploadFile) -> Tuple[int, str]
    # Processes file and returns (docs_added, extracted_text)

def _create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> list
    # Smart chunking with sentence/paragraph boundary detection

def index_document(request_data: DocumentRequest) -> int
    # Clears old index, creates chunks, stores in ChromaDB

async def get_rag_response(request_data: ChatRequest, api_key: str) -> str
    # Enhanced RAG pipeline with conversation history
    # Steps: cache check → query expansion → chunk retrieval →
    #        deduplication → context assembly → LLM generation

async def execute_task(request_data: TaskRequest, api_key: str) -> str
    # Execute specialized tasks: summarize, plan, creative writing
```

**Response Caching:**
- 10-minute TTL cache (`CACHE_EXPIRATION_SECONDS = 600`)
- Cache key includes API key + conversation history hash + prompt
- Automatic expiration and cleanup

**Query Enhancement:**
- Query expansion with key term extraction
- Multiple retrieval passes with deduplication
- Smart ranking based on chunk metadata (position, length)

### 4. RAG Implementation (`app/rag_setup.py`)

**Custom TF-IDF Embeddings:**

```python
class TFIDFEmbeddingFunction:
    # 384-dimensional vectors
    # N-gram range: (1, 2)
    # Stop words: English
    # Fallback: MD5 hash-based embeddings for graceful degradation
```

**Vector Database:**
- **ChromaDB** with in-memory storage (Hugging Face compatible)
- Collection name: `"context_aware_collection"`
- Metadata per chunk:
  - `chunk_index` - Position in document
  - `timestamp` - Indexing time
  - `chunk_length` - Character count
  - `position` - "start", "middle", or "end"
  - `total_chunks` - Total chunks in document

**LLM Clients:**

```python
class OpenRouterLLM:
    # HTTP-based client using requests library
    # Retry logic: 3 attempts with exponential backoff
    # Dynamic timeout calculation based on max_tokens
    # Streaming: disabled (full response)

class OpenAILLM:
    # SDK-based client using official OpenAI library
    # Retry logic: 3 attempts
    # Rate limit handling with exponential backoff
    # Temperature: 0.7, Top-p: 0.9

def create_llm(api_key: str, provider: Optional[str] = None)
    # Factory function for creating appropriate LLM client
    # Auto-detects provider from API key prefix
```

**Default Model Initialization:**
- Attempts OpenRouter first if `OPENROUTER_API_KEY` is set
- Falls back to OpenAI if `OPENAI_API_KEY` is set
- Creates `DummyLLM` for graceful degradation if both fail

### 5. Data Models (`app/schemas.py`)

**Request Models:**
```python
ConversationMessage      # role: str, content: str
DocumentRequest          # context: str (min 10 chars)
ChatRequest             # prompt: str, conversation_history: Optional[List[ConversationMessage]]
TaskRequest             # context: str, task_type: str, prompt: Optional[str]
ApiKeyRequest           # api_key: str, provider: Optional[str]
```

**Response Models:**
```python
ChatResponse            # response: str
TaskResponse            # result: str
IndexResponse           # message: str, documents_added: int, extracted_text: Optional[str]
GeneralResponse         # message: str
ApiKeyTestResponse      # valid: bool, message: str, model_info: Optional[dict]
```

---

## Development Workflows

### Running Locally

```bash
# 1. Clone repository
git clone https://github.com/Ab-Romia/ContextIQ-RAG.git
cd ContextIQ-RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set environment variables
export OPENROUTER_API_KEY="sk-or-xxxxx"  # or
export OPENAI_API_KEY="sk-xxxxx"

# 4. Run application
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 7860

# Access at http://localhost:7860
```

### Docker Deployment

```bash
# Build image
docker build -t contextiq .

# Run container
docker run -p 7860:7860 contextiq

# With environment variables
docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY="sk-or-xxxxx" \
  contextiq
```

**Dockerfile Notes:**
- Base image: `python:3.10-slim`
- Installs build-essential and curl
- Creates `/tmp/chroma_db` with 777 permissions
- Health check: `curl -f http://localhost:7860/health`
- Working directory: `/app`
- Port: 7860

### Hugging Face Spaces

The application is optimized for Hugging Face Spaces deployment:
1. Set Space SDK to "Docker"
2. Upload repository files
3. Optionally add `OPENROUTER_API_KEY` or `OPENAI_API_KEY` as secrets
4. Deploy (automatically uses Docker)

**Live Demo:** https://huggingface.co/spaces/Ab-Romia/Context-Aware-AI

---

## Code Conventions and Best Practices

### Logging

All modules use structured logging:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 STARTING PROCESS")
logger.info("=" * 80)
logger.info(f"📏 Input length: {len(input)}")
logger.warning("⚠️  Warning message")
logger.error("❌ Error message")
```

**Logging Conventions:**
- Use emojis for visual scanning: 🚀 ✅ ❌ ⚠️ 📊 🔍 💾 🔧 🔑
- Section headers with `=` separators (80 chars)
- Subsections with `-` separators (60 chars)
- Log input/output sizes for debugging
- Log timing information for performance monitoring

### Error Handling

**FastAPI Exception Handling:**
```python
try:
    result = await some_operation()
    return result
except HTTPException as e:
    raise e  # Re-raise HTTP exceptions
except Exception as e:
    logger.error(f"❌ Error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
```

**LLM Error Handling:**
- Retry logic: 3 attempts with exponential backoff
- Rate limit detection with cooldown
- Timeout handling with dynamic calculation
- Authentication error detection (401, 403)
- Graceful degradation with informative error messages

### Async Patterns

Use async/await for I/O operations:
```python
async def process_and_index_file(file: UploadFile):
    file_content = await file.read()  # Async file read
    text = await _process_pdf_file(file_content)
    # ...

async def get_rag_response(request_data: ChatRequest, api_key: str):
    cached = _get_cached_response(cache_key)  # Sync cache lookup
    chunks = await _retrieve_chunks_async(prompt)  # Async DB query
    response = await _generate_response_async(prompt, api_key)  # Async LLM call
```

**Executor Pattern for Blocking Operations:**
```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(
    None,
    functools.partial(rag_setup.collection.query, query_texts=[prompt], n_results=n)
)
```

### Testing

When testing changes:
1. Test API key validation endpoint first
2. Test file upload with each supported format
3. Test RAG pipeline with small and large contexts
4. Test conversation history functionality
5. Test all task types (summarize, plan, creative)
6. Check logs for performance metrics
7. Verify error handling with invalid inputs

### Performance Optimization

**Context Truncation:**
- Max context length enforced before LLM calls
- Intelligent truncation preserving question/instructions
- Warnings logged when truncation occurs

**Chunking Strategy:**
- Smart boundary detection (sentence > paragraph > word)
- Overlapping chunks for context continuity
- Minimum chunk size: 20 characters

**Caching:**
- Response cache with conversation history hashing
- 10-minute TTL to balance freshness and performance
- Cache key includes API key for multi-user isolation

**Timeout Management:**
- Dynamic timeout based on token count and prompt length
- Base: 120s + (max_tokens / 100) + (prompt_length / 1000)
- Maximum: 600 seconds (10 minutes)

---

## Important Considerations for AI Assistants

### When Modifying Code

1. **Maintain Logging Standards:**
   - Always add comprehensive logging to new functions
   - Use emojis consistently with existing patterns
   - Log input/output sizes and timing information

2. **Preserve Error Handling:**
   - Keep retry logic for LLM calls
   - Maintain graceful degradation patterns
   - Always log errors with `exc_info=True` for stack traces

3. **API Compatibility:**
   - Don't change Pydantic model field names (breaks API contract)
   - Maintain backward compatibility with frontend
   - Update version number if API changes

4. **File Processing:**
   - Test new file formats with sample files
   - Handle encoding errors gracefully
   - Limit output size to prevent memory issues
   - Log processing statistics (chars, words, pages)

5. **Security:**
   - Never log full API keys (use `[:10]` or `[:20]` preview)
   - Validate all user inputs
   - Sanitize file content before processing
   - Don't commit `.env` files or secrets

6. **Configuration Changes:**
   - Update `app/config.py` for new settings
   - Document environment variables in this file
   - Maintain default values for all settings
   - Use Pydantic validation for type safety

### Common Tasks

**Adding a New File Format:**
1. Add import with try/except in `app/services.py`
2. Create `async def _process_XXX_file(file_content: bytes) -> str`
3. Add file extension to `process_and_index_file()` elif chain
4. Update supported_extensions list
5. Test with sample file

**Adding a New Task Type:**
1. Add task type constant to `app/config.py` (e.g., `MAX_TOKENS_NEWTASK`)
2. Update `get_max_tokens_for_task()` mapping
3. Add task logic to `execute_task()` in `app/services.py`
4. Update frontend task selection dropdown in `templates/index.html`

**Adding a New LLM Provider:**
1. Create new LLM class in `app/rag_setup.py` (similar to `OpenRouterLLM`)
2. Update `create_llm()` factory function
3. Add provider configuration to `app/config.py`
4. Update `detect_provider_from_key()` if needed
5. Test API key validation

**Modifying Chunking Strategy:**
1. Update `_create_overlapping_chunks()` in `app/services.py`
2. Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `app/config.py`
3. Test with various document sizes
4. Monitor retrieval quality

### Testing Checklist

- [ ] Health check endpoint responds: `curl http://localhost:7860/health`
- [ ] API key validation works for both providers
- [ ] File upload and indexing for all 11+ formats
- [ ] RAG responses with and without conversation history
- [ ] All task types execute correctly
- [ ] Error handling for invalid inputs
- [ ] Cache functionality (check logs for cache hits)
- [ ] Response quality with different chunk configurations
- [ ] Docker build and deployment
- [ ] Frontend integration (UI updates, error messages)

---

## Deployment Considerations

### Environment Configuration

**Production Mode:**
```bash
OPENROUTER_API_KEY=""           # Leave empty to require user keys
OPENAI_API_KEY=""                # Leave empty to require user keys
REQUIRE_USER_API_KEY=True        # Force user-provided keys
```

**Development/Personal Mode:**
```bash
OPENROUTER_API_KEY="sk-or-xxxxx"  # Your personal key
REQUIRE_USER_API_KEY=False        # Allow fallback to server key
```

### Monitoring

Key metrics to monitor:
- Response time per endpoint
- Cache hit rate (`💾 CACHE HIT` in logs)
- LLM retry attempts
- File processing success rate
- Average chunk retrieval time
- API error rates by provider

### Scaling Considerations

**Current Limitations:**
- In-memory ChromaDB (resets on restart)
- In-memory response cache (not shared across instances)
- No persistent storage for indexed documents

**For Production:**
- Consider persistent ChromaDB storage (client = chromadb.PersistentClient())
- Implement Redis for shared caching
- Add rate limiting per API key
- Use connection pooling for LLM API calls
- Monitor memory usage (ChromaDB grows with documents)

---

## File References

When working on specific features, refer to these files:

- **API routes:** `app/main.py:1-179`
- **Configuration:** `app/config.py:1-175`
- **File processing:** `app/services.py:247-678`
- **RAG pipeline:** `app/services.py:856-1009`
- **Chunking logic:** `app/services.py:680-724`
- **LLM clients:** `app/rag_setup.py:130-638`
- **ChromaDB setup:** `app/rag_setup.py:1-128`
- **Data models:** `app/schemas.py:1-91`
- **Entry point:** `main.py:1-42`

---

## Quick Reference

**Default Models:**
- OpenRouter: `deepseek/deepseek-r1-0528:free` (completely free)
- OpenAI: `gpt-4o-mini` (cost-efficient)

**Default Port:** 7860 (Hugging Face standard)

**Max File Size:** Not explicitly limited (depends on server memory)

**Supported Browsers:** All modern browsers (Chrome, Firefox, Safari, Edge)

**License:** MIT - Free for commercial and personal use

**Repository:** https://github.com/Ab-Romia/ContextIQ-RAG

---

## Change History

**Version 2.2.0** (Current)
- Enhanced chunking with paragraph detection
- Increased context limits (12K chat, 16K task)
- Improved conversation history handling
- Added RAG system info panel to frontend
- Free DeepSeek model emphasized in documentation

**Previous Versions:**
- Multi-provider support (OpenAI + OpenRouter)
- 11+ file format support
- Custom TF-IDF embeddings
- Response caching
- Conversation history

---

*Last Updated: 2025-11-14*
*Generated by: Claude AI Assistant*
*Maintained by: Ab-Romia*
