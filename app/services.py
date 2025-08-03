import asyncio
import functools
import logging
import textwrap
import time
from fastapi import HTTPException
import rag_setup
from schemas import ChatRequest, DocumentRequest, TaskRequest, ApiKeyRequest
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("rag-service")

# A simple cache to store recent responses to avoid redundant API calls for the same query.
# The cache stores a tuple of (timestamp, response).
_response_cache = {}
CACHE_EXPIRATION_SECONDS = 600  # 10 minutes


def create_llm_instance(api_key: str) -> rag_setup.OpenRouterLLM:
    """Create a new LLM instance with the provided API key."""
    from app.config import settings
    return rag_setup.OpenRouterLLM(
        api_key=api_key,
        base_url=settings.OPENROUTER_URL,
        model=settings.MODEL_NAME
    )


async def test_api_key(api_key: str) -> dict:
    """Test if the provided API key is valid."""
    logger.info(f"🔍 Testing API key: {api_key[:10]}...")

    try:
        # Validate API key format first
        if not api_key or not api_key.strip():
            logger.error("❌ API key is empty")
            return {
                "valid": False,
                "message": "API key cannot be empty",
                "model_info": None
            }

        if not api_key.startswith('sk-or-'):
            logger.error("❌ API key has incorrect format")
            return {
                "valid": False,
                "message": "OpenRouter API keys should start with 'sk-or-'",
                "model_info": None
            }

        if len(api_key) < 40:
            logger.error("❌ API key is too short")
            return {
                "valid": False,
                "message": "API key appears to be too short",
                "model_info": None
            }

        # Create a temporary LLM instance
        test_llm = create_llm_instance(api_key)

        # Test with a minimal prompt to avoid quota usage
        test_response = test_llm._make_api_request("Hi", max_tokens=1)

        # Check for explicit errors first
        if "error" in test_response:
            error_msg = test_response["error"]
            logger.error(f"❌ API key test failed: {error_msg}")

            # Parse specific error types
            if "401" in str(error_msg) or "403" in str(error_msg) or "Unauthorized" in str(error_msg):
                return {
                    "valid": False,
                    "message": "Invalid API key: Authentication failed",
                    "model_info": None
                }
            elif "429" in str(error_msg):
                return {
                    "valid": False,
                    "message": "API key valid but rate limited. Try again in a moment.",
                    "model_info": None
                }
            elif "insufficient" in str(error_msg).lower() or "quota" in str(error_msg).lower():
                # API key is valid but no credits - this is still a valid key
                logger.info("✅ API key is valid but has no credits")
                return {
                    "valid": True,
                    "message": "API key is valid but has insufficient credits",
                    "model_info": {"model": "unknown", "usage": {}, "credits": "insufficient"}
                }
            else:
                return {
                    "valid": False,
                    "message": f"API key test failed: {error_msg}",
                    "model_info": None
                }

        # Check for successful response with proper structure
        if "choices" in test_response and test_response["choices"]:
            choice = test_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                logger.info("✅ API key test successful")
                model_info = {
                    "model": test_response.get("model", "unknown"),
                    "usage": test_response.get("usage", {}),
                    "credits": "available"
                }
                return {
                    "valid": True,
                    "message": "API key is valid and working!",
                    "model_info": model_info
                }

        # If we get here, the response format is unexpected
        logger.error(f"❌ API key test failed: Unexpected response format - {test_response}")
        return {
            "valid": False,
            "message": "API key test failed: Unexpected response format from OpenRouter",
            "model_info": None
        }

    except Exception as e:
        logger.error(f"❌ API key test failed with exception: {str(e)}")
        error_msg = str(e)

        # Parse common error patterns
        if "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg:
            return {
                "valid": False,
                "message": "Invalid API key: Authentication failed",
                "model_info": None
            }
        elif "timeout" in error_msg.lower():
            return {
                "valid": False,
                "message": "API key test timed out. Please try again.",
                "model_info": None
            }
        elif "connection" in error_msg.lower():
            return {
                "valid": False,
                "message": "Network connection error. Please check your internet connection.",
                "model_info": None
            }
        else:
            return {
                "valid": False,
                "message": f"API key test failed: {error_msg}",
                "model_info": None
            }


def index_document(request_data: DocumentRequest, api_key: str) -> int:
    logger.info("=" * 80)
    logger.info("📚 STARTING DOCUMENT INDEXING PROCESS")
    logger.info("=" * 80)

    # Log the incoming context
    context_preview = request_data.context[:200] + "..." if len(request_data.context) > 200 else request_data.context
    logger.info(f"📝 CONTEXT TO INDEX (length: {len(request_data.context)} chars):")
    logger.info(f" Preview: {context_preview}")
    logger.info("-" * 60)

    try:
        # Step 1: Clear any existing documents
        existing_ids = rag_setup.collection.get()["ids"]
        if existing_ids:
            rag_setup.collection.delete(ids=existing_ids)
            logger.info("✅ Cleared existing documents from vector collection.")
        else:
            logger.info("🔍 No existing documents found to clear.")

        # Step 2: Chunk document
        # Splitting the text into manageable chunks for embedding and retrieval
        text_chunks = textwrap.wrap(
            request_data.context,
            width=800,
            break_long_words=False,
            replace_whitespace=False
        )

        if not text_chunks:
            logger.warning("⚠️ No text chunks were generated.")
            return 0

        # Step 3: Add chunks to ChromaDB
        # Generate unique IDs for each chunk
        chunk_ids = [f"doc_chunk_{i}_{int(time.time())}" for i in range(len(text_chunks))]

        logger.info(f"➡️ Attempting to add {len(chunk_ids)} chunks to ChromaDB...")
        rag_setup.collection.add(documents=text_chunks, ids=chunk_ids)

        logger.info("✅ Document chunks added successfully to ChromaDB.")
        return len(text_chunks)
    except Exception as e:
        logger.error(f"❌ Error during indexing: {str(e)}", exc_info=True)
        raise


def clear_index():
    """Clears all documents from the vector database."""
    rag_setup.collection.delete(where={})
    logger.info("Successfully cleared the vector index.")


async def get_rag_response(request_data: ChatRequest, api_key: str) -> str:
    """
    Performs the RAG pipeline: checks cache, retrieves context, generates a response.
    """
    start_total = time.time()
    logger.info(f"Processing query: '{request_data.prompt}'")

    try:
        # Step 1: Check cache for a recent, identical query
        cached_response = _get_cached_response(request_data.prompt)
        if cached_response:
            logger.info("Cache hit! Returning cached response.")
            return f"{cached_response}\n\n(This response was retrieved from cache)"

        logger.info("Cache miss. Proceeding with RAG pipeline.")

        # Step 2: Retrieve context from the vector DB
        start_retrieval = time.time()
        logger.info("Retrieving relevant documents from ChromaDB...")

        # The number of documents to retrieve is a configurable setting
        from app.config import settings

        results = rag_setup.collection.query(
            query_texts=[request_data.prompt],
            n_results=settings.MAX_CHUNKS_RETRIEVE
        )

        retrieved_documents = results.get("documents", [[]])[0]

        # If no documents are found, respond without context
        if not retrieved_documents:
            logger.warning("No relevant documents found. Generating a response without context.")
            context = ""
        else:
            context = "\n".join(retrieved_documents)
            logger.info(f"✅ Retrieved {len(retrieved_documents)} documents in {time.time() - start_retrieval:.2f}s.")
            logger.debug(f"Retrieved Context: {context}")

        # Step 3: Generate a response using the LLM with the retrieved context
        start_generation = time.time()
        logger.info("Generating response with LLM...")

        # Initialize LLM with the provided API key
        llm = create_llm_instance(api_key)

        system_prompt = (
            "You are an intelligent assistant. Use the provided context to answer the user's question. "
            "If the context doesn't contain the answer, state that you don't know based on the provided information. "
            f"Context:\n{context}"
        )

        generated_response = llm.generate_response(
            prompt=request_data.prompt,
            max_tokens=settings.MAX_TOKENS_CHAT,
            system_prompt=system_prompt
        )

        logger.info(f"✅ Response generated by LLM in {time.time() - start_generation:.2f}s.")

        # Step 4: Cache the response and return
        _cache_response(request_data.prompt, generated_response)

        logger.info(f"Total processing time: {time.time() - start_total:.2f}s")
        return generated_response

    except Exception as e:
        logger.error(f"❌ Error during RAG pipeline execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during response generation: {str(e)}"
        )


# Helper functions for caching
def _get_cached_response(query: str) -> Optional[str]:
    """Retrieves a cached response if it's still valid."""
    cached_data = _response_cache.get(query)
    if cached_data and (time.time() - cached_data[0] < CACHE_EXPIRATION_SECONDS):
        return cached_data[1]
    return None


def _cache_response(query: str, response: str):
    """Stores a response in the cache with the current timestamp."""
    _response_cache[query] = (time.time(), response)


async def perform_task_on_documents(request_data: TaskRequest, api_key: str) -> dict:
    """Performs a specific task on all indexed documents."""
    start_total = time.time()
    logger.info(f"Performing task: {request_data.task_name}")

    # Create LLM instance
    llm = create_llm_instance(api_key)

    if request_data.task_name == "summarize":
        try:
            # Retrieve all documents to summarize
            all_documents = rag_setup.collection.get(include=["documents"])["documents"]
            if not all_documents:
                return {"task_name": "summarize", "result": "No documents available to summarize."}

            combined_text = " ".join(all_documents)

            # Initialize LLM with the provided API key
            llm = create_llm_instance(api_key)

            # Use a system prompt to guide the summarization
            system_prompt = (
                "You are an intelligent summarization assistant. "
                "Your task is to create a concise and accurate summary of the provided text."
            )

            logger.info("Generating summary with LLM...")

            from app.config import settings
            summary_response = llm.generate_response(
                prompt=f"Please summarize the following text:\n\n{combined_text}",
                max_tokens=settings.MAX_TOKENS_SUMMARIZE,
                system_prompt=system_prompt
            )

            logger.info("✅ Summary generated successfully.")
            return {"task_name": "summarize", "result": summary_response}

        except Exception as e:
            logger.error(f"❌ Error during summarization task: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during summarization: {str(e)}"
            )
    elif request_data.task_name == "plan":
        # Get documents from ChromaDB
        docs = rag_setup.collection.get()
        documents_text = "\n\n".join(docs.get("documents", []))

        system_prompt = "Create a detailed action plan based on the provided documents."
        prompt = f"Based on the following information, generate an action plan:\n\n{documents_text}"
        response = llm.generate_response(prompt, max_tokens=5000)

        return {"task_name": "plan", "result": response}

    elif request_data.task_name == "creative":
        # Get documents from ChromaDB
        docs = rag_setup.collection.get()
        documents_text = "\n\n".join(docs.get("documents", []))

        system_prompt = "Create creative content based on the provided documents."
        prompt = f"Use the following information as inspiration for creative writing:\n\n{documents_text}"
        response = llm.generate_response(prompt, max_tokens=6000)

        return {"task_name": "creative", "result": response}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {request_data.task_name}"
        )