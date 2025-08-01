import asyncio
import functools
import logging
import textwrap
import time
import rag_setup
from schemas import ChatRequest, DocumentRequest, TaskRequest
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


def index_document(request_data: DocumentRequest) -> int:
    logger.info("Starting document indexing process.")

    try:
        # Step 1: Clear any existing documents properly
        existing_ids = rag_setup.collection.get()["ids"]
        if existing_ids:
            rag_setup.collection.delete(ids=existing_ids)
        logger.info("Cleared existing documents from vector collection.")

        # Step 2: Chunk document
        text_chunks = textwrap.wrap(
            request_data.context,
            width=800,
            break_long_words=False,
            replace_whitespace=False
        )

        if not text_chunks:
            logger.warning("No text chunks were generated.")
            return 0

        # Step 3: Add chunks to ChromaDB
        chunk_ids = [f"doc_chunk_{i}_{int(time.time())}" for i in range(len(text_chunks))]
        logger.info(f"Attempting to add {len(chunk_ids)} chunks to ChromaDB...")
        rag_setup.collection.add(documents=text_chunks, ids=chunk_ids)

        return len(text_chunks)
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}", exc_info=True)
        raise

def clear_index():
    """Clears all documents from the vector database."""
    rag_setup.collection.delete(where={})
    logger.info("Successfully cleared the vector index.")


async def get_rag_response(request_data: ChatRequest) -> str:
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

        # Step 2: Check if the vector database has any content
        if rag_setup.collection.count() == 0:
            logger.warning("Vector DB is empty. Cannot answer query.")
            return "The knowledge base is empty. Please provide some context in the left panel and click 'Index Context' before asking questions."

        # Step 3: Retrieve relevant chunks from ChromaDB
        logger.info("Retrieving relevant chunks from vector DB...")
        retrieved_chunks = await _retrieve_chunks_async(request_data.prompt)

        if not retrieved_chunks or not retrieved_chunks.get('documents') or not retrieved_chunks['documents'][0]:
            logger.warning("No relevant chunks found in the vector DB for this query.")
            return "I could not find any relevant information in the provided context to answer your question."

        context_for_prompt = "\n\n---\n\n".join(retrieved_chunks['documents'][0])

        # Step 4: Construct the final prompt for the LLM
        full_prompt = (
            "You are a helpful AI assistant. Based strictly and only on the following context, "
            "please answer the user's question. Do not use any external knowledge or make assumptions. "
            "If the answer cannot be found in the context, state that clearly.\n\n"
            "--- CONTEXT START ---\n"
            f"{context_for_prompt}\n"
            "--- CONTEXT END ---\n\n"
            f'User\'s Question: "{request_data.prompt}"'
        )

        # Step 5: Generate the response using the LLM
        logger.info("Generating response from OpenRouter...")
        response_text = await _generate_response_async(full_prompt)

        # Step 6: Cache the newly generated response
        _cache_response(request_data.prompt, response_text)

        total_time = time.time() - start_total
        logger.info(f"Total processing time: {total_time:.2f}s")
        return response_text

    except asyncio.TimeoutError:
        logger.error("Request timed out during retrieval or generation.")
        return "The request timed out. Please try again or simplify your question."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return f"An unexpected error occurred: {e}"


async def execute_task(request_data: TaskRequest) -> str:
    """
    Executes a specific task on the given context.
    """
    start_total = time.time()
    logger.info(f"Executing task '{request_data.task_type}' with prompt: '{request_data.prompt}'")

    try:
        # For tasks, we use the full context, not just retrieved chunks
        context = request_data.context
        if not context:
            return "Context is empty. Please provide some text in the 'Knowledge Base' to perform a task."

        # Construct the prompt based on the task type
        if request_data.task_type == "summarize":
            full_prompt = f"Summarize the following text:\n\n---\n{context}"
        elif request_data.task_type == "plan":
            full_prompt = f"Based on the following context, create a detailed action plan. If a specific goal is provided in the prompt, tailor the plan to that goal.\n\n--- CONTEXT ---\n{context}\n\n--- GOAL ---\n{request_data.prompt or 'General objective derived from context'}"
        elif request_data.task_type == "creative":
            full_prompt = f"Use the following text as inspiration to write a creative piece (e.g., a poem, a short story, a metaphor). The user's prompt can guide the style or topic.\n\n--- INSPIRATION ---\n{context}\n\n--- PROMPT ---\n{request_data.prompt or 'Write a short poem'}"
        else:
            return "Invalid task type specified."

        # Generate the response
        logger.info("Generating task-based response from OpenRouter...")
        response_text = await _generate_response_async(full_prompt)

        total_time = time.time() - start_total
        logger.info(f"Task execution time: {total_time:.2f}s")
        return response_text

    except asyncio.TimeoutError:
        logger.error("Request timed out during task execution.")
        return "The request timed out. Please try again."
    except Exception as e:
        logger.error(f"An unexpected error occurred during task execution: {e}", exc_info=True)
        return f"An unexpected error occurred: {e}"

# --- ASYNC WRAPPERS & CACHE HELPERS ---

async def _retrieve_chunks_async(prompt: str):
    """Asynchronously queries the ChromaDB collection."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(rag_setup.collection.query, query_texts=[prompt], n_results=3)
    )


async def _generate_response_async(full_prompt: str):
    """Asynchronously calls the LLM to generate content."""


    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        rag_setup.generation_model.generate_content,
        full_prompt
    )

def _get_cached_response(key: str):
    """Checks the cache for a valid (non-expired) entry."""
    if key in _response_cache:
        timestamp, response = _response_cache[key]
        if time.time() - timestamp < CACHE_EXPIRATION_SECONDS:
            return response
        else:
            # Expired, remove from cache
            del _response_cache[key]
    return None


def _cache_response(key: str, response: str):
    """Adds a response to the cache with the current timestamp."""
    _response_cache[key] = (time.time(), response)