from . import rag_setup
from .schemas import ChatRequest, DocumentRequest
import textwrap
import asyncio
import functools
import time
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("rag-service")


def index_document(request_data: DocumentRequest) -> int:
    """
    Chunks text, creates embeddings, and stores them in the vector DB.
    """
    logger.info(f"Starting document indexing process. Text length: {len(request_data.context)}")

    # 1. Chunk the document
    start_time = time.time()
    text_chunks = [chunk for chunk in textwrap.wrap(request_data.context, width=500, replace_whitespace=False)]
    chunk_time = time.time() - start_time
    logger.info(f"Document chunked into {len(text_chunks)} segments ({chunk_time:.2f}s)")

    # 2. Embed and store chunks in ChromaDB
    start_time = time.time()
    rag_setup.collection.add(
        documents=text_chunks,
        ids=[f"doc_chunk_{i}" for i in range(len(text_chunks))]
    )
    embed_time = time.time() - start_time
    logger.info(f"Chunks embedded and stored in vector DB ({embed_time:.2f}s)")

    total_count = rag_setup.collection.count()
    logger.info(f"Vector DB now contains {total_count} total chunks")

    return len(text_chunks)


async def get_rag_response(request_data: ChatRequest) -> str:
    """
    Performs RAG with timeout and caching improvements
    """
    start_total = time.time()
    logger.info(f"Processing query: '{request_data.prompt}'")

    try:
        # Define cache key at the beginning
        cache_key = f"{request_data.prompt}"

        # Check cache first
        logger.info("Checking response cache...")
        cached_response = _get_cached_response(cache_key)
        if cached_response:
            logger.info("Cache hit! Returning cached response")
            return f"{cached_response}\n(Cached response)"
        logger.info("Cache miss, continuing with retrieval")

        # Check collection status
        collection_count = rag_setup.collection.count()
        logger.info(f"Vector DB contains {collection_count} chunks")

        if collection_count == 0:
            logger.warning("Empty vector database, returning explanation message")
            return "The vector database is empty. Please provide context text in the left panel and it will be indexed automatically."

        # Retrieve with timeout
        logger.info("Retrieving relevant chunks from vector DB...")
        start_retrieve = time.time()
        retrieved_chunks = await asyncio.wait_for(
            _retrieve_chunks(request_data.prompt),
            timeout=5.0
        )
        retrieve_time = time.time() - start_retrieve
        logger.info(f"Retrieved {len(retrieved_chunks['documents'][0])} chunks ({retrieve_time:.2f}s)")

        # Check if any relevant chunks were found
        if len(retrieved_chunks['documents'][0]) == 0:
            logger.warning("No relevant chunks found in the vector DB")
            return "No relevant information was found in the provided context."

        # Format and generate
        logger.info("Preparing prompt with retrieved context...")
        context_chunks = retrieved_chunks['documents'][0]
        context_for_prompt = "\n\n".join(context_chunks)

        full_prompt = (
            "Based strictly and only on the following context, please answer the user's question. "
            "Do not use any external knowledge. If the answer cannot be found in the context, state that clearly.\n\n"
            "--- CONTEXT START ---\n"
            f"{context_for_prompt}\n"
            "--- CONTEXT END ---\n\n"
            f'User\'s Question: "{request_data.prompt}"'
        )
        logger.info(f"Prompt ready, length: {len(full_prompt)} characters")

        # Generate response with timeout
        logger.info("Generating response from Gemini API...")
        start_generate = time.time()
        response_text = await asyncio.wait_for(
            _generate_response(full_prompt),
            timeout=15.0
        )
        generate_time = time.time() - start_generate
        logger.info(f"Response generated ({generate_time:.2f}s)")

        # Cache the response
        logger.info("Caching response for future queries")
        _cache_response(cache_key, response_text)

        total_time = time.time() - start_total
        logger.info(f"Total processing time: {total_time:.2f}s")
        return response_text

    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return "The request timed out. Please try again or simplify your question."
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}"


# Async wrapper for ChromaDB calls
async def _retrieve_chunks(prompt):
    logger.debug(f"Executing vector query: '{prompt}'")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            rag_setup.collection.query,
            query_texts=[prompt],
            n_results=3
        )
    )


# Async wrapper for Gemini calls
async def _generate_response(prompt):
    logger.debug("Sending prompt to Gemini API")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        functools.partial(
            rag_setup.generation_model.generate_content,
            prompt
        )
    )
    return response.text


# Simple in-memory cache
_response_cache = {}


def _get_cached_response(key):
    if key in _response_cache:
        timestamp, response = _response_cache[key]
        # Cache expires after 10 minutes
        if time.time() - timestamp < 600:
            return response
    return None


def _cache_response(key, response):
    _response_cache[key] = (time.time(), response)