import chromadb
import logging
from openai import OpenAI
from app.config import settings
from chromadb.utils import embedding_functions
import time
import os
import shutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-setup")

# Initialize variables
embedding_model = None
generation_model = None

# Use the default SentenceTransformer for creating embeddings locally.
# This is efficient as it doesn't require an API call for embedding.
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
db_path = "./chroma_db"
if os.path.exists(db_path):
    logger.info("Removing existing ChromaDB to avoid embedding conflicts")
    shutil.rmtree(db_path)

# Set up a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="context_aware_collection",
    embedding_function=sentence_transformer_ef
)
logger.info("ChromaDB collection 'context_aware_collection' loaded/created.")
class OpenRouterLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is missing. Please set it in your .env file.")
        self.client = OpenAI(base_url=base_url, api_key=api_key,timeout=45.0)
        self.model = model
        logger.info(f"OpenRouter client initialized for model: {self.model}")

    def generate_content(self, prompt: str) -> str:
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/Ab-Romia/ContextIQ-RAG",
                        "X-Title": "Context Aware AI",
                    },
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"API call failed (attempt {retry_count + 1}): {str(e)}")
                retry_count += 1
                if retry_count > max_retries:
                    return f"Error: The model did not respond in time. Try a shorter question or simplify your context."
                time.sleep(2)  # Wait before retrying


# Initialize the generation model with settings from config.py
generation_model = OpenRouterLLM(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_URL,
    model=settings.MODEL_NAME
)

logger.info("RAG setup initialized successfully.")
