import chromadb
import logging
import requests
import json
from app.config import settings
import time
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import hashlib
import re

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-setup")


# Custom TF-IDF based embedding function
class TFIDFEmbeddingFunction:
    def __init__(self, max_features=384):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.is_fitted = False
        self.max_features = max_features
    def name(self):
        """Return a name identifier for this embedding function."""
        return "tfidf_embedder"
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip().lower()

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate TF-IDF based embeddings."""
        try:
            logger.info(f"🔢 Generating embeddings for {len(input)} texts")
            processed_texts = [self._preprocess_text(text) for text in input]

            if not self.is_fitted and processed_texts:
                # Fit the vectorizer on the input texts
                self.vectorizer.fit(processed_texts)
                self.is_fitted = True
                logger.info("✅ TF-IDF vectorizer fitted on input texts")

            if not self.is_fitted:
                # Return simple fallback if no data to fit on
                logger.warning("⚠️  Using fallback embeddings - vectorizer not fitted")
                return self._fallback_embeddings(input)

            # Transform texts to vectors
            tfidf_matrix = self.vectorizer.transform(processed_texts)
            embeddings = tfidf_matrix.toarray()

            # Ensure consistent dimensions
            result_embeddings = []
            for embedding in embeddings:
                if len(embedding) < self.max_features:
                    padded = np.zeros(self.max_features)
                    padded[:len(embedding)] = embedding
                    result_embeddings.append(padded.tolist())
                else:
                    result_embeddings.append(embedding[:self.max_features].tolist())

            logger.info(f"✅ Generated {len(result_embeddings)} embeddings of dimension {self.max_features}")
            return result_embeddings

        except Exception as e:
            logger.error(f"❌ Error generating TF-IDF embeddings: {e}")
            return self._fallback_embeddings(input)

    def _fallback_embeddings(self, input: List[str]) -> List[List[float]]:
        """Simple fallback embedding method."""
        logger.info(f"🔧 Using fallback embeddings for {len(input)} texts")
        embeddings = []
        for text in input:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = []

            # Convert hash to numbers
            for i in range(0, min(len(text_hash), 32), 2):
                hex_pair = text_hash[i:i + 2]
                embedding.append(int(hex_pair, 16) / 255.0)

            # Add text features
            embedding.extend([
                len(text) / 1000.0,
                len(text.split()) / 100.0,
                text.count('.') / 10.0,
            ])

            # Pad to desired size
            while len(embedding) < self.max_features:
                embedding.extend(embedding[:min(len(embedding), self.max_features - len(embedding))])

            embeddings.append(embedding[:self.max_features])

        return embeddings


# Simple ChromaDB setup - use in-memory storage for Hugging Face
logger.info("🔧 Initializing ChromaDB with in-memory storage for Hugging Face compatibility")

try:
    # Use in-memory client to avoid permission issues
    client = chromadb.Client()
    embedding_function = TFIDFEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="context_aware_collection",
        embedding_function=embedding_function
    )

    logger.info("✅ ChromaDB collection initialized successfully with in-memory storage")

except Exception as e:
    logger.error(f"❌ Error initializing ChromaDB: {e}")
    raise RuntimeError(f"Failed to initialize ChromaDB: {e}")


class OpenRouterLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"

        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING OPENROUTER LLM")
        logger.info("=" * 60)
        logger.info(f"🤖 Model: {model}")
        logger.info(f"🔑 API Key present: {'Yes' if api_key else 'No'}")
        logger.info(f"📏 API Key length: {len(api_key) if api_key else 0}")
        logger.info(f"🌐 API URL: {self.api_url}")

        if not api_key or not api_key.strip():
            logger.error("❌ OpenRouter API key is missing or empty")
            self.client_ready = False
            return

        # Test the connection with minimal tokens
        try:
            logger.info("🔍 Testing OpenRouter connection...")
            test_response = self._make_api_request("Hello", max_tokens=5)
            if test_response and "error" not in test_response:
                logger.info("✅ OpenRouter connection test successful")
                self.client_ready = True
            else:
                logger.error(f"❌ OpenRouter connection test failed: {test_response}")
                self.client_ready = False
        except Exception as e:
            logger.error(f"❌ OpenRouter connection test failed: {e}")
            self.client_ready = False

        logger.info("=" * 60)

    def _make_api_request(self, prompt: str, max_tokens: int, system_prompt: str = "") -> dict:
        """Helper to make the API call with the correct headers and payload."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://huggingface.co/spaces/google/contextiq-rag",  # Replace with your deployment URL
            "X-Title": "ContextIQ RAG",  # Replace with your app name
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        logger.info(f"➡️ Sending request to {self.api_url} with prompt: {prompt[:50]}...")

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()  # Raise an exception for bad status codes
            response_json = response.json()

            logger.info(f"✅ Received response from LLM (status code: {response.status_code})")
            return response_json

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error making API request: {e}")
            return {"error": str(e)}

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        """Generates a response from the LLM."""
        if not self.client_ready:
            logger.error("❌ LLM client is not ready. Cannot generate response.")
            return "Error: LLM client is not ready. Please check the API key."

        response_json = self._make_api_request(prompt, max_tokens)

        if "error" in response_json:
            error_msg = response_json["error"]
            logger.error(f"❌ LLM API Error: {error_msg}")
            return f"Error from LLM API: {error_msg}"

        try:
            generated_text = response_json["choices"][0]["message"]["content"]
            return generated_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"❌ Unexpected response format from LLM: {e}")
            return f"Error: Unexpected response format from LLM. Raw response: {response_json}"