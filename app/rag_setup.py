import chromadb
import logging
import requests
import json
from config import settings, detect_provider_from_key  # Fixed: removed 'app.' prefix
import time
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import hashlib
import re
from openai import OpenAI

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

    def _make_api_request(self, prompt: str, max_tokens: int = 2000, timeout: int = None) -> dict:
        """Make a direct HTTP request to OpenRouter API with configurable token limits."""

        # Calculate dynamic timeout based on max_tokens and prompt length
        if timeout is None:
            base_timeout = 120
            # More tokens = longer generation time
            token_timeout = max(20, max_tokens // 100)  # ~1 second per 100 tokens
            prompt_timeout = max(10, len(prompt) // 1000)  # ~1 second per 2000 characters
            timeout = min(base_timeout + token_timeout + prompt_timeout, 600)  # Cap at 5 minutes

        logger.info(f"🌐 Making API request to OpenRouter")
        logger.info(f"📏 Prompt length: {len(prompt)} characters")
        logger.info(f"🎯 Max tokens: {max_tokens}")
        logger.info(f"⏱️  Timeout: {timeout}s")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ab-Romia/ContextIQ-RAG",
            "X-Title": "Context Aware AI"
        }

        # Optimize payload for longer responses
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False,
            # Add parameters to encourage complete responses
            "presence_penalty": 0.1,  # Slight penalty for repetition
            "frequency_penalty": 0.1,  # Slight penalty for frequency
        }

        # Log the request payload (without sensitive data)
        safe_payload = payload.copy()
        safe_payload["messages"] = [{"role": "user", "content": f"[CONTENT: {len(prompt)} chars]"}]
        logger.info(f"📤 Request payload: {json.dumps(safe_payload, indent=2)}")

        try:
            start_time = time.time()

            with requests.Session() as session:
                response = session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

            request_time = time.time() - start_time

            logger.info(f"⏱️  API request completed in {request_time:.2f}s")
            logger.info(f"📊 Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                logger.info("✅ API request successful")

                # Log response details
                if "choices" in response_data and response_data["choices"]:
                    content = response_data["choices"][0]["message"]["content"]
                    logger.info(f"📝 Response content length: {len(content)} characters")

                    # Check if response was truncated
                    if "usage" in response_data:
                        usage = response_data["usage"]
                        completion_tokens = usage.get("completion_tokens", 0)
                        logger.info(f"📊 Token usage: {usage}")

                        if completion_tokens >= max_tokens * 0.95:  # If we used 95% of max tokens
                            logger.warning(
                                f"⚠️  Response may be truncated (used {completion_tokens}/{max_tokens} tokens)")

                    content_preview = content[:300] + "..." if len(content) > 300 else content
                    logger.info(f"📄 Response preview: {content_preview}")

                return response_data
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                logger.error(f"📄 Response text: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}

        except requests.exceptions.Timeout:
            logger.error(f"⏱️  API request timed out after {timeout}s")
            return {"error": f"Request timed out after {timeout}s. Try reducing the context length or max tokens."}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"🌐 Connection error: {e}")
            return {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            return {"error": str(e)}

    def generate_content(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate content with configurable token limits."""
        logger.info("=" * 80)
        logger.info("🧠 LLM CONTENT GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"📏 Input prompt length: {len(prompt)} characters")
        logger.info(f"🎯 Requested max tokens: {max_tokens}")
        logger.info(f"🔧 Client status: {'Ready' if self.client_ready else 'Not ready'}")
        logger.info(f"🔑 API key status: {'Present' if self.api_key else 'Missing'}")

        # Dynamic prompt optimization based on max_tokens
        original_length = len(prompt)
        max_prompt_length = 12000 if max_tokens > 3000 else 8000  # Allow longer prompts for longer responses

        if len(prompt) > max_prompt_length:
            logger.warning(f"⚠️  Prompt is quite long ({original_length} chars), truncating for better performance")
            # Intelligent truncation that preserves structure
            if "Context:" in prompt and "Question:" in prompt:
                parts = prompt.split("Question:")
                if len(parts) == 2:
                    context_part = parts[0]
                    question_part = "Question:" + parts[1]

                    # Keep the question and instructions, truncate context if needed
                    available_for_context = max_prompt_length - len(question_part) - 500  # Reserve space
                    if len(context_part) > available_for_context:
                        context_part = context_part[
                                       :available_for_context] + "\n\n[... content truncated for performance ...]"

                    prompt = context_part + question_part
                    logger.info(f"📏 Prompt intelligently truncated from {original_length} to {len(prompt)} characters")
            else:
                prompt = prompt[:max_prompt_length] + "\n\n[... content truncated for performance ...]"
                logger.info(f"📏 Prompt truncated from {original_length} to {len(prompt)} characters")

        # Log prompt preview
        prompt_preview = prompt[:400] + "..." if len(prompt) > 400 else prompt
        logger.info(f"📝 PROMPT PREVIEW:")
        logger.info(f"   {prompt_preview}")
        logger.info("-" * 60)

        # Check API key first
        if not self.api_key or not self.api_key.strip():
            error_msg = "❌ OpenRouter API key is not configured. Please set the OPENROUTER_API_KEY environment variable."
            logger.error(error_msg)
            return error_msg

        # Check client readiness
        if not self.client_ready:
            error_msg = "❌ OpenRouter client is not ready. Please check your API key and connection."
            logger.error(error_msg)
            return error_msg

        max_retries = 3
        retry_count = 0
        base_wait_time = 2

        while retry_count <= max_retries:
            try:
                logger.info(f"🔄 API call attempt {retry_count + 1}/{max_retries + 1}")

                # Adjust parameters based on retry attempt
                current_max_tokens = max_tokens
                timeout = None  # Let _make_api_request calculate dynamic timeout

                if retry_count > 0:
                    # Reduce max_tokens on retries for faster responses
                    current_max_tokens = max(1000, max_tokens - (retry_count * 500))
                    logger.info(f"🔧 Retry attempt - reducing max_tokens to {current_max_tokens}")

                response = self._make_api_request(prompt, max_tokens=current_max_tokens, timeout=timeout)

                if "error" in response:
                    error_msg = response["error"]

                    # Handle specific error types
                    if "timeout" in error_msg.lower() or "408" in error_msg:
                        logger.warning(f"⏱️  Timeout error on attempt {retry_count + 1}")
                        if retry_count < max_retries:
                            continue
                    elif "429" in error_msg:
                        logger.warning(f"🚦 Rate limit error on attempt {retry_count + 1}")
                        wait_time = base_wait_time * (2 ** retry_count)
                        logger.info(f"⏳ Waiting {wait_time}s for rate limit cooldown...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    elif "401" in error_msg or "403" in error_msg:
                        logger.error(f"🔑 Authentication error: {error_msg}")
                        return f"❌ Authentication error: {error_msg}"

                    raise Exception(error_msg)

                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    if content:
                        logger.info(f"✅ Successfully generated response")
                        logger.info(f"📏 Response length: {len(content)} characters")

                        # Check if response seems complete
                        if "usage" in response:
                            usage = response["usage"]
                            completion_tokens = usage.get("completion_tokens", 0)
                            if completion_tokens >= current_max_tokens * 0.95:
                                logger.warning(
                                    f"⚠️  Response may be incomplete (used {completion_tokens}/{current_max_tokens} tokens)")
                                content += "\n\n[Note: Response may be truncated due to token limits. Consider asking for specific parts if needed.]"

                        response_preview = content[:400] + "..." if len(content) > 400 else content
                        logger.info(f"📤 RESPONSE PREVIEW:")
                        logger.info(f"   {response_preview}")
                        logger.info("=" * 80)

                        return content
                    else:
                        logger.error("❌ Received empty response from AI model")
                        if retry_count < max_retries:
                            retry_count += 1
                            continue
                        return "❌ Received empty response from AI model."
                else:
                    logger.error("❌ Invalid response format from AI model")
                    logger.error(f"📄 Response structure: {list(response.keys())}")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    return "❌ Invalid response format from AI model."

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                logger.error(f"❌ API call failed (attempt {retry_count + 1}): {error_type}: {error_msg}")

                retry_count += 1
                if retry_count > max_retries:
                    final_error = f"❌ Error: Failed to get response from AI model after {max_retries + 1} attempts. Final error: {error_msg}"
                    logger.error(final_error)
                    logger.info("=" * 80)
                    return final_error

                wait_time = base_wait_time * retry_count + (retry_count * 0.5)
                logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)


class OpenAILLM:
    """OpenAI LLM client for GPT models"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None

        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING OPENAI LLM")
        logger.info("=" * 60)
        logger.info(f"🤖 Model: {model}")
        logger.info(f"🔑 API Key present: {'Yes' if api_key else 'No'}")
        logger.info(f"📏 API Key length: {len(api_key) if api_key else 0}")
        logger.info(f"🌐 API URL: {base_url}")

        if not api_key or not api_key.strip():
            logger.error("❌ OpenAI API key is missing or empty")
            self.client_ready = False
            return

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("✅ OpenAI client initialized")

            # Test the connection with minimal tokens
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            if test_response:
                logger.info("✅ OpenAI connection test successful")
                self.client_ready = True
            else:
                logger.error("❌ OpenAI connection test failed")
                self.client_ready = False
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            self.client_ready = False

        logger.info("=" * 60)

    def generate_content(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate content using OpenAI API"""
        logger.info("=" * 80)
        logger.info("🧠 OPENAI CONTENT GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"📏 Input prompt length: {len(prompt)} characters")
        logger.info(f"🎯 Requested max tokens: {max_tokens}")
        logger.info(f"🔧 Client status: {'Ready' if self.client_ready else 'Not ready'}")

        # Check client readiness
        if not self.client_ready or not self.client:
            error_msg = "❌ OpenAI client is not ready. Please check your API key and connection."
            logger.error(error_msg)
            return error_msg

        # Optimize prompt length
        original_length = len(prompt)
        max_prompt_length = 12000 if max_tokens > 3000 else 8000

        if len(prompt) > max_prompt_length:
            logger.warning(f"⚠️  Prompt is quite long ({original_length} chars), truncating for better performance")
            if "Context:" in prompt and "Question:" in prompt:
                parts = prompt.split("Question:")
                if len(parts) == 2:
                    context_part = parts[0]
                    question_part = "Question:" + parts[1]
                    available_for_context = max_prompt_length - len(question_part) - 500
                    if len(context_part) > available_for_context:
                        context_part = context_part[:available_for_context] + "\n\n[... content truncated ...]"
                    prompt = context_part + question_part
                    logger.info(f"📏 Prompt intelligently truncated from {original_length} to {len(prompt)} characters")
            else:
                prompt = prompt[:max_prompt_length] + "\n\n[... content truncated ...]"
                logger.info(f"📏 Prompt truncated from {original_length} to {len(prompt)} characters")

        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                logger.info(f"🔄 API call attempt {retry_count + 1}/{max_retries + 1}")

                current_max_tokens = max_tokens
                if retry_count > 0:
                    current_max_tokens = max(1000, max_tokens - (retry_count * 500))
                    logger.info(f"🔧 Retry attempt - reducing max_tokens to {current_max_tokens}")

                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=current_max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )

                request_time = time.time() - start_time
                logger.info(f"⏱️  API request completed in {request_time:.2f}s")

                if response and response.choices:
                    content = response.choices[0].message.content
                    if content:
                        logger.info(f"✅ Successfully generated response")
                        logger.info(f"📏 Response length: {len(content)} characters")

                        # Check token usage
                        if hasattr(response, 'usage') and response.usage:
                            logger.info(f"📊 Token usage: {response.usage}")
                            if response.usage.completion_tokens >= current_max_tokens * 0.95:
                                logger.warning(f"⚠️  Response may be truncated")
                                content += "\n\n[Note: Response may be truncated due to token limits.]"

                        response_preview = content[:400] + "..." if len(content) > 400 else content
                        logger.info(f"📤 RESPONSE PREVIEW: {response_preview}")
                        logger.info("=" * 80)
                        return content
                    else:
                        logger.error("❌ Received empty response")
                        if retry_count < max_retries:
                            retry_count += 1
                            continue
                        return "❌ Received empty response from OpenAI."
                else:
                    logger.error("❌ Invalid response format")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    return "❌ Invalid response format from OpenAI."

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ API call failed (attempt {retry_count + 1}): {error_msg}")

                # Handle rate limits
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** retry_count
                    logger.info(f"⏳ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)

                retry_count += 1
                if retry_count > max_retries:
                    final_error = f"❌ Error: Failed after {max_retries + 1} attempts. Final error: {error_msg}"
                    logger.error(final_error)
                    logger.info("=" * 80)
                    return final_error

                time.sleep(2 * retry_count)


def create_llm(api_key: str, provider: Optional[str] = None) -> 'OpenRouterLLM | OpenAILLM':
    """
    Factory function to create the appropriate LLM client based on API key or provider.

    Args:
        api_key: The API key to use
        provider: Optional provider name ('openrouter' or 'openai'). If not provided, will auto-detect.

    Returns:
        An instance of OpenRouterLLM or OpenAILLM
    """
    # Auto-detect provider if not specified
    if provider is None:
        provider = detect_provider_from_key(api_key)
        logger.info(f"🔍 Auto-detected provider: {provider}")

    provider = provider.lower()

    if provider == "openai":
        logger.info("🎯 Creating OpenAI LLM client")
        return OpenAILLM(
            api_key=api_key,
            base_url=settings.OPENAI_URL,
            model=settings.OPENAI_MODEL
        )
    elif provider == "openrouter":
        logger.info("🎯 Creating OpenRouter LLM client")
        return OpenRouterLLM(
            api_key=api_key,
            base_url=settings.OPENROUTER_URL,
            model=settings.OPENROUTER_MODEL
        )
    else:
        logger.warning(f"⚠️  Unknown provider '{provider}', defaulting to OpenRouter")
        return OpenRouterLLM(
            api_key=api_key,
            base_url=settings.OPENROUTER_URL,
            model=settings.OPENROUTER_MODEL
        )


# Initialize the generation model with default settings
logger.info("🚀 Creating default LLM instance...")
try:
    # Use OpenRouter by default if API key is available
    if settings.OPENROUTER_API_KEY:
        generation_model = OpenRouterLLM(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_URL,
            model=settings.OPENROUTER_MODEL
        )
    elif settings.OPENAI_API_KEY:
        generation_model = OpenAILLM(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_URL,
            model=settings.OPENAI_MODEL
        )
    else:
        raise ValueError("No API key configured")

    if generation_model.client_ready:
        logger.info("✅ RAG setup completed successfully - LLM client is ready")
    else:
        logger.error("❌ RAG setup completed but LLM client is not ready")

except Exception as e:
    logger.error(f"❌ Error creating LLM: {e}")

    # Create a dummy model for graceful degradation
    class DummyLLM:
        def generate_content(self, prompt: str, max_tokens: int = 2000) -> str:
            return f"❌ AI model is not available. Initialization error: {str(e)}"

    generation_model = DummyLLM()
    logger.warning("⚠️  Using dummy LLM due to initialization failure")

logger.info("🎉 RAG setup initialization complete")