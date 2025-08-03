import os
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger("config")


class Settings(BaseSettings):
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"
    MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"

    # Token Limits Configuration
    MAX_TOKENS_CHAT: int = 4000  # For Q&A responses
    MAX_TOKENS_SUMMARIZE: int = 3000  # For summaries
    MAX_TOKENS_PLAN: int = 5000  # For action plans
    MAX_TOKENS_CREATIVE: int = 6000  # For creative writing
    MAX_TOKENS_TEST: int = 50  # For API key testing

    # Context Limits
    MAX_CONTEXT_LENGTH_CHAT: int = 8000  # For chat context
    MAX_CONTEXT_LENGTH_TASK: int = 12000  # For task context
    MAX_CHUNKS_RETRIEVE: int = 3  # Number of chunks to retrieve

    # Performance Settings
    REQUEST_TIMEOUT_BASE: int = 60  # Base timeout in seconds
    REQUEST_TIMEOUT_PER_1K_TOKENS: int = 2  # Additional seconds per 1000 tokens

    # New setting to control fallback behavior
    REQUIRE_USER_API_KEY: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Debug logging for API key configuration
logger.info("=" * 80)
logger.info("🔧 CONFIGURATION DEBUG")
logger.info("=" * 80)

# Check environment variables
logger.info("🌍 Environment Variables Check:")
logger.info(f" OPENROUTER_API_KEY in os.environ: {'OPENROUTER_API_KEY' in os.environ}")
logger.info(f" OPENROUTER_API_KEY length from env: {len(os.environ.get('OPENROUTER_API_KEY', ''))}")
logger.info("-" * 80)
logger.info("⚙️ Loaded Settings:")
logger.info(f" OPENROUTER_API_KEY set: {'Yes' if settings.OPENROUTER_API_KEY else 'No'}")
logger.info(f" MODEL_NAME: {settings.MODEL_NAME}")
logger.info(f" REQUIRE_USER_API_KEY: {settings.REQUIRE_USER_API_KEY}")
logger.info("=" * 80)