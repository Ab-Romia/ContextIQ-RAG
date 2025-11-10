import os
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger("config")


class Settings(BaseSettings):
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "deepseek/deepseek-r1-0528:free"

    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"  # Default to GPT-4o-mini for cost efficiency

    # Legacy field for backward compatibility
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
    REQUEST_TIMEOUT_BASE: int = 120  # Base timeout in seconds
    REQUEST_TIMEOUT_PER_1K_TOKENS: int = 4  # Additional seconds per 1000 tokens

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
logger.info(f"   OPENROUTER_API_KEY in os.environ: {'OPENROUTER_API_KEY' in os.environ}")
logger.info(f"   OPENROUTER_API_KEY length from env: {len(os.environ.get('OPENROUTER_API_KEY', ''))}")

# Check settings object
logger.info("⚙️  Settings Object Check:")
logger.info(f"   settings.OPENROUTER_API_KEY present: {bool(settings.OPENROUTER_API_KEY)}")
logger.info(f"   settings.OPENROUTER_API_KEY length: {len(settings.OPENROUTER_API_KEY)}")
logger.info(f"   settings.OPENROUTER_URL: {settings.OPENROUTER_URL}")
logger.info(f"   settings.MODEL_NAME: {settings.MODEL_NAME}")
logger.info(f"   settings.REQUIRE_USER_API_KEY: {settings.REQUIRE_USER_API_KEY}")

# Configuration mode detection
if settings.REQUIRE_USER_API_KEY and not settings.OPENROUTER_API_KEY:
    logger.info("🔑 CONFIGURATION MODE: User API Key Required")
    logger.info("   - Users must provide their own OpenRouter API key")
    logger.info("   - No fallback key available")
elif settings.OPENROUTER_API_KEY and not settings.REQUIRE_USER_API_KEY:
    logger.info("🔧 CONFIGURATION MODE: Server API Key (Legacy)")
    logger.info("   - Using server-provided API key as fallback")
    logger.warning("   - ⚠️  This may cause quota conflicts with multiple users")
else:
    logger.info("🔀 CONFIGURATION MODE: Hybrid")
    logger.info("   - Users can provide their own API key")
    logger.info("   - Server API key available as fallback")

# Check if API key starts with expected prefix (only if present)
if settings.OPENROUTER_API_KEY:
    api_key_preview = settings.OPENROUTER_API_KEY[:20] + "..." if len(
        settings.OPENROUTER_API_KEY) > 20 else settings.OPENROUTER_API_KEY
    logger.info(f"🔑 Server API Key Preview: {api_key_preview}")

    # OpenRouter API keys typically start with "sk-or-"
    if settings.OPENROUTER_API_KEY.startswith("sk-or-"):
        logger.info("✅ Server API key format looks correct (starts with 'sk-or-')")
    else:
        logger.warning("⚠️  Server API key might be incorrect format (should start with 'sk-or-')")
else:
    logger.info("🔑 No server API key configured - users must provide their own")

# Additional debugging for Hugging Face environment
if os.path.exists("/app"):
    logger.info("🤗 Running in Hugging Face environment")
    logger.info("💡 RECOMMENDATION: Remove server API key and require users to provide their own")
    logger.info("   This prevents quota conflicts and gives users better control")

logger.info("=" * 80)


def get_max_tokens_for_task(task_type: str) -> int:
    """Get the appropriate max tokens for a specific task type."""
    token_map = {
        "q_and_a": settings.MAX_TOKENS_CHAT,
        "summarize": settings.MAX_TOKENS_SUMMARIZE,
        "plan": settings.MAX_TOKENS_PLAN,
        "creative": settings.MAX_TOKENS_CREATIVE,
        "test": settings.MAX_TOKENS_TEST
    }
    return token_map.get(task_type, settings.MAX_TOKENS_CHAT)


def get_timeout_for_tokens(max_tokens: int) -> int:
    """Calculate appropriate timeout based on token count."""
    additional_time = (max_tokens // 1000) * settings.REQUEST_TIMEOUT_PER_1K_TOKENS
    return settings.REQUEST_TIMEOUT_BASE + additional_time


def validate_api_key(api_key: str, provider: str = "openrouter") -> bool:
    """Validate API key format for OpenRouter or OpenAI"""
    if not api_key:
        return False

    provider = provider.lower()

    if provider == "openrouter":
        # OpenRouter keys should start with "sk-or-" and be at least 40 characters
        if not api_key.startswith("sk-or-"):
            logger.warning("⚠️  OpenRouter API key should start with 'sk-or-'")
            return False
        if len(api_key) < 40:
            logger.warning("⚠️  OpenRouter API key seems too short")
            return False
    elif provider == "openai":
        # OpenAI keys should start with "sk-" and be at least 40 characters
        if not api_key.startswith("sk-"):
            logger.warning("⚠️  OpenAI API key should start with 'sk-'")
            return False
        if len(api_key) < 40:
            logger.warning("⚠️  OpenAI API key seems too short")
            return False
    else:
        logger.warning(f"⚠️  Unknown provider: {provider}")
        return False

    return True


def detect_provider_from_key(api_key: str) -> str:
    """Detect provider from API key format"""
    if not api_key:
        return "unknown"

    if api_key.startswith("sk-or-"):
        return "openrouter"
    elif api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
        return "openai"
    else:
        return "unknown"


# Validate the current server API key (if present)
if settings.OPENROUTER_API_KEY:
    is_valid = validate_api_key(settings.OPENROUTER_API_KEY)
    logger.info(f"🔍 Server API Key Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
else:
    logger.info("🔍 No server API key to validate")

# Export settings
__all__ = ['settings', 'validate_api_key', 'detect_provider_from_key', 'get_max_tokens_for_task', 'get_timeout_for_tokens']