from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )

settings = Settings()