from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = str(BASE_DIR / ".env")

class Settings(BaseSettings):
    google_ai_api_key: str

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()