import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base: str = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    debug: bool = os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes"}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return s
