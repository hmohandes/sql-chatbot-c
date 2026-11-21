from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/sql_chatbot"
    model_id: str = "google/gemma-4-E4B-it"
    hf_token: str | None = None
    top_k: int = 5
    sample_rows: int = 3
    max_agent_steps: int = 16
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    debug: bool = False
    dialect: str = "postgresql"


@lru_cache
def get_settings() -> Settings:
    return Settings()
