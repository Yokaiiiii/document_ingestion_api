from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    APP_NAME: str
    UPLOAD_DIR: str
    DATABASE_URL: str
    QDRANT_PATH: str
    REDIS_URL: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore

# Ensure the upload storage directory exists immediately
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
