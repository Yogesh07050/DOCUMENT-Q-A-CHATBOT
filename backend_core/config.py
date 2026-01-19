"""
Centralized configuration for the backend.
Values are sourced from environment variables with sensible defaults for local development.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parent.parent / ".env", extra="ignore")

    # Storage
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    upload_dir: Path = data_dir / "uploads"
    index_path: Path = data_dir / "faiss.index"
    metadata_path: Path = data_dir / "faiss-metadata.json"

    # RAG parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 80
    top_k: int = 5
    max_file_size_mb: int = 20

    # Generation
    generator_model: str = "gpt-3.5-turbo"
    openai_api_key: str | None = None

    # File validation
    allowed_extensions: tuple[str, ...] = (".txt", ".pdf", ".md")


settings = Settings()


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
