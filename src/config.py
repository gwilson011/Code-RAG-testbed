"""
Configuration management for Code-RAG testbed.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the testbed."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    CONFIG_DIR = PROJECT_ROOT / "config"

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # ChromaDB Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "code_embeddings")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file. "
                "Copy .env.example to .env and add your API key."
            )
        return True

    @classmethod
    def get_chroma_path(cls) -> Path:
        """Get the ChromaDB path as a Path object."""
        return Path(cls.CHROMA_DB_PATH)


# Ensure required directories exist
Config.DATA_DIR.mkdir(exist_ok=True)
Config.RESULTS_DIR.mkdir(exist_ok=True)
Config.CONFIG_DIR.mkdir(exist_ok=True)

# Create .gitkeep files to track empty directories
(Config.RESULTS_DIR / ".gitkeep").touch(exist_ok=True)
