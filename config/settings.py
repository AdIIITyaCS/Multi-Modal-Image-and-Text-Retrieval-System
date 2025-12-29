"""
Configuration Management for Multi-Modal Retrieval System
Author: Custom Implementation
Date: 2025
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the retrieval system"""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CORPUS_DIR = DATA_DIR / "corpus"
    INDEX_DIR = DATA_DIR / "indices"
    LOG_DIR = BASE_DIR / "logs"

    # Flask configuration
    FLASK_APP = os.getenv("FLASK_APP", "app.py")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Model configuration
    TEXT_MODEL = os.getenv(
        "TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
    IMAGE_MODEL = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 512))

    # FAISS configuration
    FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(INDEX_DIR)))
    NPROBE = int(os.getenv("NPROBE", 10))
    # Use Flat for small datasets (<100 items), IVF for large datasets
    INDEX_TYPE = "Flat"
    NLIST = 100  # Number of clusters for IVF

    # Data configuration
    DATA_PATH = Path(os.getenv("DATA_PATH", str(CORPUS_DIR)))
    MAX_CORPUS_SIZE = int(os.getenv("MAX_CORPUS_SIZE", 10000))

    # Processing configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 224))
    IMAGE_CHANNELS = 3
    MAX_TEXT_LENGTH = 512

    # Search configuration
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 100

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.CORPUS_DIR,
            cls.INDEX_DIR,
            cls.LOG_DIR,
            cls.CORPUS_DIR / "images",
            cls.CORPUS_DIR / "texts",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_index_path(cls, index_name="main"):
        """Get the full path for a specific index"""
        return cls.INDEX_DIR / f"{index_name}.index"

    @classmethod
    def get_metadata_path(cls, index_name="main"):
        """Get the full path for index metadata"""
        return cls.INDEX_DIR / f"{index_name}_metadata.pkl"


# Create directories on import
Config.create_directories()
