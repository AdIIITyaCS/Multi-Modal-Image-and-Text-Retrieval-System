"""
Indexer - High-level interface for building search indices
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
from tqdm import tqdm

from indexing.faiss_manager import FAISSManager
from models.embedder import MultiModalEmbedder
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalIndexer:
    """
    High-level indexer for building multi-modal search indices
    Handles both text and image corpora
    """

    def __init__(self):
        """Initialize the indexer"""
        logger.info("Initializing Multi-Modal Indexer...")

        self.embedder = MultiModalEmbedder()
        self.faiss_manager = FAISSManager(
            embedding_dim=self.embedder.get_embedding_dimension()
        )

        # Separate indices for images and texts
        self.image_index = None
        self.text_index = None

    def index_images(self, image_dir: Path, index_name: str = "images",
                     max_images: Optional[int] = None) -> int:
        """
        Index images from a directory

        Args:
            image_dir: Directory containing images
            index_name: Name for the index
            max_images: Maximum number of images to index

        Returns:
            Number of images indexed
        """
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = [
            p for p in image_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]

        if max_images:
            image_paths = image_paths[:max_images]

        logger.info(f"Found {len(image_paths)} images to index")

        if not image_paths:
            logger.warning("No images found to index")
            return 0

        # Create index
        self.image_index = FAISSManager(
            self.embedder.get_embedding_dimension())
        self.image_index.create_index(index_type=Config.INDEX_TYPE)

        # Process in batches
        batch_size = Config.BATCH_SIZE
        all_embeddings = []
        all_metadata = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Indexing images"):
            batch_paths = image_paths[i:i + batch_size]

            try:
                # Generate embeddings
                embeddings = self.embedder.embed_images(batch_paths)

                # Create metadata
                metadata = [
                    {
                        "type": "image",
                        "path": str(p.absolute()),
                        "filename": p.name,
                        "index": i + j
                    }
                    for j, p in enumerate(batch_paths)
                ]

                all_embeddings.append(embeddings)
                all_metadata.extend(metadata)

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")

        # Combine all embeddings
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)

            # Train and add to index
            if Config.INDEX_TYPE == "IVF":
                self.image_index.train_index(all_embeddings)

            self.image_index.add_embeddings(all_embeddings, all_metadata)

            # Save index
            self.image_index.save_index(index_name)

            logger.info(f"Successfully indexed {len(all_metadata)} images")
            return len(all_metadata)

        return 0

    def index_texts(self, text_source: Path, index_name: str = "texts",
                    max_texts: Optional[int] = None) -> int:
        """
        Index texts from a file or directory

        Args:
            text_source: Path to text file or directory
            index_name: Name for the index
            max_texts: Maximum number of texts to index

        Returns:
            Number of texts indexed
        """
        text_source = Path(text_source)

        if not text_source.exists():
            raise ValueError(f"Text source not found: {text_source}")

        texts = []
        metadata = []

        # Load texts
        if text_source.is_file():
            texts, metadata = self._load_text_file(text_source)
        elif text_source.is_dir():
            texts, metadata = self._load_text_directory(text_source)

        if max_texts:
            texts = texts[:max_texts]
            metadata = metadata[:max_texts]

        logger.info(f"Found {len(texts)} texts to index")

        if not texts:
            logger.warning("No texts found to index")
            return 0

        # Create index
        self.text_index = FAISSManager(self.embedder.get_embedding_dimension())
        self.text_index.create_index(index_type=Config.INDEX_TYPE)

        # Generate embeddings
        logger.info("Generating embeddings for texts...")
        embeddings = self.embedder.embed_texts(texts, use_cache=False)

        # Train and add to index
        if Config.INDEX_TYPE == "IVF":
            self.text_index.train_index(embeddings)

        self.text_index.add_embeddings(embeddings, metadata)

        # Save index
        self.text_index.save_index(index_name)

        logger.info(f"Successfully indexed {len(texts)} texts")
        return len(texts)

    def _load_text_file(self, file_path: Path) -> tuple:
        """Load texts from a JSON file"""
        texts = []
        metadata = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)

                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            if isinstance(item, dict):
                                texts.append(item.get('text', str(item)))
                                metadata.append({
                                    "type": "text",
                                    "source": str(file_path),
                                    "index": idx,
                                    **item
                                })
                            else:
                                texts.append(str(item))
                                metadata.append({
                                    "type": "text",
                                    "source": str(file_path),
                                    "index": idx
                                })
                    elif isinstance(data, dict):
                        for key, value in data.items():
                            texts.append(str(value))
                            metadata.append({
                                "type": "text",
                                "source": str(file_path),
                                "key": key
                            })
                else:
                    # Plain text file
                    content = f.read()
                    # Split into paragraphs or sentences
                    paragraphs = [p.strip()
                                  for p in content.split('\n\n') if p.strip()]
                    texts.extend(paragraphs)
                    metadata.extend([
                        {
                            "type": "text",
                            "source": str(file_path),
                            "index": i
                        }
                        for i in range(len(paragraphs))
                    ])

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")

        return texts, metadata

    def _load_text_directory(self, directory: Path) -> tuple:
        """Load texts from all files in a directory"""
        texts = []
        metadata = []

        text_files = list(directory.glob('**/*.txt')) + \
            list(directory.glob('**/*.json'))

        for file_path in text_files:
            file_texts, file_metadata = self._load_text_file(file_path)
            texts.extend(file_texts)
            metadata.extend(file_metadata)

        return texts, metadata

    def load_index(self, index_name: str, index_type: str = "image") -> bool:
        """
        Load a saved index

        Args:
            index_name: Name of the index
            index_type: "image" or "text"

        Returns:
            True if successful
        """
        manager = FAISSManager(self.embedder.get_embedding_dimension())
        success = manager.load_index(index_name)

        if success:
            if index_type == "image":
                self.image_index = manager
            else:
                self.text_index = manager

        return success

    def get_stats(self) -> Dict:
        """Get statistics for all indices"""
        stats = {}

        if self.image_index:
            stats['image_index'] = self.image_index.get_stats()

        if self.text_index:
            stats['text_index'] = self.text_index.get_stats()

        return stats
