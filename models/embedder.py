"""
Embedder - High-level interface for generating embeddings
"""

import numpy as np
from typing import List, Union, Dict
from pathlib import Path
import logging

from models.encoders import DualEncoder, EmbeddingCache
from preprocessing.image_processor import ImageProcessor
from preprocessing.text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalEmbedder:
    """
    High-level interface for embedding generation
    Handles preprocessing and encoding in a unified pipeline
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize the embedder

        Args:
            use_cache: Whether to use embedding caching
        """
        logger.info("Initializing Multi-Modal Embedder...")

        self.encoder = DualEncoder()
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        self.cache = EmbeddingCache() if use_cache else None

        logger.info("Embedder ready!")

    def embed_texts(self, texts: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            use_cache: Whether to use cached embeddings

        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        texts_to_encode = []
        cached_indices = []

        # Check cache
        if use_cache and self.cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get_text_embedding(text)
                if cached is not None:
                    embeddings.append(cached)
                    cached_indices.append(idx)
                else:
                    texts_to_encode.append(text)
        else:
            texts_to_encode = texts

        # Encode uncached texts
        if texts_to_encode:
            # Preprocess
            processed_texts = [self.text_processor.preprocess(
                t) for t in texts_to_encode]

            # Encode
            new_embeddings = self.encoder.encode_text(processed_texts)

            # Update cache
            if self.cache:
                for text, emb in zip(texts_to_encode, new_embeddings):
                    self.cache.set_text_embedding(text, emb)

            embeddings.extend(new_embeddings)

        return np.array(embeddings)

    def embed_images(self, image_paths: Union[str, Path, List]) -> np.ndarray:
        """
        Generate embeddings for image(s)

        Args:
            image_paths: Single path or list of paths to images

        Returns:
            Embeddings array
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        # Load and preprocess images
        images = []
        valid_paths = []

        for path in image_paths:
            try:
                img = self.image_processor.load_image(path)
                img = self.image_processor.preprocess(img)
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to process image {path}: {e}")

        if not images:
            logger.error("No valid images to encode")
            return np.array([])

        # Encode
        embeddings = self.encoder.encode_images(images)

        logger.info(f"Generated embeddings for {len(embeddings)} images")
        return embeddings

    def embed_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Generate embedding for an image array

        Args:
            image_array: Image as numpy array

        Returns:
            Embedding array
        """
        # Preprocess
        processed = self.image_processor.preprocess(image_array)

        # Encode
        embedding = self.encoder.encode_images([processed])

        return embedding

    def embed_query(self, query: Union[str, np.ndarray, Path], query_type: str = "auto") -> Dict:
        """
        Embed a query (automatically detect or specify type)

        Args:
            query: Query as text, image array, or image path
            query_type: "text", "image", or "auto"

        Returns:
            Dictionary with embedding and metadata
        """
        if query_type == "auto":
            # Auto-detect query type
            if isinstance(query, str) and not Path(query).exists():
                query_type = "text"
            else:
                query_type = "image"

        if query_type == "text":
            embedding = self.embed_texts(query)[0]
            return {
                "embedding": embedding,
                "type": "text",
                "query": query
            }

        elif query_type == "image":
            if isinstance(query, (str, Path)):
                embedding = self.embed_images([query])[0]
            else:
                embedding = self.embed_image_from_array(query)[0]

            return {
                "embedding": embedding,
                "type": "image",
                "query": str(query) if isinstance(query, (str, Path)) else "array"
            }

        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.encoder.get_embedding_dimension()

    def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
