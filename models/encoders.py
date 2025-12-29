"""
Dual-Encoder Models for Image and Text Encoding
Implements unique architecture for unified embedding space
"""

import numpy as np
import tensorflow as tf
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer
import torch
from typing import Union, List
import logging

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualEncoder:
    """
    Unified dual-encoder system for images and text.
    Uses pre-trained CLIP-based models to create a shared embedding space.
    """

    def __init__(self):
        """Initialize both text and image encoders"""
        logger.info("Initializing Dual-Encoder system...")

        # Text encoder using sentence-transformers
        self.text_model = SentenceTransformer('clip-ViT-B-32')
        logger.info(f"Text encoder loaded: {Config.TEXT_MODEL}")

        # Image encoder using CLIP
        self.image_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.image_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        logger.info(f"Image encoder loaded: {Config.IMAGE_MODEL}")

        self.embedding_dim = Config.EMBEDDING_DIMENSION

    def encode_text(self, texts: Union[str, List[str]], normalize=True) -> np.ndarray:
        """
        Encode text into embedding vectors

        Args:
            texts: Single text string or list of texts
            normalize: Whether to L2 normalize embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Encoding {len(texts)} text samples...")

        # Encode texts using sentence-transformers
        embeddings = self.text_model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )

        if normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    def encode_images(self, images: Union[np.ndarray, List], normalize=True) -> np.ndarray:
        """
        Encode images into embedding vectors

        Args:
            images: Single image array or list of images (PIL Images or numpy arrays)
            normalize: Whether to L2 normalize embeddings

        Returns:
            numpy array of shape (n_images, embedding_dim)
        """
        if not isinstance(images, list):
            images = [images]

        logger.info(f"Encoding {len(images)} image samples...")

        # Process images in batches
        all_embeddings = []

        for i in range(0, len(images), Config.BATCH_SIZE):
            batch = images[i:i + Config.BATCH_SIZE]

            # Preprocess images
            inputs = self.image_processor(
                images=batch,
                return_tensors="pt",
                padding=True
            )

            # Get image features
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)

            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        if normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    def encode_text_query(self, query: str, normalize=True) -> np.ndarray:
        """
        Encode a single text query

        Args:
            query: Text query string
            normalize: Whether to L2 normalize

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        return self.encode_text([query], normalize=normalize)

    def encode_image_query(self, image, normalize=True) -> np.ndarray:
        """
        Encode a single image query

        Args:
            image: PIL Image or numpy array
            normalize: Whether to L2 normalize

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        return self.encode_images([image], normalize=normalize)

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity

        Args:
            embeddings: Input embeddings

        Returns:
            L2 normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

    def compute_similarity(self, query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus embeddings

        Args:
            query_emb: Query embedding (1, embedding_dim)
            corpus_emb: Corpus embeddings (n, embedding_dim)

        Returns:
            Similarity scores (n,)
        """
        # Ensure both are L2 normalized
        query_emb = self._l2_normalize(query_emb)
        corpus_emb = self._l2_normalize(corpus_emb)

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(corpus_emb, query_emb.T).squeeze()

        return similarities

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings"""
        return self.embedding_dim


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings efficiently
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.text_cache = {}
        self.image_cache = {}

    def get_text_embedding(self, text: str) -> Union[np.ndarray, None]:
        """Get cached text embedding"""
        return self.text_cache.get(text)

    def set_text_embedding(self, text: str, embedding: np.ndarray):
        """Cache text embedding"""
        if len(self.text_cache) >= self.max_size:
            # Remove oldest entry
            self.text_cache.pop(next(iter(self.text_cache)))
        self.text_cache[text] = embedding

    def clear(self):
        """Clear all cached embeddings"""
        self.text_cache.clear()
        self.image_cache.clear()
        logger.info("Embedding cache cleared")
