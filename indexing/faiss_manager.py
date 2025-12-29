"""
FAISS Index Manager
Handles creation, loading, and searching of FAISS indices
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Manages FAISS indices for efficient similarity search
    Supports both flat and IVF (Inverted File) indices
    """

    def __init__(self, embedding_dim: int = None):
        """
        Initialize FAISS Manager

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIMENSION
        self.index = None
        self.metadata = None
        self.index_type = Config.INDEX_TYPE

    def create_index(self, index_type: str = "IVF", nlist: int = None) -> faiss.Index:
        """
        Create a new FAISS index

        Args:
            index_type: Type of index ("Flat" or "IVF")
            nlist: Number of clusters for IVF (default from config)

        Returns:
            FAISS index object
        """
        logger.info(
            f"Creating {index_type} index with dimension {self.embedding_dim}")

        if index_type == "Flat":
            # Simple flat index with L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)

        elif index_type == "IVF":
            # IVF index for faster search on large datasets
            nlist = nlist or Config.NLIST
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.index = index
        self.index_type = index_type

        logger.info(f"{index_type} index created successfully")
        return index

    def train_index(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices)

        Args:
            embeddings: Training embeddings array
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")

        if self.index_type == "IVF":
            logger.info(
                f"Training IVF index with {len(embeddings)} samples...")

            # Ensure embeddings are contiguous and float32
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))

            # Train the index
            self.index.train(embeddings)

            logger.info("Index training completed")
        else:
            logger.info("Flat index doesn't require training")

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict] = None):
        """
        Add embeddings to the index

        Args:
            embeddings: Embeddings to add
            metadata: Optional metadata for each embedding
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")

        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        logger.info(f"Adding {len(embeddings)} embeddings to index...")

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        if metadata:
            if self.metadata is None:
                self.metadata = []
            self.metadata.extend(metadata)

        logger.info(f"Total embeddings in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty or not initialized")

        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(
            query_embedding.astype('float32'))

        # Set nprobe for IVF indices
        if self.index_type == "IVF":
            self.index.nprobe = Config.NPROBE

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        return distances[0], indices[0]

    def search_with_metadata(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search and return results with metadata

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return

        Returns:
            List of result dictionaries with scores and metadata
        """
        distances, indices = self.search(query_embedding, top_k)

        results = []
        for idx, (dist, index) in enumerate(zip(distances, indices)):
            if index == -1:  # FAISS returns -1 for invalid indices
                continue

            result = {
                "rank": idx + 1,
                # Convert L2 distance to similarity score
                "score": float(1.0 / (1.0 + dist)),
                "distance": float(dist),
                "index": int(index)
            }

            # Add metadata if available
            if self.metadata and index < len(self.metadata):
                result["metadata"] = self.metadata[index]

            results.append(result)

        return results

    def save_index(self, index_name: str = "main"):
        """
        Save index and metadata to disk

        Args:
            index_name: Name for the index files
        """
        if self.index is None:
            raise ValueError("No index to save")

        index_path = Config.get_index_path(index_name)
        metadata_path = Config.get_metadata_path(index_name)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Index saved to {index_path}")

        # Save metadata
        if self.metadata:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Metadata saved to {metadata_path}")

    def load_index(self, index_name: str = "main") -> bool:
        """
        Load index and metadata from disk

        Args:
            index_name: Name of the index to load

        Returns:
            True if successful, False otherwise
        """
        index_path = Config.get_index_path(index_name)
        metadata_path = Config.get_metadata_path(index_name)

        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Index loaded from {index_path}")
        logger.info(f"Index contains {self.index.ntotal} embeddings")

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")

        return True

    def get_stats(self) -> Dict:
        """
        Get index statistics

        Returns:
            Dictionary with index stats
        """
        if self.index is None:
            return {"status": "No index initialized"}

        stats = {
            "total_embeddings": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "is_trained": self.index.is_trained if self.index_type == "IVF" else True,
            "has_metadata": self.metadata is not None,
            "metadata_count": len(self.metadata) if self.metadata else 0
        }

        return stats

    def reset(self):
        """Reset the index"""
        self.index = None
        self.metadata = None
        logger.info("Index reset")
