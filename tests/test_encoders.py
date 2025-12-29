"""
Unit Tests for Encoders
"""

from models.encoders import DualEncoder, EmbeddingCache
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDualEncoder:
    """Test cases for DualEncoder"""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance"""
        return DualEncoder()

    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly"""
        assert encoder is not None
        assert encoder.text_model is not None
        assert encoder.image_model is not None
        assert encoder.embedding_dim > 0

    def test_encode_single_text(self, encoder):
        """Test encoding a single text"""
        text = "A beautiful sunset over mountains"
        embedding = encoder.encode_text(text)

        assert embedding.shape[0] == 1
        assert embedding.shape[1] == encoder.embedding_dim
        assert np.isfinite(embedding).all()

    def test_encode_multiple_texts(self, encoder):
        """Test encoding multiple texts"""
        texts = [
            "A beautiful sunset",
            "City skyline at night",
            "Fresh vegetables"
        ]
        embeddings = encoder.encode_text(texts)

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == encoder.embedding_dim
        assert np.isfinite(embeddings).all()

    def test_l2_normalization(self, encoder):
        """Test L2 normalization"""
        texts = ["test text"]
        embeddings = encoder.encode_text(texts, normalize=True)

        # Check that norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_compute_similarity(self, encoder):
        """Test similarity computation"""
        text1 = "beautiful sunset"
        text2 = "gorgeous sunrise"
        text3 = "computer keyboard"

        emb1 = encoder.encode_text(text1)
        emb2 = encoder.encode_text(text2)
        emb3 = encoder.encode_text(text3)

        # Similar texts should have higher similarity
        sim_similar = encoder.compute_similarity(emb1, emb2)
        sim_different = encoder.compute_similarity(emb1, emb3)

        assert sim_similar > sim_different


class TestEmbeddingCache:
    """Test cases for EmbeddingCache"""

    @pytest.fixture
    def cache(self):
        """Create cache instance"""
        return EmbeddingCache(max_size=3)

    def test_cache_initialization(self, cache):
        """Test cache initializes correctly"""
        assert cache is not None
        assert cache.max_size == 3

    def test_set_and_get_text_embedding(self, cache):
        """Test setting and getting text embeddings"""
        text = "test text"
        embedding = np.random.rand(512)

        cache.set_text_embedding(text, embedding)
        retrieved = cache.get_text_embedding(text)

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_cache_max_size(self, cache):
        """Test cache respects max size"""
        for i in range(5):
            text = f"text {i}"
            embedding = np.random.rand(512)
            cache.set_text_embedding(text, embedding)

        # Cache should only have max_size items
        assert len(cache.text_cache) == cache.max_size

    def test_clear_cache(self, cache):
        """Test clearing cache"""
        cache.set_text_embedding("test", np.random.rand(512))
        assert len(cache.text_cache) > 0

        cache.clear()
        assert len(cache.text_cache) == 0
