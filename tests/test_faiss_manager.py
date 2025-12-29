"""
Unit Tests for FAISS Manager
"""

from config.settings import Config
from indexing.faiss_manager import FAISSManager
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFAISSManager:
    """Test cases for FAISSManager"""

    @pytest.fixture
    def manager(self):
        """Create FAISSManager instance"""
        return FAISSManager(embedding_dim=512)

    def test_manager_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager is not None
        assert manager.embedding_dim == 512
        assert manager.index is None

    def test_create_flat_index(self, manager):
        """Test creating flat index"""
        index = manager.create_index(index_type="Flat")

        assert index is not None
        assert manager.index is not None
        assert manager.index_type == "Flat"

    def test_create_ivf_index(self, manager):
        """Test creating IVF index"""
        index = manager.create_index(index_type="IVF", nlist=10)

        assert index is not None
        assert manager.index is not None
        assert manager.index_type == "IVF"

    def test_add_embeddings(self, manager):
        """Test adding embeddings to index"""
        manager.create_index(index_type="Flat")

        # Generate random embeddings
        embeddings = np.random.rand(100, 512).astype('float32')
        metadata = [{"id": i} for i in range(100)]

        manager.add_embeddings(embeddings, metadata)

        assert manager.index.ntotal == 100
        assert len(manager.metadata) == 100

    def test_search(self, manager):
        """Test searching in index"""
        manager.create_index(index_type="Flat")

        # Add embeddings
        embeddings = np.random.rand(100, 512).astype('float32')
        manager.add_embeddings(embeddings)

        # Search with first embedding
        query = embeddings[0:1]
        distances, indices = manager.search(query, top_k=5)

        assert len(distances) == 5
        assert len(indices) == 5
        # First result should be the query itself (distance ~0)
        assert np.isclose(distances[0], 0.0, atol=1e-5)
        assert indices[0] == 0

    def test_search_with_metadata(self, manager):
        """Test searching with metadata"""
        manager.create_index(index_type="Flat")

        # Add embeddings with metadata
        embeddings = np.random.rand(50, 512).astype('float32')
        metadata = [{"id": i, "name": f"item_{i}"} for i in range(50)]
        manager.add_embeddings(embeddings, metadata)

        # Search
        query = embeddings[0:1]
        results = manager.search_with_metadata(query, top_k=3)

        assert len(results) == 3
        assert "score" in results[0]
        assert "metadata" in results[0]
        assert results[0]["metadata"]["id"] == 0

    def test_get_stats(self, manager):
        """Test getting index statistics"""
        manager.create_index(index_type="Flat")

        embeddings = np.random.rand(25, 512).astype('float32')
        manager.add_embeddings(embeddings)

        stats = manager.get_stats()

        assert stats["total_embeddings"] == 25
        assert stats["embedding_dimension"] == 512
        assert stats["index_type"] == "Flat"

    def test_reset(self, manager):
        """Test resetting index"""
        manager.create_index(index_type="Flat")
        embeddings = np.random.rand(10, 512).astype('float32')
        manager.add_embeddings(embeddings)

        assert manager.index is not None

        manager.reset()

        assert manager.index is None
        assert manager.metadata is None


class TestFAISSManagerIntegration:
    """Integration tests for FAISSManager"""

    def test_full_workflow(self):
        """Test complete index workflow"""
        # Create manager
        manager = FAISSManager(embedding_dim=128)

        # Create index
        manager.create_index(index_type="Flat")

        # Generate and add data
        n_items = 1000
        embeddings = np.random.rand(n_items, 128).astype('float32')
        metadata = [{"index": i, "value": f"item_{i}"} for i in range(n_items)]

        manager.add_embeddings(embeddings, metadata)

        # Search
        query = np.random.rand(1, 128).astype('float32')
        results = manager.search_with_metadata(query, top_k=10)

        # Verify results
        assert len(results) == 10
        for result in results:
            assert "score" in result
            assert "metadata" in result
            assert "index" in result["metadata"]
            assert "value" in result["metadata"]
