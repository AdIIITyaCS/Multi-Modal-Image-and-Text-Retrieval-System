"""
API Routes for Multi-Modal Retrieval System
Implements all REST endpoints
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
import logging

from api.validators import RequestValidator
from models.embedder import MultiModalEmbedder
from indexing.faiss_manager import FAISSManager
from indexing.indexer import MultiModalIndexer
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components (will be set by app.py)
embedder = None
image_index_manager = None
text_index_manager = None
indexer = None


def initialize_components():
    """Initialize API components"""
    global embedder, image_index_manager, text_index_manager, indexer

    logger.info("Initializing API components...")

    embedder = MultiModalEmbedder()
    embedding_dim = embedder.get_embedding_dimension()

    image_index_manager = FAISSManager(embedding_dim)
    text_index_manager = FAISSManager(embedding_dim)
    indexer = MultiModalIndexer()

    # Try to load existing indices
    try:
        image_index_manager.load_index("images")
        logger.info("Loaded existing image index")
    except Exception as e:
        logger.info(f"No existing image index found: {e}")

    try:
        text_index_manager.load_index("texts")
        logger.info("Loaded existing text index")
    except Exception as e:
        logger.info(f"No existing text index found: {e}")

    logger.info("API components initialized")


@api_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get system status and statistics

    Returns:
        JSON with system status
    """
    try:
        status = {
            "status": "online",
            "version": "1.0.0",
            "embedding_dimension": Config.EMBEDDING_DIMENSION,
            "indices": {}
        }

        if image_index_manager and image_index_manager.index:
            status["indices"]["images"] = image_index_manager.get_stats()

        if text_index_manager and text_index_manager.index:
            status["indices"]["texts"] = text_index_manager.get_stats()

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/index', methods=['POST'])
def create_index():
    """
    Create or update search indices

    Request body:
        {
            "data_path": "/path/to/data",
            "index_name": "main",
            "index_type": "both|images|texts"
        }

    Returns:
        JSON with indexing results
    """
    try:
        data = request.get_json()

        # Validate request
        is_valid, error = RequestValidator.validate_index_request(data)
        if not is_valid:
            return jsonify({"error": error}), 400

        data_path = Path(data['data_path'])
        index_name = data.get('index_name', 'main')
        index_type = data.get('index_type', 'both')

        if not data_path.exists():
            return jsonify({"error": f"Data path not found: {data_path}"}), 404

        results = {}

        # Index images
        if index_type in ['images', 'both']:
            image_dir = data_path / 'images' if data_path.is_dir() else data_path
            if image_dir.exists():
                try:
                    count = indexer.index_images(
                        image_dir, f"{index_name}_images")
                    results['images'] = {
                        "status": "success",
                        "count": count
                    }
                    # Reload the index
                    image_index_manager.load_index(f"{index_name}_images")
                except Exception as e:
                    results['images'] = {
                        "status": "error",
                        "error": str(e)
                    }

        # Index texts
        if index_type in ['texts', 'both']:
            text_dir = data_path / 'texts' if data_path.is_dir() else data_path
            if text_dir.exists():
                try:
                    count = indexer.index_texts(
                        text_dir, f"{index_name}_texts")
                    results['texts'] = {
                        "status": "success",
                        "count": count
                    }
                    # Reload the index
                    text_index_manager.load_index(f"{index_name}_texts")
                except Exception as e:
                    results['texts'] = {
                        "status": "error",
                        "error": str(e)
                    }

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/search/text', methods=['POST'])
def search_with_text():
    """
    Search using a text query

    Request body:
        {
            "query": "search text",
            "top_k": 5,
            "search_type": "images|texts|both"
        }

    Returns:
        JSON with search results
    """
    try:
        data = request.get_json()

        # Validate request
        is_valid, error = RequestValidator.validate_search_request(data)
        if not is_valid:
            return jsonify({"error": error}), 400

        query = RequestValidator.sanitize_string(data['query'])
        top_k = int(data.get('top_k', Config.DEFAULT_TOP_K))
        search_type = data.get('search_type', 'images')

        # Generate query embedding
        query_embedding = embedder.embed_texts(query)[0]

        results = {}

        # Search images
        if search_type in ['images', 'both']:
            if image_index_manager and image_index_manager.index:
                try:
                    image_results = image_index_manager.search_with_metadata(
                        query_embedding, top_k
                    )
                    results['images'] = image_results
                except Exception as e:
                    results['images'] = {"error": str(e)}
            else:
                results['images'] = {"error": "Image index not loaded"}

        # Search texts
        if search_type in ['texts', 'both']:
            if text_index_manager and text_index_manager.index:
                try:
                    text_results = text_index_manager.search_with_metadata(
                        query_embedding, top_k
                    )
                    results['texts'] = text_results
                except Exception as e:
                    results['texts'] = {"error": str(e)}
            else:
                results['texts'] = {"error": "Text index not loaded"}

        response = {
            "query": query,
            "query_type": "text",
            "results": results
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in text search: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/search/image', methods=['POST'])
def search_with_image():
    """
    Search using an image query

    Form data:
        image: file
        top_k: int (optional)
        search_type: "images"|"texts"|"both" (optional)

    Returns:
        JSON with search results
    """
    try:
        # Validate file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        is_valid, error = RequestValidator.validate_image_file(file)
        if not is_valid:
            return jsonify({"error": error}), 400

        # Get parameters
        top_k = int(request.form.get('top_k', Config.DEFAULT_TOP_K))
        search_type = request.form.get('search_type', 'images')

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Generate query embedding
            query_embedding = embedder.embed_images([tmp_path])[0]

            results = {}

            # Search images
            if search_type in ['images', 'both']:
                if image_index_manager and image_index_manager.index:
                    try:
                        image_results = image_index_manager.search_with_metadata(
                            query_embedding, top_k
                        )
                        results['images'] = image_results
                    except Exception as e:
                        results['images'] = {"error": str(e)}
                else:
                    results['images'] = {"error": "Image index not loaded"}

            # Search texts
            if search_type in ['texts', 'both']:
                if text_index_manager and text_index_manager.index:
                    try:
                        text_results = text_index_manager.search_with_metadata(
                            query_embedding, top_k
                        )
                        results['texts'] = text_results
                    except Exception as e:
                        results['texts'] = {"error": str(e)}
                else:
                    results['texts'] = {"error": "Text index not loaded"}

            response = {
                "query": secure_filename(file.filename),
                "query_type": "image",
                "results": results
            }

            return jsonify(response), 200

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error in image search: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON with health status
    """
    return jsonify({
        "status": "healthy",
        "message": "Multi-Modal Retrieval System is running"
    }), 200
