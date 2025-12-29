"""
Main Flask Application
Entry point for the Multi-Modal Retrieval System API
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging
from pathlib import Path

from config.settings import Config
from api.routes import api_bp, initialize_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Flask application

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)

    # Load configuration
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Enable CORS
    CORS(app)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Initialize API components
    with app.app_context():
        initialize_components()

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            "name": "Multi-Modal Image and Text Retrieval System",
            "version": "1.0.0",
            "description": "A sophisticated retrieval system for cross-modal search",
            "endpoints": {
                "GET /": "API information",
                "GET /api/status": "System status and statistics",
                "GET /api/health": "Health check",
                "POST /api/index": "Create or update search indices",
                "POST /api/search/text": "Search using text query",
                "POST /api/search/image": "Search using image query"
            },
            "documentation": "https://github.com/yourusername/multimodal-retrieval"
        })

    logger.info("Flask application created successfully")
    return app


def main():
    """
    Run the Flask application
    """
    # Ensure directories exist
    Config.create_directories()

    # Create app
    app = create_app()

    # Run server
    logger.info(f"Starting server on port {Config.FLASK_PORT}")
    logger.info(f"Environment: {Config.FLASK_ENV}")

    app.run(
        host='0.0.0.0',
        port=Config.FLASK_PORT,
        debug=(Config.FLASK_ENV == 'development')
    )


if __name__ == '__main__':
    main()
