"""
Request Validators
Validate and sanitize API requests
"""

from flask import request
from typing import Dict, Tuple, Optional
import logging

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates API request parameters"""

    @staticmethod
    def validate_search_request(data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate search request

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if 'query' not in data:
            return False, "Missing required field: query"

        # Validate query
        query = data['query']
        if not query or (isinstance(query, str) and not query.strip()):
            return False, "Query cannot be empty"

        # Validate top_k
        top_k = data.get('top_k', Config.DEFAULT_TOP_K)
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > Config.MAX_TOP_K:
                return False, f"top_k must be between 1 and {Config.MAX_TOP_K}"
        except (ValueError, TypeError):
            return False, "top_k must be an integer"

        # Validate search_type
        search_type = data.get('search_type', 'images')
        if search_type not in ['images', 'texts', 'both']:
            return False, "search_type must be 'images', 'texts', or 'both'"

        return True, None

    @staticmethod
    def validate_index_request(data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate index creation request

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if 'data_path' not in data:
            return False, "Missing required field: data_path"

        # Validate index_name
        index_name = data.get('index_name', 'main')
        if not isinstance(index_name, str) or not index_name.strip():
            return False, "index_name must be a non-empty string"

        # Validate index_type
        index_type = data.get('index_type', 'both')
        if index_type not in ['images', 'texts', 'both']:
            return False, "index_type must be 'images', 'texts', or 'both'"

        return True, None

    @staticmethod
    def validate_image_file(file) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded image file

        Args:
            file: File from request

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file:
            return False, "No file provided"

        if file.filename == '':
            return False, "No file selected"

        # Check file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}
        if '.' not in file.filename:
            return False, "File has no extension"

        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in allowed_extensions:
            return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"

        return True, None

    @staticmethod
    def sanitize_string(s: str, max_length: int = 1000) -> str:
        """
        Sanitize string input

        Args:
            s: Input string
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(s, str):
            s = str(s)

        # Truncate
        s = s[:max_length]

        # Remove control characters
        s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\r\t')

        return s.strip()
