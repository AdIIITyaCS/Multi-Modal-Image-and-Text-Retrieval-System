"""
Utility Helper Functions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timeit(func):
    """
    Decorator to measure function execution time

    Args:
        func: Function to measure

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def load_json(file_path: Path) -> Dict:
    """
    Load JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}


def save_json(data: Dict, file_path: Path, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_file_stats(file_path: Path) -> Dict:
    """
    Get file statistics

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file stats
    """
    if not file_path.exists():
        return {"exists": False}

    stat = file_path.stat()

    return {
        "exists": True,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "created": stat.st_ctime,
        "modified": stat.st_mtime
    }


def batch_iterator(items: List[Any], batch_size: int):
    """
    Iterate over items in batches

    Args:
        items: List of items
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def validate_path(path: Path, must_exist: bool = False) -> bool:
    """
    Validate a file path

    Args:
        path: Path to validate
        must_exist: Whether path must exist

    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(path)

        if must_exist and not path.exists():
            logger.error(f"Path does not exist: {path}")
            return False

        return True

    except Exception as e:
        logger.error(f"Invalid path {path}: {e}")
        return False


def create_response(success: bool, data: Any = None, error: str = None) -> Dict:
    """
    Create standardized API response

    Args:
        success: Whether operation succeeded
        data: Response data
        error: Error message if failed

    Returns:
        Response dictionary
    """
    response = {
        "success": success,
        "timestamp": time.time()
    }

    if data is not None:
        response["data"] = data

    if error is not None:
        response["error"] = error

    return response


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def chunks(lst: List, n: int):
    """
    Yield successive n-sized chunks from list

    Args:
        lst: Input list
        n: Chunk size

    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ProgressTracker:
    """Simple progress tracker for long operations"""

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker

        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress"""
        self.current += n

        if self.current % max(1, self.total // 20) == 0:
            self._print_progress()

    def _print_progress(self):
        """Print progress"""
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        logger.info(
            f"{self.description}: {self.current}/{self.total} "
            f"({percent:.1f}%) - {rate:.1f} items/sec"
        )

    def finish(self):
        """Finish and print final statistics"""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        logger.info(
            f"{self.description} completed: {self.current} items "
            f"in {elapsed:.1f}s ({rate:.1f} items/sec)"
        )
