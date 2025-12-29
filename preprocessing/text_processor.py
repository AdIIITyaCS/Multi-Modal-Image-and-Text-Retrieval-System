"""
Text Preprocessing Module
Handles text cleaning and normalization
"""

import re
import unicodedata
from typing import List, Union
import logging

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles all text preprocessing operations
    """

    def __init__(self, max_length: int = None):
        """
        Initialize text processor

        Args:
            max_length: Maximum text length (default from config)
        """
        self.max_length = max_length or Config.MAX_TEXT_LENGTH

    def preprocess(self, text: str, clean: bool = True) -> str:
        """
        Preprocess text for encoding

        Args:
            text: Input text
            clean: Whether to apply cleaning operations

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        if clean:
            text = self._clean_text(text)

        # Normalize unicode
        text = self._normalize_unicode(text)

        # Truncate if needed
        text = self._truncate(text)

        return text

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)

        return text

    def _truncate(self, text: str) -> str:
        """
        Truncate text to maximum length

        Args:
            text: Input text

        Returns:
            Truncated text
        """
        if len(text) > self.max_length:
            # Truncate at word boundary
            text = text[:self.max_length]
            last_space = text.rfind(' ')
            if last_space > 0:
                text = text[:last_space]
            text += '...'

        return text

    def batch_preprocess(self, texts: List[str], clean: bool = True) -> List[str]:
        """
        Preprocess a batch of texts

        Args:
            texts: List of input texts
            clean: Whether to apply cleaning

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, clean=clean) for text in texts]

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation)

        Args:
            text: Input text
            top_n: Number of keywords to extract

        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        # Tokenize and count words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = {}

        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in keywords[:top_n]]

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def split_into_chunks(self, text: str, chunk_size: int = 500,
                          overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                search_start = max(start, end - 100)
                search_end = min(len(text), end + 100)
                chunk_text = text[search_start:search_end]

                # Find last sentence end
                sentence_ends = [m.end()
                                 for m in re.finditer(r'[.!?]\s', chunk_text)]
                if sentence_ends:
                    last_end = sentence_ends[-1]
                    end = search_start + last_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def validate_text(self, text: str, min_length: int = 1) -> bool:
        """
        Validate if text is suitable for processing

        Args:
            text: Text to validate
            min_length: Minimum required length

        Returns:
            True if valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        if len(text.strip()) < min_length:
            return False

        return True
