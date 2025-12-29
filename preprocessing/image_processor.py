"""
Image Preprocessing Module
Handles image loading, resizing, and normalization
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import logging

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles all image preprocessing operations
    """

    def __init__(self, target_size: int = None):
        """
        Initialize image processor

        Args:
            target_size: Target size for images (default from config)
        """
        self.target_size = target_size or Config.MAX_IMAGE_SIZE
        self.channels = Config.IMAGE_CHANNELS

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file

        Args:
            image_path: Path to image file

        Returns:
            PIL Image
        """
        try:
            image_path = Path(image_path)

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load with PIL
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Preprocess image for encoding

        Args:
            image: PIL Image or numpy array

        Returns:
            Preprocessed PIL Image
        """
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize while maintaining aspect ratio
        image = self._resize_with_aspect_ratio(image)

        return image

    def _resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio

        Args:
            image: PIL Image

        Returns:
            Resized PIL Image
        """
        width, height = image.size

        # Calculate new dimensions
        if width > height:
            new_width = self.target_size
            new_height = int(height * (self.target_size / width))
        else:
            new_height = self.target_size
            new_width = int(width * (self.target_size / height))

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create square canvas and paste resized image
        canvas = Image.new(
            'RGB', (self.target_size, self.target_size), (255, 255, 255))

        # Calculate position to center the image
        x = (self.target_size - new_width) // 2
        y = (self.target_size - new_height) // 2

        canvas.paste(image, (x, y))

        return canvas

    def load_and_preprocess(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and preprocess an image in one step

        Args:
            image_path: Path to image

        Returns:
            Preprocessed PIL Image
        """
        image = self.load_image(image_path)
        return self.preprocess(image)

    def batch_preprocess(self, images: list) -> list:
        """
        Preprocess a batch of images

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img) for img in images]

    def validate_image(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Validate if an image is suitable for processing

        Args:
            image: Image to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    return False
                if len(image.shape) not in [2, 3]:
                    return False
            elif isinstance(image, Image.Image):
                if image.size[0] == 0 or image.size[1] == 0:
                    return False
            else:
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def save_image(image: Union[Image.Image, np.ndarray],
                   output_path: Union[str, Path]):
        """
        Save an image to file

        Args:
            image: Image to save
            output_path: Output file path
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
