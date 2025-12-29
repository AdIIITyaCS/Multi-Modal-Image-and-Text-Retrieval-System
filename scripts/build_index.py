"""
Build Index Script
Script to build FAISS indices from corpus data
"""

import logging
from config.settings import Config
from indexing.indexer import MultiModalIndexer
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Build FAISS indices for multi-modal retrieval'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=str(Config.CORPUS_DIR),
        help='Path to corpus data directory'
    )

    parser.add_argument(
        '--index-name',
        type=str,
        default='main',
        help='Name for the index files'
    )

    parser.add_argument(
        '--index-type',
        type=str,
        choices=['images', 'texts', 'both'],
        default='both',
        help='Type of data to index'
    )

    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Maximum number of items to index (for testing)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)

    logger.info("="*60)
    logger.info("Multi-Modal Index Builder")
    logger.info("="*60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Index name: {args.index_name}")
    logger.info(f"Index type: {args.index_type}")

    # Initialize indexer
    indexer = MultiModalIndexer()

    # Index images
    if args.index_type in ['images', 'both']:
        image_dir = data_path / 'images' if data_path.is_dir() else data_path

        if image_dir.exists():
            logger.info("\n" + "="*60)
            logger.info("Indexing Images")
            logger.info("="*60)

            try:
                count = indexer.index_images(
                    image_dir,
                    "images",
                    max_images=args.max_items
                )
                logger.info(f"✓ Successfully indexed {count} images")
            except Exception as e:
                logger.error(f"✗ Error indexing images: {e}")
        else:
            logger.warning(f"Image directory not found: {image_dir}")

    # Index texts
    if args.index_type in ['texts', 'both']:
        text_dir = data_path / 'texts' if data_path.is_dir() else data_path

        if text_dir.exists():
            logger.info("\n" + "="*60)
            logger.info("Indexing Texts")
            logger.info("="*60)

            try:
                count = indexer.index_texts(
                    text_dir,
                    "texts",
                    max_texts=args.max_items
                )
                logger.info(f"✓ Successfully indexed {count} texts")
            except Exception as e:
                logger.error(f"✗ Error indexing texts: {e}")
        else:
            logger.warning(f"Text directory not found: {text_dir}")

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("Index Statistics")
    logger.info("="*60)

    stats = indexer.get_stats()
    for index_type, index_stats in stats.items():
        logger.info(f"\n{index_type.upper()}:")
        for key, value in index_stats.items():
            logger.info(f"  {key}: {value}")

    logger.info("\n" + "="*60)
    logger.info("Indexing Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
