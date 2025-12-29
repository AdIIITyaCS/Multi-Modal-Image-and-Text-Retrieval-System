"""
Test Queries Script
Script to test the retrieval system with sample queries
"""

import logging
from config.settings import Config
from indexing.faiss_manager import FAISSManager
from models.embedder import MultiModalEmbedder
import argparse
import sys
from pathlib import Path
import json

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
        description='Test multi-modal retrieval system with queries'
    )

    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query text or path to query image'
    )

    parser.add_argument(
        '--query-type',
        type=str,
        choices=['text', 'image', 'auto'],
        default='auto',
        help='Type of query'
    )

    parser.add_argument(
        '--search-in',
        type=str,
        choices=['images', 'texts', 'both'],
        default='images',
        help='Where to search'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return'
    )

    parser.add_argument(
        '--index-name',
        type=str,
        default='main',
        help='Name of the index to use'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    logger.info("="*60)
    logger.info("Multi-Modal Retrieval System - Query Test")
    logger.info("="*60)

    # Initialize embedder
    logger.info("Initializing embedder...")
    embedder = MultiModalEmbedder()

    # Determine query type
    query_type = args.query_type
    if query_type == 'auto':
        query_path = Path(args.query)
        if query_path.exists() and query_path.is_file():
            query_type = 'image'
        else:
            query_type = 'text'

    logger.info(f"Query type: {query_type}")
    logger.info(f"Query: {args.query}")

    # Generate query embedding
    logger.info("Generating query embedding...")
    if query_type == 'text':
        query_embedding = embedder.embed_texts(args.query)[0]
    else:
        query_embedding = embedder.embed_images([args.query])[0]

    logger.info(f"Embedding shape: {query_embedding.shape}")

    # Search
    results = {}

    if args.search_in in ['images', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Searching in Images")
        logger.info("="*60)

        try:
            image_index = FAISSManager(embedder.get_embedding_dimension())
            if image_index.load_index(f"{args.index_name}_images"):
                image_results = image_index.search_with_metadata(
                    query_embedding,
                    args.top_k
                )
                results['images'] = image_results

                logger.info(f"\nTop {args.top_k} Image Results:")
                for i, result in enumerate(image_results, 1):
                    logger.info(f"\n{i}. Score: {result['score']:.4f}")
                    if 'metadata' in result:
                        logger.info(
                            f"   Path: {result['metadata'].get('path', 'N/A')}")
                        logger.info(
                            f"   Filename: {result['metadata'].get('filename', 'N/A')}")
            else:
                logger.warning("Image index not found")
        except Exception as e:
            logger.error(f"Error searching images: {e}")

    if args.search_in in ['texts', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Searching in Texts")
        logger.info("="*60)

        try:
            text_index = FAISSManager(embedder.get_embedding_dimension())
            if text_index.load_index(f"{args.index_name}_texts"):
                text_results = text_index.search_with_metadata(
                    query_embedding,
                    args.top_k
                )
                results['texts'] = text_results

                logger.info(f"\nTop {args.top_k} Text Results:")
                for i, result in enumerate(text_results, 1):
                    logger.info(f"\n{i}. Score: {result['score']:.4f}")
                    if 'metadata' in result:
                        logger.info(
                            f"   Source: {result['metadata'].get('source', 'N/A')}")
                        text = result['metadata'].get('text', '')
                        if text:
                            preview = text[:100] + \
                                '...' if len(text) > 100 else text
                            logger.info(f"   Text: {preview}")
            else:
                logger.warning("Text index not found")
        except Exception as e:
            logger.error(f"Error searching texts: {e}")

    # Save results to file
    output_file = Path(__file__).parent.parent / 'logs' / 'query_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'query': args.query,
            'query_type': query_type,
            'search_in': args.search_in,
            'top_k': args.top_k,
            'results': results
        }, f, indent=2, default=str)

    logger.info(f"\n✓ Results saved to: {output_file}")

    logger.info("\n" + "="*60)
    logger.info("Query Test Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
