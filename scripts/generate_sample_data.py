"""
Sample Data Generator
Generates sample text data for testing the retrieval system
"""

from config.settings import Config
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_sample_texts():
    """Generate sample text data"""

    sample_texts = [
        {
            "text": "A beautiful sunset over mountains with orange and purple colors filling the sky",
            "category": "nature",
            "tags": ["sunset", "mountains", "landscape"]
        },
        {
            "text": "Modern city skyline at night with illuminated skyscrapers and reflections on water",
            "category": "urban",
            "tags": ["city", "night", "architecture"]
        },
        {
            "text": "Close-up of a red rose with water droplets on its petals in a garden",
            "category": "nature",
            "tags": ["flower", "rose", "macro"]
        },
        {
            "text": "Golden retriever puppy playing with a ball in a green park",
            "category": "animals",
            "tags": ["dog", "puppy", "park"]
        },
        {
            "text": "Fresh vegetables and fruits arranged on a wooden table for healthy eating",
            "category": "food",
            "tags": ["vegetables", "fruits", "healthy"]
        },
        {
            "text": "Snowy mountain peaks under a clear blue sky with hiking trail",
            "category": "nature",
            "tags": ["mountains", "snow", "hiking"]
        },
        {
            "text": "Tropical beach with white sand, palm trees, and turquoise water",
            "category": "nature",
            "tags": ["beach", "tropical", "ocean"]
        },
        {
            "text": "Modern office workspace with computer, notebook, and coffee on desk",
            "category": "business",
            "tags": ["office", "workspace", "technology"]
        },
        {
            "text": "Abstract geometric pattern with colorful triangles and circles",
            "category": "art",
            "tags": ["abstract", "geometric", "pattern"]
        },
        {
            "text": "Vintage car parked on a cobblestone street in an old European city",
            "category": "urban",
            "tags": ["car", "vintage", "europe"]
        },
        {
            "text": "Black and white portrait of an elderly person with wrinkled face showing wisdom",
            "category": "portrait",
            "tags": ["portrait", "elderly", "black and white"]
        },
        {
            "text": "Delicious pizza with cheese, pepperoni, and vegetables fresh from the oven",
            "category": "food",
            "tags": ["pizza", "food", "italian"]
        },
        {
            "text": "Autumn forest with colorful leaves in red, orange, and yellow tones",
            "category": "nature",
            "tags": ["autumn", "forest", "leaves"]
        },
        {
            "text": "Spaceship launching into space with fire and smoke at sunset",
            "category": "space",
            "tags": ["spaceship", "launch", "technology"]
        },
        {
            "text": "Cozy living room with fireplace, comfortable sofa, and warm lighting",
            "category": "interior",
            "tags": ["living room", "cozy", "fireplace"]
        },
        {
            "text": "Professional athlete running on track during competition in stadium",
            "category": "sports",
            "tags": ["running", "athlete", "competition"]
        },
        {
            "text": "Colorful hot air balloons floating over countryside landscape at dawn",
            "category": "travel",
            "tags": ["balloons", "countryside", "dawn"]
        },
        {
            "text": "Classical music orchestra performing on stage in concert hall",
            "category": "music",
            "tags": ["orchestra", "music", "concert"]
        },
        {
            "text": "Ancient temple ruins surrounded by jungle vegetation and moss",
            "category": "history",
            "tags": ["temple", "ruins", "ancient"]
        },
        {
            "text": "Mother and child playing together in park on sunny spring day",
            "category": "family",
            "tags": ["family", "child", "spring"]
        }
    ]

    return sample_texts


def save_sample_data():
    """Save sample data to files"""

    # Ensure directory exists
    text_dir = Config.CORPUS_DIR / 'texts'
    text_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save texts
    texts = generate_sample_texts()

    output_file = text_dir / 'sample_texts.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(texts)} sample texts to: {output_file}")

    # Create README
    readme_file = text_dir / 'README.txt'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("Sample Text Corpus for Multi-Modal Retrieval System\n")
        f.write("=" * 60 + "\n\n")
        f.write("This directory contains sample text data for testing.\n\n")
        f.write(f"Total texts: {len(texts)}\n\n")
        f.write("Categories:\n")

        categories = {}
        for text in texts:
            cat = text['category']
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            f.write(f"  - {cat}: {count}\n")

    print(f"✓ Created README at: {readme_file}")

    # Create instructions for images
    image_dir = Config.CORPUS_DIR / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    image_readme = image_dir / 'README.txt'
    with open(image_readme, 'w', encoding='utf-8') as f:
        f.write("Image Corpus Directory\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            "Place your image files (.jpg, .png, .gif, etc.) in this directory.\n\n")
        f.write("The system will automatically:\n")
        f.write("  1. Load all images from this directory\n")
        f.write("  2. Preprocess and resize them\n")
        f.write("  3. Generate embeddings\n")
        f.write("  4. Build a searchable index\n\n")
        f.write("For testing, you can download sample images from:\n")
        f.write("  - Unsplash: https://unsplash.com/\n")
        f.write("  - Pexels: https://www.pexels.com/\n")
        f.write("  - Pixabay: https://pixabay.com/\n\n")
        f.write("Make sure to comply with image licenses and usage rights.\n")

    print(f"✓ Created image directory guide at: {image_readme}")


if __name__ == '__main__':
    print("Generating sample data...\n")
    save_sample_data()
    print("\n✓ Sample data generation complete!")
    print("\nNext steps:")
    print("  1. (Optional) Add your own images to data/corpus/images/")
    print("  2. Run: python scripts/build_index.py")
    print("  3. Run: python app.py")
