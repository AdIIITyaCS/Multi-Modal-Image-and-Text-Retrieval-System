# API Usage Guide

## Getting Started

### 1. Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd 11reaidy.io

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 2. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates sample text data in `data/corpus/texts/`. For images, add your own to `data/corpus/images/`.

### 3. Build Indices

```bash
# Index both images and texts
python scripts/build_index.py --data-path ./data/corpus

# Index only images
python scripts/build_index.py --data-path ./data/corpus --index-type images

# Index only texts
python scripts/build_index.py --data-path ./data/corpus --index-type texts
```

### 4. Start the API Server

```bash
python app.py
```

Server will start at `http://localhost:5000`

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /api/health`

**Description:** Check if the API is running

**Example:**
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Multi-Modal Retrieval System is running"
}
```

### 2. System Status

**Endpoint:** `GET /api/status`

**Description:** Get system status and index statistics

**Example:**
```bash
curl http://localhost:5000/api/status
```

**Response:**
```json
{
  "status": "online",
  "version": "1.0.0",
  "embedding_dimension": 512,
  "indices": {
    "images": {
      "total_embeddings": 150,
      "embedding_dimension": 512,
      "index_type": "IVF",
      "is_trained": true
    },
    "texts": {
      "total_embeddings": 20,
      "embedding_dimension": 512,
      "index_type": "IVF",
      "is_trained": true
    }
  }
}
```

### 3. Create/Update Index

**Endpoint:** `POST /api/index`

**Description:** Build or update search indices

**Request Body:**
```json
{
  "data_path": "./data/corpus",
  "index_name": "main",
  "index_type": "both"
}
```

**Parameters:**
- `data_path` (required): Path to corpus directory
- `index_name` (optional): Name for the index (default: "main")
- `index_type` (optional): "images", "texts", or "both" (default: "both")

**Example:**
```bash
curl -X POST http://localhost:5000/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "./data/corpus",
    "index_name": "main",
    "index_type": "both"
  }'
```

**Response:**
```json
{
  "images": {
    "status": "success",
    "count": 150
  },
  "texts": {
    "status": "success",
    "count": 20
  }
}
```

### 4. Search with Text Query

**Endpoint:** `POST /api/search/text`

**Description:** Search using a text query

**Request Body:**
```json
{
  "query": "beautiful sunset over mountains",
  "top_k": 5,
  "search_type": "images"
}
```

**Parameters:**
- `query` (required): Text query string
- `top_k` (optional): Number of results (default: 5, max: 100)
- `search_type` (optional): "images", "texts", or "both" (default: "images")

**Example:**
```bash
curl -X POST http://localhost:5000/api/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "beautiful sunset over mountains",
    "top_k": 5,
    "search_type": "images"
  }'
```

**Response:**
```json
{
  "query": "beautiful sunset over mountains",
  "query_type": "text",
  "results": {
    "images": [
      {
        "rank": 1,
        "score": 0.8542,
        "distance": 0.1234,
        "index": 42,
        "metadata": {
          "type": "image",
          "path": "/path/to/image.jpg",
          "filename": "sunset.jpg"
        }
      },
      {
        "rank": 2,
        "score": 0.8123,
        "distance": 0.1567,
        "index": 15,
        "metadata": {
          "type": "image",
          "path": "/path/to/mountain.jpg",
          "filename": "mountain.jpg"
        }
      }
    ]
  }
}
```

### 5. Search with Image Query

**Endpoint:** `POST /api/search/image`

**Description:** Search using an image query

**Form Data:**
- `image` (required): Image file
- `top_k` (optional): Number of results (default: 5)
- `search_type` (optional): "images", "texts", or "both" (default: "images")

**Example:**
```bash
curl -X POST http://localhost:5000/api/search/image \
  -F "image=@path/to/query_image.jpg" \
  -F "top_k=5" \
  -F "search_type=texts"
```

**Response:**
```json
{
  "query": "query_image.jpg",
  "query_type": "image",
  "results": {
    "texts": [
      {
        "rank": 1,
        "score": 0.7892,
        "distance": 0.2345,
        "index": 7,
        "metadata": {
          "type": "text",
          "text": "A beautiful sunset over mountains...",
          "source": "sample_texts.json",
          "category": "nature"
        }
      }
    ]
  }
}
```

## Python Client Examples

### Example 1: Text Search for Images

```python
import requests

url = "http://localhost:5000/api/search/text"

data = {
    "query": "city skyline at night",
    "top_k": 10,
    "search_type": "images"
}

response = requests.post(url, json=data)
results = response.json()

for result in results['results']['images']:
    print(f"Score: {result['score']:.4f}")
    print(f"Image: {result['metadata']['filename']}")
    print()
```

### Example 2: Image Search for Similar Images

```python
import requests

url = "http://localhost:5000/api/search/image"

with open('query_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'top_k': 5, 'search_type': 'images'}
    
    response = requests.post(url, files=files, data=data)
    results = response.json()

for result in results['results']['images']:
    print(f"Similar image: {result['metadata']['path']}")
```

### Example 3: Cross-Modal Search (Image → Text)

```python
import requests

url = "http://localhost:5000/api/search/image"

with open('sunset_photo.jpg', 'rb') as f:
    files = {'image': f}
    data = {'top_k': 3, 'search_type': 'texts'}
    
    response = requests.post(url, files=files, data=data)
    results = response.json()

for result in results['results']['texts']:
    print(f"Matching text: {result['metadata']['text'][:100]}...")
    print(f"Score: {result['score']:.4f}\n")
```

## Command-Line Testing

### Test with Sample Queries

```bash
# Text query for images
python scripts/test_queries.py \
  --query "beautiful tropical beach" \
  --query-type text \
  --search-in images \
  --top-k 5

# Text query for texts
python scripts/test_queries.py \
  --query "office workspace" \
  --query-type text \
  --search-in texts \
  --top-k 3

# Image query for similar images
python scripts/test_queries.py \
  --query path/to/image.jpg \
  --query-type image \
  --search-in images \
  --top-k 10
```

## Response Format

All endpoints return JSON responses with the following general structure:

### Success Response
```json
{
  "query": "query string or filename",
  "query_type": "text|image",
  "results": {
    "images": [...],
    "texts": [...]
  }
}
```

### Error Response
```json
{
  "error": "Error message describing what went wrong"
}
```

### Result Object
Each result in the results array contains:
```json
{
  "rank": 1,
  "score": 0.8542,
  "distance": 0.1234,
  "index": 42,
  "metadata": {
    // Type-specific metadata
  }
}
```

## Best Practices

1. **Index Updates**: Rebuild indices after adding new data
2. **Query Length**: Keep text queries under 512 characters
3. **Image Format**: Use common formats (JPG, PNG, GIF)
4. **Image Size**: Images are automatically resized to 224x224
5. **Top-K**: Use reasonable values (5-20 for most cases)
6. **Error Handling**: Always check response status codes

## Performance Tips

1. **Batch Processing**: Index data in batches for large datasets
2. **IVF Index**: Use IVF index type for faster search on large corpora (>1000 items)
3. **Caching**: Text query embeddings are cached automatically
4. **Image Size**: Smaller images process faster

## Troubleshooting

### Index Not Found
```json
{"error": "Image index not loaded"}
```
**Solution:** Run `python scripts/build_index.py` to build indices

### File Too Large
```json
{"error": "File too large. Maximum size is 16MB"}
```
**Solution:** Compress or resize your image

### No Results
**Check:**
- Index is properly built
- Query is meaningful
- Corpus contains relevant data

## Rate Limiting

Currently, there are no rate limits, but consider implementing them for production use.

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the main README.md
3. Open an issue on GitHub
