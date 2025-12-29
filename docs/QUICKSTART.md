# Quick Start Guide

Get up and running with the Multi-Modal Image and Text Retrieval System in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of RAM
- 1GB of free disk space

## Installation (5 minutes)

### Step 1: Set Up Environment

```bash
# Navigate to project directory
cd 11reaidy.io

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning)
- FAISS (Vector Search)
- Flask (API Server)
- Sentence Transformers (Pre-trained Models)
- And other required packages

### Step 3: Configure Environment

```bash
# Copy example configuration
copy .env.example .env  # Windows
# or
cp .env.example .env    # macOS/Linux
```

Default configuration works out of the box!

## First Run (2 minutes)

### Step 1: Generate Sample Data

```bash
python scripts\generate_sample_data.py
```

This creates 20 sample text descriptions in `data/corpus/texts/`.

### Step 2: (Optional) Add Sample Images

Add some images to `data/corpus/images/`:
- Use your own images, or
- Download from free sources like Unsplash, Pexels, or Pixabay

Skip this step if you only want to test text search.

### Step 3: Build the Index

```bash
python scripts\build_index.py --data-path ./data/corpus
```

This will:
- Load all texts and images
- Generate embeddings
- Build searchable FAISS indices
- Save indices to `data/indices/`

Expected time: 30 seconds for sample data

### Step 4: Start the Server

```bash
python app.py
```

You should see:
```
* Running on http://0.0.0.0:5000
```

## Your First Search (1 minute)

Open a new terminal and try these examples:

### Example 1: Check Server Status

```bash
curl http://localhost:5000/api/health
```

### Example 2: Search Texts

```bash
curl -X POST http://localhost:5000/api/search/text ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"beautiful sunset\", \"top_k\": 3, \"search_type\": \"texts\"}"
```

For PowerShell:
```powershell
$body = @{
    query = "beautiful sunset"
    top_k = 3
    search_type = "texts"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/search/text" -Method Post -Body $body -ContentType "application/json"
```

### Example 3: Search Images (if you added images)

```bash
curl -X POST http://localhost:5000/api/search/text ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"city skyline\", \"top_k\": 5, \"search_type\": \"images\"}"
```

## Next Steps

### Test Different Queries

```bash
# Use the test script
python scripts\test_queries.py ^
  --query "tropical beach" ^
  --search-in texts ^
  --top-k 5
```

### View Results

Check `logs/query_results.json` for detailed results.

### Try Image Search

```bash
# Search with an image
curl -X POST http://localhost:5000/api/search/image ^
  -F "image=@path\to\your\image.jpg" ^
  -F "top_k=5" ^
  -F "search_type=texts"
```

## What Just Happened?

1. **Embeddings Generated**: Your texts and images were converted to 512-dimensional vectors
2. **Index Built**: FAISS created an efficient search structure
3. **Server Started**: Flask API is ready to handle search requests
4. **Search Working**: You can now find similar content across modalities!

## Common Issues

### Issue: "No module named 'tensorflow'"
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Index not found"
**Solution:** Build the index first:
```bash
python scripts\build_index.py --data-path ./data/corpus
```

### Issue: "CUDA not found" warnings
**Solution:** This is normal! The system will use CPU, which works fine for testing.

### Issue: Port 5000 already in use
**Solution:** Change the port in `.env`:
```
FLASK_PORT=5001
```

## Architecture Overview

```
Your Query (Text/Image)
         ↓
   Encoder Model
         ↓
   Embedding Vector
         ↓
   FAISS Search
         ↓
   Similar Items
```

## File Structure

```
11reaidy.io/
├── app.py              # Start server here
├── config/             # Configuration
├── models/             # Encoder models
├── indexing/           # FAISS index management
├── preprocessing/      # Data preprocessing
├── api/                # REST API endpoints
├── scripts/            # Utility scripts
├── data/               # Your data
│   ├── corpus/         # Source data
│   └── indices/        # Built indices
└── logs/               # Query results
```

## Performance Expectations

With the sample data:
- **Indexing**: ~30 seconds
- **Search**: <50ms per query
- **Memory**: ~500MB

With 10,000 images:
- **Indexing**: ~10-15 minutes
- **Search**: <100ms per query
- **Memory**: ~2GB

## Configuration Options

Edit `.env` to customize:

```bash
# Change embedding model
TEXT_MODEL=sentence-transformers/clip-ViT-B-32-multilingual-v1

# Change server port
FLASK_PORT=5000

# Change batch size (for memory management)
BATCH_SIZE=32

# Change image size
MAX_IMAGE_SIZE=224
```

## Development Mode

The server runs in development mode by default:
- Auto-reload on code changes
- Detailed error messages
- CORS enabled

For production, set in `.env`:
```
FLASK_ENV=production
```

## API Endpoints

- `GET /` - API information
- `GET /api/health` - Health check
- `GET /api/status` - System status
- `POST /api/index` - Build/update index
- `POST /api/search/text` - Text query search
- `POST /api/search/image` - Image query search

Full API documentation: see `docs/API_GUIDE.md`

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

## Get Help

1. Check logs in `logs/` directory
2. Review `README.md` for detailed documentation
3. See `docs/API_GUIDE.md` for API details
4. Open an issue on GitHub

## Success!

You now have a working multi-modal retrieval system! 🎉

Try experimenting with:
- Different query texts
- Your own images
- Cross-modal searches (text → image, image → text)
- Adjusting top-k values

Enjoy exploring your multi-modal search system!
