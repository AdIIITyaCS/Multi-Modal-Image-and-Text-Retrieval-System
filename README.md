# Multi-Modal Image and Text Retrieval System

This project is a multi-modal retrieval system that takes a text query or an image query and returns the most similar images/texts from a stored dataset using CLIP dual-encoders and FAISS vector similarity search.

## Tech Stack
Python, Flask, FAISS, Sentence-Transformers (CLIP), TensorFlow (deps), OpenCV/Pillow, NumPy

## How to Run (Windows)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.generate_sample_data
python -m scripts.build_index --data-path .\data\corpus
python app.py 
```
text->image
```bat
curl -X POST http://127.0.0.1:5000/api/search/text -H "Content-Type: application/json" -d "{\"query\":\"city skyline\",\"top_k\":3,\"search_type\":\"images\"}"
```

image->image
```bat
curl -X POST "http://127.0.0.1:5000/api/search/image" -F "image=@data/corpus/images/Urban & Architecture_8.jpg" -F "top_k=5" -F "search_type=images"
```
API Endpoints:
GET /api/health
GET /api/status
POST /api/search/text (text query)
POST /api/search/image (image query)
