"""Download models at build time so the container starts fast."""
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings

SentenceTransformer(settings.embed_model)
CrossEncoder(settings.rerank_model)
SparseTextEmbedding("Qdrant/bm25")
print("models cached")
