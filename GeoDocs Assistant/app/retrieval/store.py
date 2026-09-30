"""Embeddings + Qdrant: dense, hybrid (dense + BM25 via RRF), and reranked search."""
import uuid
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.chunker import Chunk
from app.retrieval.reranker import rerank


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


@lru_cache
def _sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding("Qdrant/bm25")


@lru_cache
def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


def _sv(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def ensure_collection() -> None:
    client = _client()
    if not client.collection_exists(settings.collection):
        client.create_collection(
            settings.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=_model().get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    ensure_collection()
    texts = [c.text for c in chunks]
    dense = embed(texts)
    sparse = list(_sparse().embed(texts))
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}:{c.page}:{i}")),
            vector={"dense": d, "bm25": _sv(s)},
            payload={"text": c.text, "source": c.source, "page": c.page, "heading": c.heading},
        )
        for i, (c, d, s) in enumerate(zip(chunks, dense, sparse))
    ]
    _client().upsert(settings.collection, points=points)
    return len(points)


def _hits(res) -> list[dict]:
    return [{"score": p.score, **p.payload} for p in res.points]


def search(query: str, k: int = 5, mode: str | None = None) -> list[dict]:
    mode = mode or settings.default_mode
    client, col = _client(), settings.collection
    dense_q = embed([query])[0]

    if mode == "dense":
        return _hits(client.query_points(col, query=dense_q, using="dense", limit=k, with_payload=True))

    n = settings.candidates if mode == "hybrid_rerank" else k
    sparse_q = _sv(next(iter(_sparse().query_embed(query))))
    res = client.query_points(
        col,
        prefetch=[
            models.Prefetch(query=dense_q, using="dense", limit=settings.candidates),
            models.Prefetch(query=sparse_q, using="bm25", limit=settings.candidates),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=n,
        with_payload=True,
    )
    hits = _hits(res)
    return rerank(query, hits, k) if mode == "hybrid_rerank" else hits


def threshold_for(mode: str) -> float | None:
    """Score cutoff for refusing. Hybrid-only uses RRF scores, which aren't comparable."""
    if mode == "dense":
        return settings.min_score_dense
    if mode == "hybrid_rerank":
        return settings.min_score_rerank
    return None
