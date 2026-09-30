from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    if not hits:
        return []
    scores = _model().predict([(query, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
    return [{**h, "score": float(s)} for h, s in ranked]
