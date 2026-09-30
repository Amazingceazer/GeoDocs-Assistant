"""Shared retrieval step used by the API and the eval script."""
from app.config import settings
from app.retrieval.store import search, threshold_for


def retrieve(question: str, k: int = 5, mode: str | None = None) -> list[dict]:
    mode = mode or settings.default_mode
    hits = search(question, k, mode)
    thr = threshold_for(mode)
    if thr is not None:
        hits = [h for h in hits if h["score"] >= thr]
    return hits
