"""Thin LLM layer: swap providers by changing this file only."""
from collections.abc import Iterator
from functools import lru_cache

from anthropic import Anthropic

from app.config import settings


@lru_cache
def _client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)


def generate(system: str, user: str) -> str:
    resp = _client().messages.create(
        model=settings.llm_model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def stream(system: str, user: str) -> Iterator[str]:
    with _client().messages.stream(
        model=settings.llm_model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as s:
        yield from s.text_stream
