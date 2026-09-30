"""Extract place names from an answer (LLM) and geocode them (Nominatim / OpenStreetMap)."""
import json
import re
import time

import httpx

from app.config import settings
from app.llm import provider

MAX_PLACES = 8

SYSTEM = (
    "Extract the geographic place names explicitly mentioned in the text "
    "(cities, regions, countries, rivers, lakes, coastlines). "
    "Reply with ONLY a JSON array of strings, at most 8, using the most specific name given. "
    "If there are none, reply with []."
)

_cache: dict[str, tuple[float, float] | None] = {}
_last_call = 0.0


def extract_places(text: str) -> list[str]:
    raw = provider.generate(SYSTEM, text)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    names = [n.strip() for n in data if isinstance(n, str) and n.strip()]
    return list(dict.fromkeys(names))[:MAX_PLACES]


def geocode(name: str) -> tuple[float, float] | None:
    """Nominatim allows ~1 request/second and requires an identifying User-Agent."""
    global _last_call
    if name in _cache:
        return _cache[name]
    wait = 1.1 - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    try:
        r = httpx.get(
            settings.geocoder_url,
            params={"q": name, "format": "json", "limit": 1},
            headers={"User-Agent": settings.geocoder_user_agent},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        result = (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None  # transient failure: don't cache
    _cache[name] = result
    return result


def locate(text: str) -> list[dict]:
    found = []
    for name in extract_places(text):
        coords = geocode(name)
        if coords:
            found.append({"name": name, "lat": coords[0], "lon": coords[1]})
    return found
