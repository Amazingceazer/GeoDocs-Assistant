from fastapi.testclient import TestClient

from app.geo import places as pl
from app.main import app

client = TestClient(app)


def test_extract_parses_json_and_dedupes(monkeypatch):
    monkeypatch.setattr(pl.provider, "generate", lambda s, u: 'Sure: ["Lagos", "Lagos", " Niger River "]')
    assert pl.extract_places("text") == ["Lagos", "Niger River"]


def test_extract_bad_output_returns_empty(monkeypatch):
    monkeypatch.setattr(pl.provider, "generate", lambda s, u: "no places here")
    assert pl.extract_places("text") == []


def test_places_endpoint_skips_unlocatable(monkeypatch):
    monkeypatch.setattr(pl, "extract_places", lambda t: ["Lagos", "Nowhereville"])
    monkeypatch.setattr(pl, "geocode", lambda n: (6.5, 3.4) if n == "Lagos" else None)
    r = client.post("/places", json={"text": "Flooding hit Lagos."})
    assert r.json()["places"] == [{"name": "Lagos", "lat": 6.5, "lon": 3.4}]


def test_places_rejects_oversized_text():
    assert client.post("/places", json={"text": "x" * 5000}).status_code == 422


def test_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "GeoDocs" in r.text
