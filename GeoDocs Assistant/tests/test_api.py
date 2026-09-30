from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HIT = {"score": 0.9, "text": "Flood depth was 2m.", "source": "a.pdf", "page": 3, "heading": "Results"}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_rejects_non_pdf():
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_query_refuses_without_context(monkeypatch):
    monkeypatch.setattr("app.api.routes.retrieve", lambda q, k, m: [])

    def boom(*a, **kw):
        raise AssertionError("LLM must not be called when nothing is retrieved")

    monkeypatch.setattr("app.llm.provider.generate", boom)
    body = client.post("/query", json={"question": "anything?"}).json()
    assert body["answer"].startswith("I don't know")
    assert body["sources"] == []


def test_query_returns_answer_with_sources(monkeypatch):
    monkeypatch.setattr("app.api.routes.retrieve", lambda q, k, m: [HIT])
    monkeypatch.setattr("app.llm.provider.generate", lambda s, u: "Two metres [1].")
    body = client.post("/query", json={"question": "How deep was the flood?"}).json()
    assert body["answer"] == "Two metres [1]."
    assert body["sources"][0]["page"] == 3
