import json
import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.geo.places import locate
from app.ingestion.chunker import chunk_pdf
from app.llm import provider
from app.llm.prompts import SYSTEM, build_user_prompt
from app.rag import retrieve
from app.retrieval.store import upsert_chunks
from app.schemas import (
    PlacesRequest,
    PlacesResponse,
    QueryRequest,
    QueryResponse,
    Source,
)

log = logging.getLogger("geodocs")
router = APIRouter()

NO_ANSWER = "I don't know based on the provided documents."


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest")
async def ingest(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
    try:
        chunks = chunk_pdf(tmp.name, source=file.filename)
        stored = upsert_chunks(chunks) if chunks else 0
    finally:
        os.unlink(tmp.name)
    log.info("ingested %s: %d chunks", file.filename, stored)
    return {"source": file.filename, "chunks": stored}


def _sources(hits: list[dict]) -> list[Source]:
    return [
        Source(id=i, source=h["source"], page=h["page"], heading=h["heading"],
               score=round(h["score"], 3), snippet=h["text"][:200])
        for i, h in enumerate(hits, 1)
    ]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    mode = req.mode or settings.default_mode
    hits = retrieve(req.question, req.k, mode)
    log.info("query=%r mode=%s kept=%d", req.question, mode, len(hits))

    if not hits:  # refuse without calling the LLM
        if req.stream:
            return StreamingResponse(iter([_sse("token", NO_ANSWER)]), media_type="text/event-stream")
        return QueryResponse(answer=NO_ANSWER, sources=[])

    prompt = build_user_prompt(req.question, hits)
    sources = _sources(hits)

    if req.stream:
        def events():
            yield _sse("sources", [s.model_dump() for s in sources])
            for tok in provider.stream(SYSTEM, prompt):
                yield _sse("token", tok)
        return StreamingResponse(events(), media_type="text/event-stream")

    return QueryResponse(answer=provider.generate(SYSTEM, prompt), sources=sources)


@router.post("/places", response_model=PlacesResponse)
def places(req: PlacesRequest):
    return PlacesResponse(places=locate(req.text))


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
