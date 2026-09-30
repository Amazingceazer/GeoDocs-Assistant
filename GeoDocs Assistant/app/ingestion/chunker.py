"""PDF -> section-aware chunks with page numbers and headings."""
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import fitz  # PyMuPDF

from app.config import settings


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    heading: str


def _lines(page):
    """Yield (text, font_size, is_bold) for each line on a page."""
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = " ".join(span["text"].strip() for span in spans)
            size = max(span["size"] for span in spans)
            bold = all("bold" in span["font"].lower() for span in spans)
            yield text, size, bold


def _split(text: str, size: int, overlap: int):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    start = 0
    while start < len(text):
        yield text[start : start + size]
        if start + size >= len(text):
            break
        start += size - overlap


def chunk_pdf(path: str, source: str | None = None) -> list[Chunk]:
    source = source or Path(path).name
    chunks: list[Chunk] = []
    heading, buf, buf_page = "Introduction", [], 1

    with fitz.open(path) as doc:
        sizes = [size for page in doc for _, size, _ in _lines(page)]
        body = median(sizes) if sizes else 10

        def flush():
            text = " ".join(buf).strip()
            if text:
                for piece in _split(text, settings.chunk_chars, settings.chunk_overlap):
                    chunks.append(Chunk(piece, source, buf_page, heading))
            buf.clear()

        for page in doc:
            for text, size, bold in _lines(page):
                is_heading = len(text) < 100 and (
                    size > body * 1.15 or (bold and size >= body)
                )
                if is_heading:
                    flush()
                    heading, buf_page = text, page.number + 1
                else:
                    if not buf:
                        buf_page = page.number + 1
                    buf.append(text)
        flush()
    return chunks
