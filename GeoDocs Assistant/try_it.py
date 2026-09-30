"""Quick smoke test: python try_it.py data/report.pdf "your question" """
import sys

from app.ingestion.chunker import chunk_pdf
from app.retrieval.store import search, upsert_chunks

chunks = chunk_pdf(sys.argv[1])
print(f"{len(chunks)} chunks; stored {upsert_chunks(chunks)}")
for hit in search(sys.argv[2]):
    print(f"[{hit['score']:.2f}] p.{hit['page']} {hit['heading']}: {hit['text'][:120]}...")
