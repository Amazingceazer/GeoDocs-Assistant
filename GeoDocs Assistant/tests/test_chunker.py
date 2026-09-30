import fitz

from app.ingestion.chunker import _split, chunk_pdf


def make_pdf(path: str) -> None:
    doc = fitz.open()
    for title in ("Flood Risk", "Methods"):
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=22)
        for i in range(6):
            page.insert_text((72, 120 + i * 16), f"Body sentence {i} about {title.lower()}.", fontsize=10)
    doc.save(path)


def test_chunks_keep_headings_and_pages(tmp_path):
    pdf = str(tmp_path / "sample.pdf")
    make_pdf(pdf)
    chunks = chunk_pdf(pdf)
    assert {c.heading for c in chunks} == {"Flood Risk", "Methods"}
    methods = next(c for c in chunks if c.heading == "Methods")
    assert methods.page == 2
    assert methods.source == "sample.pdf"


def test_split_overlaps_without_redundant_tail():
    text = "abcdefghijklmnopqrstuvwxy"  # 25 chars
    pieces = list(_split(text, 10, 2))
    assert len(pieces) == 3
    assert pieces[0][-2:] == pieces[1][:2]
    assert pieces[-1].endswith("y")
