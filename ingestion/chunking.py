"""Split raw article text into paragraph-level chunks with stable ids."""
import hashlib
import pathlib
from dataclasses import dataclass

RAW_DIR = pathlib.Path("data/raw")
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1800


@dataclass
class Chunk:
    chunk_id: str
    doc_slug: str
    title: str
    tradition: str
    text: str


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def chunk_article(path: pathlib.Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    header, _, body = raw.partition("\n\n")
    title_line, tradition_line = header.split("\n")
    title = title_line.removeprefix("TITLE: ").strip()
    tradition = tradition_line.removeprefix("TRADITION: ").strip()

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > MAX_CHUNK_CHARS:
            chunks.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer:
        chunks.append(buffer)

    # Merge any trailing too-short chunk into the previous one.
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{c}"
        else:
            merged.append(c)

    slug = path.stem
    return [
        Chunk(
            chunk_id=f"{slug}::{i}::{_doc_hash(text)}",
            doc_slug=slug,
            title=title,
            tradition=tradition,
            text=text,
        )
        for i, text in enumerate(merged)
    ]


def chunk_all() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        all_chunks.extend(chunk_article(path))
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    print(f"{len(chunks)} chunks from {len(list(RAW_DIR.glob('*.txt')))} articles")
    for c in chunks[:3]:
        print(f"  {c.chunk_id} ({len(c.text)} chars)")
