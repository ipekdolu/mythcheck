"""Pull the seed batch of Wikipedia articles and save raw text to disk.

Run with: python -m ingestion.pull_wikipedia
"""
import pathlib

import wikipediaapi

from ingestion.seed_articles import SEED_ARTICLES

RAW_DIR = pathlib.Path("data/raw")

# Wikipedia sections that carry little mythology-relation signal and mostly
# inflate extraction cost (modern reception, bibliographies, etc.). Article
# text is truncated at the first occurrence of one of these section titles.
LOW_SIGNAL_SECTIONS = {
    "see also",
    "references",
    "notes",
    "external links",
    "bibliography",
    "further reading",
    "in popular culture",
    "in modern culture",
    "gallery",
    "sources",
}


def slugify(title: str) -> str:
    return title.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _section_text(section, depth: int = 2) -> str:
    """Recursively join a section's own text with all its subsections' text."""
    heading = f"{'=' * depth} {section.title} {'=' * depth}"
    pieces = [heading, section.text.strip()] if section.text.strip() else [heading]
    for sub in section.sections:
        pieces.append(_section_text(sub, depth + 1))
    return "\n\n".join(p for p in pieces if p)


def _trim_to_signal(page: wikipediaapi.WikipediaPage) -> str:
    """Rebuild article text from top-level sections, stopping at the first
    low-signal section (references, popular culture, etc.)."""
    parts = [page.summary.strip()]
    for section in page.sections:
        if section.title.strip().lower() in LOW_SIGNAL_SECTIONS:
            break
        parts.append(_section_text(section))
    return "\n\n".join(p for p in parts if p)


def pull_all() -> None:
    wiki = wikipediaapi.Wikipedia(user_agent="mythcheck-ingestion/0.1", language="en")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for title, tradition in SEED_ARTICLES:
        page = wiki.page(title)
        if not page.exists():
            print(f"[MISSING] {title!r} not found on Wikipedia, skipping")
            continue

        trimmed_text = _trim_to_signal(page)
        out_path = RAW_DIR / f"{slugify(title)}.txt"
        out_path.write_text(
            f"TITLE: {page.title}\nTRADITION: {tradition}\n\n{trimmed_text}",
            encoding="utf-8",
        )
        print(
            f"[OK] {page.title} ({tradition}) -> {out_path} "
            f"({len(trimmed_text)}/{len(page.text)} chars kept)"
        )


if __name__ == "__main__":
    pull_all()
