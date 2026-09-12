"""Phase 2: embed the same chunks used for graph extraction into Chroma,
keyed by the identical chunk_id used as the `chunk_id` property on graph
edges - this shared id is what lets a verdict later join a vector hit back
to the graph relationship(s) it came from.

Run with: python -m ingestion.vector_store
"""
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR
from ingestion.chunking import chunk_all

COLLECTION_NAME = "mythology_chunks"


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_vector_store() -> None:
    chunks = chunk_all()
    collection = get_collection()

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    print(f"{len(chunks)} chunks total, {len(existing_ids)} already embedded, {len(new_chunks)} to add.")

    if not new_chunks:
        return

    batch_size = 100
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {"title": c.title, "tradition": c.tradition, "doc_slug": c.doc_slug} for c in batch
            ],
        )
        print(f"  embedded {min(i + batch_size, len(new_chunks))}/{len(new_chunks)}")

    print(f"Done. Collection now has {collection.count()} chunks.")


if __name__ == "__main__":
    build_vector_store()
