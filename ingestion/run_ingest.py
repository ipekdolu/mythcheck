"""End-to-end Phase 1 ingestion: chunk -> extract -> resolve -> write to Neo4j.

Run with: python -m ingestion.run_ingest
(Run `python -m ingestion.pull_wikipedia` first if data/raw/ is empty.)
"""
from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from ingestion.chunking import chunk_all
from ingestion.extract_batch import run_batch_extraction
from ingestion.graph_writer import ensure_constraints, write_entities, write_equivalences, write_relations
from ingestion.ontology import ENTITY_TYPES
from ingestion.resolve import resolve


def main() -> None:
    chunks = chunk_all()
    print(f"Loaded {len(chunks)} chunks. Running batch extraction...")

    extractions = run_batch_extraction(chunks)

    total_relations = sum(len(r.relations) for _, r in extractions)
    print(f"\n{total_relations} raw relation triples extracted. Resolving entities...")

    result = resolve(extractions)
    entities = result["entities"]
    relations = result["relations"]
    equivalences = result["equivalences"]

    print(f"{len(entities)} resolved entities, {len(relations)} relation edges,")
    print(f"{len(equivalences)} cross-tradition equivalences found:")
    for key_a, key_b, score in equivalences:
        print(f"  {entities[key_a].canonical_name} <-> {entities[key_b].canonical_name} (score={score:.3f})")

    merged_aliases = [e for e in entities.values() if len(e.aliases) > 1]
    if merged_aliases:
        print(f"\n{len(merged_aliases)} entities with merged alias variants:")
        for e in merged_aliases:
            print(f"  {e.canonical_name}: {sorted(e.aliases)}")

    print("\nWriting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        ensure_constraints(driver, ENTITY_TYPES)
        write_entities(driver, entities)
        write_relations(driver, entities, relations)
        write_equivalences(driver, entities, equivalences)
    finally:
        driver.close()

    print("Done.")


if __name__ == "__main__":
    main()
