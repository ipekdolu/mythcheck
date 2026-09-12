"""Write resolved entities and relationships into Neo4j using MERGE so
re-running ingestion is idempotent."""
from neo4j import Driver

from ingestion.ontology import RELATIONSHIP_NAMES
from ingestion.resolve import ResolvedEntity


def ensure_constraints(driver: Driver, entity_types: list[str]) -> None:
    with driver.session() as session:
        for label in entity_types:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE"
            )


def write_entities(driver: Driver, entities: dict[str, ResolvedEntity]) -> None:
    with driver.session() as session:
        for entity in entities.values():
            session.run(
                f"""
                MERGE (n:{entity.entity_type} {{key: $key}})
                SET n.canonical_name = $canonical_name,
                    n.traditions = $traditions,
                    n.aliases = $aliases
                """,
                key=entity.key,
                canonical_name=entity.canonical_name,
                traditions=sorted(entity.traditions),
                aliases=sorted(entity.aliases),
            )


def write_relations(driver: Driver, entities: dict[str, ResolvedEntity], relations: list[dict]) -> None:
    with driver.session() as session:
        for rel in relations:
            rel_type = rel["relation"]
            if rel_type not in RELATIONSHIP_NAMES:
                continue  # defensive; schema validation should already guarantee this
            src = entities[rel["source_key"]]
            tgt = entities[rel["target_key"]]

            session.run(
                f"""
                MATCH (a:{src.entity_type} {{key: $src_key}})
                MATCH (b:{tgt.entity_type} {{key: $tgt_key}})
                MERGE (a)-[r:{rel_type} {{chunk_id: $chunk_id}}]->(b)
                SET r.confidence = $confidence, r.source = 'extraction'
                """,
                src_key=src.key,
                tgt_key=tgt.key,
                chunk_id=rel["chunk_id"],
                confidence=rel["confidence"],
            )
            if rel["symmetric"]:
                session.run(
                    f"""
                    MATCH (a:{src.entity_type} {{key: $src_key}})
                    MATCH (b:{tgt.entity_type} {{key: $tgt_key}})
                    MERGE (b)-[r:{rel_type} {{chunk_id: $chunk_id}}]->(a)
                    SET r.confidence = $confidence, r.source = 'extraction'
                    """,
                    src_key=src.key,
                    tgt_key=tgt.key,
                    chunk_id=rel["chunk_id"],
                    confidence=rel["confidence"],
                )


def write_equivalences(
    driver: Driver, entities: dict[str, ResolvedEntity], equivalences: list[tuple[str, str, float]]
) -> None:
    with driver.session() as session:
        for key_a, key_b, score in equivalences:
            a, b = entities[key_a], entities[key_b]
            for src, tgt in [(a, b), (b, a)]:
                session.run(
                    f"""
                    MATCH (x:{src.entity_type} {{key: $src_key}})
                    MATCH (y:{tgt.entity_type} {{key: $tgt_key}})
                    MERGE (x)-[r:equivalent_to]->(y)
                    SET r.confidence = $score, r.source = 'resolution'
                    """,
                    src_key=src.key,
                    tgt_key=tgt.key,
                    score=score,
                )
