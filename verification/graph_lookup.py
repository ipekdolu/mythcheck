"""Load a lightweight in-memory index of graph entities for claim resolution,
so we don't round-trip to Neo4j once per claim entity."""
from dataclasses import dataclass

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from ingestion.embeddings import cosine_similarity, embed
from ingestion.ontology import ENTITY_TYPES

FUZZY_MATCH_THRESHOLD = 0.80  # same rationale/calibration as ingestion.resolve.ALIAS_THRESHOLD


@dataclass
class GraphEntity:
    key: str
    canonical_name: str
    entity_type: str
    aliases: list[str]


class EntityIndex:
    def __init__(self, entities: list[GraphEntity]):
        self.entities = entities
        self.exact_lookup: dict[tuple[str, str], GraphEntity] = {}
        for e in entities:
            for alias in [e.canonical_name, *e.aliases]:
                self.exact_lookup[(alias.lower(), e.entity_type)] = e

        self._by_type: dict[str, list[GraphEntity]] = {}
        for e in entities:
            self._by_type.setdefault(e.entity_type, []).append(e)
        self._name_vecs: dict[str, list[list[float]]] = {}
        for etype, group in self._by_type.items():
            self._name_vecs[etype] = embed([e.canonical_name for e in group])

    def resolve(self, name: str, entity_type: str) -> GraphEntity | None:
        exact = self.exact_lookup.get((name.strip().lower(), entity_type))
        if exact:
            return exact

        group = self._by_type.get(entity_type)
        if not group:
            return None
        query_vec = embed([name])[0]
        best_entity, best_score = None, -1.0
        for e, vec in zip(group, self._name_vecs[entity_type]):
            score = cosine_similarity(query_vec, vec)
            if score > best_score:
                best_entity, best_score = e, score
        if best_score >= FUZZY_MATCH_THRESHOLD:
            return best_entity
        return None

    def resolve_any_type(self, name: str) -> GraphEntity | None:
        """For descriptive claims where the entity type isn't known upfront:
        try an exact alias match across all types first, then fall back to
        the best fuzzy match across all types."""
        name_lower = name.strip().lower()
        for (alias, _etype), entity in self.exact_lookup.items():
            if alias == name_lower:
                return entity

        best_entity, best_score = None, -1.0
        for etype in self._by_type:
            candidate = self.resolve(name, etype)
            if candidate is None:
                continue
            # resolve() already applied the fuzzy threshold; among types that
            # matched, prefer the closest name-embedding score.
            score = cosine_similarity(embed([name])[0], embed([candidate.canonical_name])[0])
            if score > best_score:
                best_entity, best_score = candidate, score
        return best_entity


def load_entity_index() -> EntityIndex:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    entities: list[GraphEntity] = []
    try:
        with driver.session() as session:
            for label in ENTITY_TYPES:
                result = session.run(
                    f"MATCH (n:{label}) RETURN n.key AS key, n.canonical_name AS name, n.aliases AS aliases"
                )
                for record in result:
                    entities.append(
                        GraphEntity(
                            key=record["key"],
                            canonical_name=record["name"],
                            entity_type=label,
                            aliases=record["aliases"] or [],
                        )
                    )
    finally:
        driver.close()
    return EntityIndex(entities)
