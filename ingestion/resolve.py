"""Entity resolution: normalize extracted names, merge alias variants via
embedding similarity, and flag cross-tradition equivalences as a separate
deliberate step.

Alias merging (ALIAS_THRESHOLD=0.80 name-embedding cosine + a lexical
plausibility gate) and cross-tradition equivalence (EQUIVALENCE_THRESHOLD=0.55
role-profile cosine + mutual-nearest-neighbor, see
find_cross_tradition_equivalences) were both tuned empirically against known
positive/negative pairs and then re-validated against the noise the full
264-chunk corpus actually produced - see the docstrings on
_lexically_plausible_alias and find_cross_tradition_equivalences for the
specific failure cases each guard exists to catch.
"""
import difflib
import re
from dataclasses import dataclass, field

from ingestion.embeddings import cosine_similarity, embed
from ingestion.ontology import SYMMETRIC_RELATIONSHIPS

ALIAS_THRESHOLD = 0.80
# Name embeddings alone are unreliable for short, diacritic-heavy proper nouns:
# unrelated Old Norse names (e.g. "Oor" vs "Njoror") can score 0.78-0.87 on
# embedding cosine alone -- the same range as genuine epithet variants like
# "Baldr"/"Balder" (0.96). A lexical-similarity gate (substring containment,
# for phrase-epithets like "Athena"/"Pallas Athena"; or a high character-level
# ratio, for spelling variants like "Baldr"/"Balder") cleanly separates the two
# on calibration data and is required in addition to the embedding threshold.
ALIAS_LEXICAL_RATIO_THRESHOLD = 0.85
EQUIVALENCE_THRESHOLD = 0.65
# Known limitation: even with mutual-NN, this heuristic step's precision on
# real (non-cherry-picked) data is mixed - profile-text embeddings for minor,
# thinly-documented entities pick up structural/generic similarity ("X
# sibling_of Y; X wields Z") more than genuine thematic correspondence, so
# some accepted pairs won't hold up to a knowledgeable reader. The pipeline's
# most reliable equivalences come from direct extraction (Wikipedia's
# interpretatio graeca/romana passages, e.g. "Odin equivalent_to Mercury" at
# confidence 0.95) rather than this step - both are written to the graph but
# tagged with a distinguishing `source` property ('extraction' vs
# 'resolution') so the two confidence tiers stay distinguishable.
EQUIVALENCE_ELIGIBLE_TYPES = {"Deity", "Hero", "Creature"}
MIN_RELATION_CONFIDENCE = 0.3


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _lexically_plausible_alias(name_a: str, name_b: str) -> bool:
    a, b = name_a.lower(), name_b.lower()
    # Whole-word containment only (e.g. "Zeus" is a word in "Zeus Eubouleus") -
    # a raw substring check would also match "jorod" inside "njorodr", which
    # are two different Norse deities, not an epithet pair.
    if a in b.split() or b in a.split():
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ALIAS_LEXICAL_RATIO_THRESHOLD


@dataclass
class ResolvedEntity:
    key: str  # "{entity_type}:{slug}" - stable id used as the graph MERGE key
    canonical_name: str
    entity_type: str
    traditions: set = field(default_factory=set)
    aliases: set = field(default_factory=set)
    source_chunk_ids: set = field(default_factory=set)
    relation_mentions: list = field(default_factory=list)  # short strings for profile embedding

    def profile_text(self) -> str:
        base = f"{self.canonical_name}, from {'/'.join(sorted(self.traditions))} tradition"
        if self.relation_mentions:
            return base + ": " + "; ".join(self.relation_mentions[:8])
        return base


@dataclass
class RawMention:
    name: str
    entity_type: str
    tradition: str
    chunk_id: str


def _collect_raw_mentions(extractions: list[tuple]) -> tuple[list[RawMention], list[dict]]:
    """extractions: list of (chunk, ExtractionResult). Returns raw entity mentions
    and raw relation dicts (source_name/type, relation, target_name/type,
    confidence, chunk_id)."""
    mentions: list[RawMention] = []
    raw_relations: list[dict] = []
    for chunk, result in extractions:
        for rel in result.relations:
            mentions.append(RawMention(rel.source_name, rel.source_type, chunk.tradition, chunk.chunk_id))
            mentions.append(RawMention(rel.target_name, rel.target_type, chunk.tradition, chunk.chunk_id))
            raw_relations.append(
                {
                    "source_name": normalize_name(rel.source_name),
                    "source_type": rel.source_type,
                    "relation": rel.relation,
                    "target_name": normalize_name(rel.target_name),
                    "target_type": rel.target_type,
                    "confidence": rel.confidence,
                    "chunk_id": chunk.chunk_id,
                }
            )
    return mentions, raw_relations


def _merge_exact(mentions: list[RawMention]) -> dict[tuple[str, str], ResolvedEntity]:
    """First pass: group mentions that share the exact same (normalized name, type)."""
    registry: dict[tuple[str, str], ResolvedEntity] = {}
    for m in mentions:
        name = normalize_name(m.name)
        dedup_key = (name.lower(), m.entity_type)
        if dedup_key not in registry:
            registry[dedup_key] = ResolvedEntity(
                key=f"{m.entity_type}:{_slug(name)}",
                canonical_name=name,
                entity_type=m.entity_type,
            )
        entity = registry[dedup_key]
        entity.traditions.add(m.tradition)
        entity.aliases.add(name)
        entity.source_chunk_ids.add(m.chunk_id)
    return registry


def _merge_aliases(registry: dict[tuple[str, str], ResolvedEntity]) -> dict[tuple[str, str], ResolvedEntity]:
    """Second pass: merge distinct name spellings of the same type whose name
    embeddings are near-duplicates (epithets, alternate spellings)."""
    entities = list(registry.values())
    if len(entities) < 2:
        return registry

    by_type: dict[str, list[ResolvedEntity]] = {}
    for e in entities:
        by_type.setdefault(e.entity_type, []).append(e)

    # union-find over entity keys
    parent = {e.key: e.key for e in entities}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pass 1: the same normalized name mentioned with an inconsistent entity
    # type across chunks (e.g. "Typhon" extracted as Deity in one chunk,
    # Creature in another) is the same entity, not two. Merge on exact name
    # match regardless of type before the fuzzy within-type pass below -
    # otherwise cross-tradition equivalence resolution later sees two nodes
    # with the same display name and nothing stopping it from "matching" them
    # to each other.
    by_name: dict[str, list[ResolvedEntity]] = {}
    for e in entities:
        by_name.setdefault(e.canonical_name.lower(), []).append(e)
    for group in by_name.values():
        for e in group[1:]:
            union(group[0].key, e.key)

    for etype, group in by_type.items():
        if len(group) < 2:
            continue
        names = [e.canonical_name for e in group]
        vecs = embed(names)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cosine_similarity(vecs[i], vecs[j]) >= ALIAS_THRESHOLD and _lexically_plausible_alias(
                    group[i].canonical_name, group[j].canonical_name
                ):
                    union(group[i].key, group[j].key)

    merged: dict[str, ResolvedEntity] = {}
    for e in entities:
        root = find(e.key)
        if root not in merged:
            merged[root] = ResolvedEntity(
                key=root, canonical_name=e.canonical_name, entity_type=e.entity_type
            )
        target = merged[root]
        # Prefer the shortest alias as canonical name (usually the base form,
        # e.g. "Athena" over "Pallas Athena").
        if len(e.canonical_name) < len(target.canonical_name):
            target.canonical_name = e.canonical_name
        target.traditions |= e.traditions
        target.aliases |= e.aliases
        target.source_chunk_ids |= e.source_chunk_ids

    # rebuild dedup-key index pointing every original (name,type) to its merged entity
    key_by_name_type: dict[tuple[str, str], ResolvedEntity] = {}
    for (name_lower, etype), e in registry.items():
        root = find(e.key)
        key_by_name_type[(name_lower, etype)] = merged[root]
    return key_by_name_type


def find_cross_tradition_equivalences(
    resolved_by_key: dict[str, ResolvedEntity]
) -> list[tuple[str, str, float]]:
    """Third pass: compare role-profile embeddings across entities of the same
    type but different traditions; flag mutual-nearest-neighbor pairs above
    EQUIVALENCE_THRESHOLD.

    A fixed cosine threshold alone does not scale to hundreds of candidate
    entities: on the full corpus, thresholding at 0.55 alone produced 500+
    "equivalences," almost all noise (e.g. "Freyr <-> Alexiares"), because
    with O(n^2) pairs even a well-separated-looking threshold accumulates
    many chance false positives. Three guards bring this back under control:
    - Only entities of the *same* entity_type are compared. Without this, an
      entity mentioned with an inconsistent type across chunks could collide
      with itself under a different key (observed in practice: "Typhon" as
      Deity vs. "Typhon" as Creature scored 0.79 against each other - fixed
      upstream in _merge_aliases, but this guard stays as defense in depth).
    - A minimum-evidence bar (>=3 captured relation mentions) excludes
      entities whose profile is close to the generic "X, from Y tradition"
      fallback - those bland, near-identical strings are a major noise source.
    - **Mutual nearest neighbor**: a pair (a, b) is only accepted if b is a's
      single best-scoring cross-tradition match of the same type *and* a is
      b's best match too - not merely "some pair that clears the threshold."
      This is standard practice for cross-lingual/cross-domain entity
      matching and is what actually suppresses the O(n^2) noise; the
      threshold becomes a secondary floor once this is in place.
    """
    candidates = [
        e
        for e in resolved_by_key.values()
        if e.entity_type in EQUIVALENCE_ELIGIBLE_TYPES and len(e.relation_mentions) >= 3
    ]
    if len(candidates) < 2:
        return []

    profiles = [e.profile_text() for e in candidates]
    vecs = embed(profiles)

    # best cross-tradition, same-type match per candidate index
    best_match: dict[int, tuple[int, float]] = {}
    for i in range(len(candidates)):
        best_j, best_score = None, -1.0
        for j in range(len(candidates)):
            if i == j:
                continue
            a, b = candidates[i], candidates[j]
            if a.entity_type != b.entity_type or (a.traditions & b.traditions):
                continue
            score = cosine_similarity(vecs[i], vecs[j])
            if score > best_score:
                best_j, best_score = j, score
        if best_j is not None:
            best_match[i] = (best_j, best_score)

    pairs: list[tuple[str, str, float]] = []
    seen: set[frozenset] = set()
    for i, (j, score) in best_match.items():
        if score < EQUIVALENCE_THRESHOLD:
            continue
        mutual = best_match.get(j)
        if mutual is None or mutual[0] != i:
            continue
        pair_id = frozenset((i, j))
        if pair_id in seen:
            continue
        seen.add(pair_id)
        pairs.append((candidates[i].key, candidates[j].key, score))
    return pairs


def resolve(extractions: list[tuple]) -> dict:
    """Full resolution pipeline.

    Returns: {
        "entities": dict[key -> ResolvedEntity],
        "relations": list[dict] (source_key/target_key/relation/confidence/chunk_id),
        "equivalences": list[(key1, key2, score)],
    }
    """
    mentions, raw_relations = _collect_raw_mentions(extractions)
    exact_registry = _merge_exact(mentions)
    key_by_name_type = _merge_aliases(exact_registry)

    resolved_by_key: dict[str, ResolvedEntity] = {}
    for e in key_by_name_type.values():
        resolved_by_key[e.key] = e

    def lookup(name: str, etype: str) -> ResolvedEntity:
        return key_by_name_type[(normalize_name(name).lower(), etype)]

    resolved_relations = []
    for rel in raw_relations:
        if rel["confidence"] < MIN_RELATION_CONFIDENCE:
            continue
        src = lookup(rel["source_name"], rel["source_type"])
        tgt = lookup(rel["target_name"], rel["target_type"])
        if src.key == tgt.key:
            continue  # no relation type in the ontology is reflexive
        src.relation_mentions.append(f"{rel['relation']} {tgt.canonical_name}")
        tgt.relation_mentions.append(f"{rel['relation']} (from {src.canonical_name})")
        resolved_relations.append(
            {
                "source_key": src.key,
                "target_key": tgt.key,
                "relation": rel["relation"],
                "confidence": rel["confidence"],
                "chunk_id": rel["chunk_id"],
                "symmetric": rel["relation"] in SYMMETRIC_RELATIONSHIPS,
            }
        )

    equivalences = find_cross_tradition_equivalences(resolved_by_key)

    return {
        "entities": resolved_by_key,
        "relations": resolved_relations,
        "equivalences": equivalences,
    }
