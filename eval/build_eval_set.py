"""Phase 5: build a synthetic eval set with programmatic gold labels.

Genuine claims are rendered as natural-language sentences from real,
high-confidence graph edges (so ground truth is "this really is in the
knowledge base" - true by construction). Corrupted variants swap a date-like
detail we don't have in this domain out for three corruption types that DO
map onto this ontology: swapping the attributed target entity, inverting an
asymmetric relationship, and swapping in a mutually-exclusive relation type.
Every claim in the set carries a `gold_verdict` (SUPPORTED for genuine,
CONTRADICTED for corrupted) that was never derived from the pipeline itself,
so scoring against it is a fair test.
"""
import json
import pathlib
import random

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from verification.verdict import ASYMMETRIC_RELATIONS, MUTUALLY_EXCLUSIVE_RELATIONS

random.seed(42)

OUT_PATH = pathlib.Path("eval/eval_set.json")

RELATIONAL_TEMPLATES = {
    "parent_of": "{a} is the parent of {b}.",
    "sibling_of": "{a} is a sibling of {b}.",
    "spouse_of": "{a} is married to {b}.",
    "rival_of": "{a} is a rival of {b}.",
    "ally_of": "{a} is an ally of {b}.",
    "wields": "{a} wields {b}.",
    "created": "{a} created {b}.",
    "rules": "{a} rules {b}.",
    "resides_in": "{a} resides in {b}.",
    "guards": "{a} guards {b}.",
    "defeats": "{a} defeats {b}.",
    "member_of_pantheon": "{a} is a member of the {b} pantheon.",
    "equivalent_to": "{a} is equivalent to {b} in another tradition.",
}


def fetch_edges(driver, rel_type: str, limit: int = 60) -> list[dict]:
    with driver.session() as session:
        return session.run(
            f"""MATCH (a)-[r:{rel_type}]->(b) WHERE r.source = 'extraction' AND r.confidence >= 0.8
            RETURN a.key AS a_key, a.canonical_name AS a_name, labels(a)[0] AS a_type,
                   b.key AS b_key, b.canonical_name AS b_name, labels(b)[0] AS b_type""",
            limit=limit,
        ).data()


def fetch_entities_by_type(driver, entity_type: str) -> list[dict]:
    with driver.session() as session:
        return session.run(
            f"MATCH (n:{entity_type}) RETURN n.key AS key, n.canonical_name AS name, n.traditions AS traditions"
        ).data()


def build_relational_items(driver, n_genuine: int = 15, n_per_corruption: int = 5) -> list[dict]:
    items = []

    # Pull a varied pool across relation types (skip member_of_pantheon/equivalent_to -
    # not meaningful targets for attribution-swap/inversion corruption).
    pool_rel_types = ["parent_of", "spouse_of", "sibling_of", "defeats", "wields", "rules", "created", "resides_in"]
    pool = []
    for rt in pool_rel_types:
        pool.extend([(rt, e) for e in fetch_edges(driver, rt)])
    random.shuffle(pool)

    entities_by_type: dict[str, list[dict]] = {}

    def entities_of(etype: str) -> list[dict]:
        if etype not in entities_by_type:
            entities_by_type[etype] = fetch_entities_by_type(driver, etype)
        return entities_by_type[etype]

    # --- genuine (uncorrupted) ---
    genuine_sample = pool[:n_genuine]
    for rel_type, e in genuine_sample:
        text = RELATIONAL_TEMPLATES[rel_type].format(a=e["a_name"], b=e["b_name"])
        items.append(
            {
                "text": text,
                "claim_type": "relational",
                "corruption_type": "none",
                "gold_verdict": "SUPPORTED",
                "source_edge": {"a": e["a_name"], "relation": rel_type, "b": e["b_name"]},
            }
        )

    remaining = pool[n_genuine:]
    idx = 0

    # --- attribution_swap: replace target with a different same-type entity ---
    count = 0
    while count < n_per_corruption and idx < len(remaining):
        rel_type, e = remaining[idx]
        idx += 1
        candidates = [x for x in entities_of(e["b_type"]) if x["key"] != e["b_key"]]
        if not candidates:
            continue
        fake_target = random.choice(candidates)
        text = RELATIONAL_TEMPLATES[rel_type].format(a=e["a_name"], b=fake_target["name"])
        items.append(
            {
                "text": text,
                "claim_type": "relational",
                "corruption_type": "attribution_swap",
                "gold_verdict": "CONTRADICTED",
                "source_edge": {"a": e["a_name"], "relation": rel_type, "b": e["b_name"], "swapped_to": fake_target["name"]},
            }
        )
        count += 1

    # --- inverted_relationship: swap source/target on an asymmetric relation ---
    count = 0
    while count < n_per_corruption and idx < len(remaining):
        rel_type, e = remaining[idx]
        idx += 1
        if rel_type not in ASYMMETRIC_RELATIONS:
            continue
        text = RELATIONAL_TEMPLATES[rel_type].format(a=e["b_name"], b=e["a_name"])
        items.append(
            {
                "text": text,
                "claim_type": "relational",
                "corruption_type": "inverted_relationship",
                "gold_verdict": "CONTRADICTED",
                "source_edge": {"a": e["b_name"], "relation": rel_type, "b": e["a_name"], "true_direction": f"{e['a_name']} {rel_type} {e['b_name']}"},
            }
        )
        count += 1

    # --- wrong_relation_type: swap to a mutually-exclusive relation type ---
    rival_ally_edges = [("rival_of", e) for e in fetch_edges(driver, "rival_of")] + [
        ("ally_of", e) for e in fetch_edges(driver, "ally_of")
    ]
    random.shuffle(rival_ally_edges)
    count = 0
    for rel_type, e in rival_ally_edges:
        if count >= n_per_corruption:
            break
        swapped_type = "ally_of" if rel_type == "rival_of" else "rival_of"
        text = RELATIONAL_TEMPLATES[swapped_type].format(a=e["a_name"], b=e["b_name"])
        items.append(
            {
                "text": text,
                "claim_type": "relational",
                "corruption_type": "wrong_relation_type",
                "gold_verdict": "CONTRADICTED",
                "source_edge": {"a": e["a_name"], "relation": rel_type, "b": e["b_name"], "swapped_to": swapped_type},
            }
        )
        count += 1

    return items


def build_descriptive_items(driver, n_genuine: int = 10, n_corrupted: int = 10) -> list[dict]:
    items = []
    all_traditions = ["Greek", "Norse", "Egyptian"]

    candidates = []
    for etype in ["Deity", "Hero"]:
        for e in fetch_entities_by_type(driver, etype):
            traditions = e["traditions"] or []
            if len(traditions) == 1 and traditions[0] in all_traditions:
                candidates.append({**e, "entity_type": etype, "tradition": traditions[0]})
    random.shuffle(candidates)

    genuine_sample = candidates[:n_genuine]
    for e in genuine_sample:
        text = f"{e['name']} is a {e['entity_type'].lower()} from the {e['tradition']} tradition."
        items.append(
            {
                "text": text,
                "claim_type": "descriptive",
                "corruption_type": "none",
                "gold_verdict": "SUPPORTED",
                "source_edge": {"entity": e["name"], "true_tradition": e["tradition"]},
            }
        )

    corrupted_sample = candidates[n_genuine : n_genuine + n_corrupted]
    for e in corrupted_sample:
        false_tradition = random.choice([t for t in all_traditions if t != e["tradition"]])
        text = f"{e['name']} is a {e['entity_type'].lower()} from the {false_tradition} tradition."
        items.append(
            {
                "text": text,
                "claim_type": "descriptive",
                "corruption_type": "tradition_swap",
                "gold_verdict": "CONTRADICTED",
                "source_edge": {"entity": e["name"], "true_tradition": e["tradition"], "swapped_to": false_tradition},
            }
        )

    return items


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        items = build_relational_items(driver) + build_descriptive_items(driver)
    finally:
        driver.close()

    random.shuffle(items)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    by_type = {}
    for it in items:
        key = (it["claim_type"], it["corruption_type"])
        by_type[key] = by_type.get(key, 0) + 1
    print(f"Built {len(items)} eval items -> {OUT_PATH}")
    for key, count in sorted(by_type.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
