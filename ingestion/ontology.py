"""Fixed ontology for mythology knowledge-graph extraction.

Deliberately small and closed: extraction prompts constrain the model to these
exact labels so output is easy to validate and MERGE into Neo4j idempotently.
Do not add types ad hoc during extraction — revise this file, then re-run.
"""

# --- Entity types ---------------------------------------------------------

ENTITY_TYPES = [
    "Deity",       # gods and goddesses (Zeus, Odin, Isis)
    "Hero",        # mortal or semi-divine figures (Heracles, Sigurd)
    "Creature",    # monsters and mythical beasts (Medusa, Fenrir, Sphinx)
    "Artifact",    # named objects (Mjolnir, Aegis, Ankh)
    "Realm",       # otherworldly places (Asgard, the Duat, the Underworld)
    "Location",    # mortal/geographic places of mythic significance (Troy, the Nile)
    "Tradition",   # the cultural pantheon/tradition itself (Greek, Norse, Egyptian)
]

# --- Relationship types -----------------------------------------------------
# Each entry: (relation_name, valid_source_types, valid_target_types, description)
# Source/target constraints are advisory for extraction prompting and light
# validation — not hard schema enforcement in Neo4j.

RELATIONSHIP_TYPES = [
    (
        "parent_of",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        "Source is the parent of target.",
    ),
    (
        "sibling_of",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        "Source and target share a parent.",
    ),
    (
        "spouse_of",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        "Source is married/partnered to target.",
    ),
    (
        "rival_of",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        "Source and target are adversaries or in conflict.",
    ),
    (
        "ally_of",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        "Source and target cooperate or fight on the same side.",
    ),
    (
        "wields",
        ["Deity", "Hero"],
        ["Artifact"],
        "Source carries/uses target as a signature weapon or tool.",
    ),
    (
        "created",
        ["Deity", "Hero"],
        ["Artifact", "Creature"],
        "Source forged, crafted, or brought target into being.",
    ),
    (
        "rules",
        ["Deity", "Hero"],
        ["Realm", "Location"],
        "Source is the ruler/patron authority over target.",
    ),
    (
        "resides_in",
        ["Deity", "Hero", "Creature"],
        ["Realm", "Location"],
        "Source's primary dwelling/domain is target.",
    ),
    (
        "guards",
        ["Creature", "Deity", "Hero"],
        ["Realm", "Location", "Artifact"],
        "Source protects or blocks access to target.",
    ),
    (
        "defeats",
        ["Deity", "Hero"],
        ["Creature", "Deity", "Hero"],
        "Source overcomes/kills/vanquishes target in a mythic episode.",
    ),
    (
        "member_of_pantheon",
        ["Deity", "Hero", "Creature"],
        ["Tradition"],
        "Source belongs to target's cultural/mythological tradition.",
    ),
    (
        "equivalent_to",
        ["Deity", "Hero", "Creature"],
        ["Deity", "Hero", "Creature"],
        (
            "Source and target are the same underlying mythic figure viewed "
            "across two different traditions (e.g. Zeus/Jupiter). This is the "
            "cross-tradition resolution relationship — always symmetric."
        ),
    ),
]

RELATIONSHIP_NAMES = [r[0] for r in RELATIONSHIP_TYPES]

# Relationships that should be written to the graph in both directions
# (undirected in meaning, even though we store one row per extraction).
SYMMETRIC_RELATIONSHIPS = {"sibling_of", "spouse_of", "rival_of", "ally_of", "equivalent_to"}
