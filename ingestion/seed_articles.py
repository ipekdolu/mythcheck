"""Seed list of Wikipedia articles for the initial ingestion batch.

Chosen to maximize real graph structure: shared parentage (Zeus -> Heracles,
Perseus, Athena; Odin -> Thor, Baldr), rivalries (Set vs Osiris, Loki vs Thor),
and plausible cross-tradition equivalences the resolution step should surface
(Zeus/Thor as sky-thunder deities, Hades/Osiris as underworld rulers).
"""

SEED_ARTICLES = [
    # (wikipedia_title, tradition)
    ("Zeus", "Greek"),
    ("Hades", "Greek"),
    ("Heracles", "Greek"),
    ("Perseus", "Greek"),
    ("Athena", "Greek"),
    ("Odin", "Norse"),
    ("Thor", "Norse"),
    ("Loki", "Norse"),
    ("Freyja", "Norse"),
    ("Baldr", "Norse"),
    ("Ra", "Egyptian"),
    ("Osiris", "Egyptian"),
    ("Isis", "Egyptian"),
    ("Anubis", "Egyptian"),
    ("Set (deity)", "Egyptian"),
]
