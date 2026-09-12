# MythCheck

A hallucination-detection pipeline: it takes AI-generated text, breaks it into discrete
factual claims, and checks each one against a trusted, curated knowledge base — a hybrid
Neo4j graph + Chroma vector store built from Wikipedia mythology articles (Greek, Norse,
Egyptian) — returning a verdict (**SUPPORTED** / **CONTRADICTED** / **UNVERIFIABLE**) with
a citation for every claim.

## Benchmark

Two eval sets, scored the same way (precision/recall on catching **CONTRADICTED** claims,
plus false-positive rate on genuine claims):

**1. Synthetic corrupted-variant set (n=50)** — claims rendered from real graph edges
(gold=SUPPORTED) and from three programmatic corruption strategies applied to those same
edges (gold=CONTRADICTED): swapping the attributed entity, inverting an asymmetric
relationship, and swapping in a mutually-exclusive relation type; plus a descriptive
tradition-swap corruption.

| | n | precision | recall |
|---|---|---|---|
| **Overall** | 50 | 1.00 | 0.44 |
| relational claims | 30 | 1.00 | 0.60 |
| descriptive claims | 20 | 1.00 | 0.20 |
| — attribution_swap | 5 | n/a | 0.00 |
| — inverted_relationship | 5 | 1.00 | 0.80 |
| — wrong_relation_type | 5 | 1.00 | 1.00 |
| — tradition_swap (descriptive) | 10 | 1.00 | 0.20 |

False-positive rate on genuine claims: **0.00** (n=25 genuine claims, none ever marked
CONTRADICTED).

**2. Real-hallucination set (n=15)** — Claude (Haiku 4.5) was asked to write facts about
entities in the corpus with no grounding provided, producing a mix of accurate and subtly
wrong statements. Gold labels were assigned by checking each sentence directly against the
pulled Wikipedia source text in `data/raw/` (see `check_note` on each item in
`eval/real_hallucination_set.json`) — this was done by Claude Code cross-referencing the
source articles, not an independent human annotator, which is worth being explicit about.

| | n | precision | recall | FP rate |
|---|---|---|---|---|
| Real hallucinations | 15 | 1.00 | 0.67 | 0.00 |

### What this shows

- **Precision is 1.00 everywhere it could be measured.** Across both eval sets, the
  pipeline never once marked a genuinely true claim as CONTRADICTED. Every miss failed
  safely into UNVERIFIABLE rather than dangerously validating a false claim as SUPPORTED.
- **Relational claims (graph traversal) catch corruption more reliably than descriptive
  claims (vector retrieval + LLM judge).** 0.60 vs. 0.20 recall. Graph traversal is a
  closed, deterministic check; vector retrieval depends on whether a passage happens to
  state something specific enough to contradict.
- **`attribution_swap` (0.00 recall) is an honest, expected gap, not a bug.** Swapping in a
  different real entity as the target (e.g. "Zeus is the parent of Hadad" — Hadad is a real
  node, just not Zeus's child) produces no edge and no contradicting edge between those two
  specific entities, so the system correctly can't distinguish "false" from "simply not
  stated" without a closed-world assumption. `wrong_relation_type` (1.00) and
  `inverted_relationship` (0.80), by contrast, leave a structural signature the verdict
  logic is built to catch (a same-pair contradicting edge, or the same relation stated in
  reverse).
- **The graph's own extraction already captures the highest-confidence cross-tradition
  equivalences directly from source text** (Zeus↔Jupiter, Athena↔Minerva,
  Heracles↔Hercules, Typhon↔Set, all at confidence 1.0) — these came from Wikipedia's
  *interpretatio graeca/romana* passages during ingestion, not from the verification
  pipeline. A separate, heuristic entity-resolution step (embedding similarity + mutual
  nearest-neighbor) adds a handful of additional candidate equivalences, but at this corpus
  scale its precision is materially lower — it's tagged with a distinguishing `source`
  property (`extraction` vs. `resolution`) in the graph so the two confidence tiers stay
  separable. See `ingestion/resolve.py` for the specific failure modes this was tuned
  against.

## A bug the eval sets didn't catch

Live-testing the API surfaced a real correctness bug the synthetic eval missed: claim
extraction was assigning source/target by grammatical word order rather than semantic role,
so "Thor is the son of Odin" (inverse-phrased) was extracted as `source=Thor,
relation=parent_of, target=Odin` — backwards — and came back CONTRADICTED even though it's
true. The synthetic eval's templates only ever use direct/active phrasing ("Odin is the
parent of Thor"), so it never exercised this path. Fixed in `verification/claim_extract.py`
by making the source/target-by-semantic-role distinction explicit in the prompt, with worked
examples for both phrasings. Worth calling out as exactly the kind of gap a benchmark can
look clean while still missing — the eval measures what it was built to measure, not
everything that matters.

## Scope

This is scoped to claim-level grounding against a corpus I control (15 Wikipedia
articles across Greek, Norse, and Egyptian mythology). It does not attempt to solve
"information pollution" at ecosystem scale — source credibility, coordinated
misinformation, or provenance beyond a single trusted knowledge base. That's a narrower,
more defensible claim than a general-purpose fact-checker, and it's the honest one.

## Architecture

```
Wikipedia articles (Greek/Norse/Egyptian)
  -> LLM extraction (Claude Sonnet 5, strict tool schema) -> entity resolution
     (alias merging + cross-tradition equivalence, both via embedding similarity)
  -> Neo4j graph (idempotent MERGE writes) + Chroma vector store, shared chunk ids
  -> [AI-generated text to verify]
  -> claim decomposition (Claude Haiku 4.5) -> entity resolution against the graph
  -> routing: relational claims -> graph traversal | descriptive claims -> vector search
  -> verdict (SUPPORTED / CONTRADICTED / UNVERIFIABLE) with a citation that resolves to
     an actually-retrieved chunk
  -> FastAPI /verify endpoint + Streamlit UI
  -> Langfuse traces every stage (extraction -> resolution -> routing -> verdict)
```

**Cut on purpose:** no LangChain agent framework (this is a fixed sequence, nothing for
an agent to orchestrate); one vector store only (Chroma — the shared chunk id with Neo4j
is the actual trick, a second vector DB adds nothing); no attempt at ecosystem-scale
misinformation detection (see Scope above).

## Project structure

```
ingestion/       Wikipedia pull, chunking, ontology, LLM extraction, entity resolution,
                  Neo4j + Chroma writers
verification/    claim decomposition, entity resolution against the graph, grounding,
                  verdicts, end-to-end pipeline
api/             FastAPI backend (POST /verify)
ui/              Streamlit demo UI
eval/            synthetic + real-hallucination eval sets and scoring
scripts/         Phase 0 connectivity check
data/raw/        pulled Wikipedia article text (trimmed to signal-dense sections)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env   # fill in Neo4j AuraDB, Anthropic, and Langfuse credentials
python scripts/check_setup.py   # confirms all three services are reachable
```

## Running the pipeline

```bash
python -m ingestion.pull_wikipedia      # pull the 15 seed articles
python -m ingestion.run_ingest          # extract, resolve, write to Neo4j (batch API + caching)
python -m ingestion.vector_store        # embed chunks into Chroma
python -m eval.build_eval_set           # build the synthetic corrupted-variant eval set
python -m eval.run_eval                 # score it
python -m eval.run_real_hallucination_eval  # score the real-hallucination set

uvicorn api.main:app --reload           # backend, in one terminal
streamlit run ui/app.py                 # UI, in another
```

## Inspecting the graph

A few Cypher queries to paste into Neo4j Browser:

```cypher
// 2-hop traversal: Zeus's children and who they defeated
MATCH (z {canonical_name:'Zeus'})-[:parent_of]->(child)-[:defeats]->(x)
RETURN z.canonical_name, child.canonical_name, x.canonical_name

// alias collapsing: entities merged from multiple name variants
MATCH (n) WHERE size(n.aliases) > 1 RETURN labels(n)[0], n.canonical_name, n.aliases

// cross-tradition equivalences extracted directly from source text (high confidence)
MATCH (a)-[r:equivalent_to {source:'extraction'}]->(b)
RETURN a.canonical_name, b.canonical_name, r.confidence ORDER BY r.confidence DESC

// cross-tradition equivalences from the heuristic resolution step (lower confidence,
// worth spot-checking)
MATCH (a)-[r:equivalent_to {source:'resolution'}]->(b)
RETURN a.canonical_name, b.canonical_name, r.confidence ORDER BY r.confidence DESC
```

## Definition of done

- [x] FastAPI endpoint taking arbitrary text, returning claim-by-claim verdicts with
      validated citations (citations are rejected-and-regenerated if they don't resolve
      to an actually-retrieved chunk).
- [x] Benchmark table (precision/recall by claim type and corruption type) above.
- [x] Honestly-scoped problem statement above.
- [ ] Live demo screen recording (pending — API usage limit hit during this build; see
      status notes).
