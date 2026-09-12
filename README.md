# MythCheck

Hallucination detection for AI-generated text, grounded in a mythology knowledge graph.

MythCheck takes a passage of text, decomposes it into atomic factual claims, and checks
each one against a trusted knowledge base — a Neo4j graph + Chroma vector store built from
Wikipedia mythology articles (Greek, Norse, Egyptian) — returning a verdict
(**SUPPORTED** / **CONTRADICTED** / **UNVERIFIABLE**) with a citation for every claim.

## Benchmark

| Eval set | n | precision | recall | false-positive rate |
|---|---|---|---|---|
| Synthetic corrupted claims | 50 | 1.00 | 0.44 | 0.00 |
| Real hallucinations (Claude-generated, manually checked) | 15 | 1.00 | 0.67 | 0.00 |

| By claim type (synthetic set) | n | precision | recall |
|---|---|---|---|
| Relational (graph traversal) | 30 | 1.00 | 0.60 |
| Descriptive (vector + LLM judge) | 20 | 1.00 | 0.20 |

Precision is 1.00 across every category measured — the system never marks a genuinely
true claim as CONTRADICTED. Recall varies by how "checkable" a corruption is: swapping in
a relation type that directly conflicts with an existing edge (`wrong_relation_type`) is
caught 100% of the time, since it leaves a clear structural signature in the graph;
swapping in an unrelated entity (`attribution_swap`) is caught 0% of the time, since
nothing distinguishes "false" from "simply not stated" without a closed-world assumption.
Full methodology and per-corruption-type breakdown: [`eval/`](eval/).

## Architecture

```
Wikipedia articles (Greek/Norse/Egyptian)
  → LLM extraction (Claude, strict tool schema) → entity resolution
    (alias merging + cross-tradition equivalence via embedding similarity)
  → Neo4j graph + Chroma vector store, shared chunk ids
  → [text to verify]
  → claim decomposition → entity resolution against the graph
  → routing: relational claims → graph traversal | descriptive claims → vector search
  → verdict with a citation that resolves to an actually-retrieved chunk
  → FastAPI /verify endpoint + Streamlit UI
  → Langfuse traces every stage end-to-end
```

**Scope:** grounded against a corpus I control (15 curated Wikipedia articles), not a
general-purpose fact-checker. No LangChain agent framework (this is a fixed pipeline, not
an autonomous agent); one vector store (Chroma) sharing chunk ids with the graph, rather
than a second database.

## Project structure

```
ingestion/       Wikipedia pull, chunking, ontology, LLM extraction, entity resolution
verification/    claim decomposition, resolution, grounding, verdicts
api/             FastAPI backend (POST /verify)
ui/              Streamlit demo UI
eval/            eval sets and scoring
scripts/         environment connectivity check
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # fill in Neo4j AuraDB, Anthropic, and Langfuse credentials
python scripts/check_setup.py
```

## Running it

```bash
python -m ingestion.pull_wikipedia
python -m ingestion.run_ingest
python -m ingestion.vector_store

uvicorn api.main:app --reload      # backend
streamlit run ui/app.py            # UI, in another terminal
```

```bash
python -m eval.build_eval_set
python -m eval.run_eval
python -m eval.run_real_hallucination_eval
```

## Inspecting the graph

```cypher
// 2-hop traversal
MATCH (z {canonical_name:'Zeus'})-[:parent_of]->(child)-[:defeats]->(x)
RETURN z.canonical_name, child.canonical_name, x.canonical_name

// alias collapsing
MATCH (n) WHERE size(n.aliases) > 1 RETURN labels(n)[0], n.canonical_name, n.aliases

// cross-tradition equivalences extracted directly from source text
MATCH (a)-[r:equivalent_to {source:'extraction'}]->(b)
RETURN a.canonical_name, b.canonical_name, r.confidence ORDER BY r.confidence DESC
```
