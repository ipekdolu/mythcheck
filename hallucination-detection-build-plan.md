# Hallucination Detection Pipeline — Build Plan
**Shape: claim verification against a trusted knowledge graph + vector store**
**Domain: mythology across cultures (locked in)**
**UI: simple single-page Streamlit app calling the FastAPI backend (locked in)**
**Observability: Langfuse included in scope (see note below)**

## The pitch (why this project, in one paragraph)

AI-generated content sounds confident whether or not it's correct — a model will invent a
date, misattribute a relationship, or blend two similar facts into one wrong one, all in
fluent, plausible prose. This project builds a pipeline that takes AI-generated text,
breaks it into discrete factual claims, and checks each one against a trusted, curated
knowledge base — a hybrid graph + vector store — returning a verdict (supported /
contradicted / unverifiable) with a citation for every claim. It's the same extraction,
resolution, and grounding skill set as hybrid RAG, aimed at a sharper, more current problem:
telling you *which specific sentences* in an AI output you should not trust.

## Domain & dataset

- **Trusted corpus:** Wikipedia articles on mythology across cultures (Greek, Norse,
  Egyptian to start — expandable). Entities: deities, heroes, artifacts, realms; real
  resolution problem: cross-culture equivalence (Zeus/Jupiter-style "this figure in one
  tradition corresponds to that figure in another"), plus the usual name-variant mess
  within a single tradition (epithets, alternate spellings).
- **Content to verify:** AI-generated text about the same domain — e.g. ask Claude to
  write short "facts about this deity/hero" pieces. These naturally contain a mix of
  correct and subtly wrong claims (wrong parentage, misattributed artifact, invented
  rivalry) without you having to fake anything.
- **Domain is locked in.** Mythology: genuinely interesting resolution problem
  (cross-cultural equivalence is a real, hard, nameable challenge — not a manufactured
  one), and the corpus choice itself doesn't matter for the CV signal — the rigor of the
  pipeline (extraction, resolution, grounding, benchmark) carries that, not the topic.


## Architecture

```
Trusted corpus (Wikipedia mythology articles — Greek/Norse/Egyptian)
   ↓
Extraction + entity resolution (LLM extraction → strict schema → Neo4j graph,
   with alias resolution AND cross-culture equivalence resolution — the hard part)
   ↓
Shared-chunk-id hybrid store: Neo4j (graph) + Chroma (embeddings)
   ↓
AI-generated text to verify (a summary/fact-sheet the model wrote about the same domain)
   ↓
Claim extraction (decompose the text into atomic, checkable claims —
   "X is the child of Y," "X wields [artifact]," "X is equivalent to Y in [tradition]")
   ↓
Entity resolution (resolve each claim's entities to graph node ids —
   reuse the same resolution logic as ingestion)
   ↓
Grounding retrieval (graph traversal for relational claims — "parent of," "rival of";
   vector search for descriptive claims — "associated with the underworld")
   ↓
Verdict: SUPPORTED / CONTRADICTED / UNVERIFIABLE, each with a citation that must
   resolve to an actually-retrieved chunk (reject-and-regenerate if not)
   ↓
FastAPI endpoint + demo UI: paste AI-generated text → see it annotated claim-by-claim
   with verdicts and a headline "hallucination rate"
   ↓
Langfuse traces the whole chain (extraction → resolution → routing → verdict) for
   observability and debugging
```

**What's in scope vs. cut on purpose (updated):**
- **Langfuse — now in scope.** Given the flagship project's timeline is uncertain,
  observability shouldn't wait for it. This pipeline already needs to log extraction
  results, routing decisions, and verdicts for the eval work — wiring that through
  Langfuse instead of plain logging is a few hours of instrumentation on top of work
  you're doing anyway, not a redesign.
- **No LangChain agent framework — still cut, but not for timing reasons.** This
  pipeline is a fixed sequence (extract → resolve → retrieve → verify), not a system
  making autonomous next-step decisions, so there's nothing for an agent framework to
  actually orchestrate. Adding one would be unjustifiable complexity, not a hedge — if
  agentic orchestration needs proving, that belongs to a project whose shape actually
  calls for it.
- One vector store only (Chroma) — the shared chunk id with Neo4j is the actual trick;
  running a second database (pgvector) alongside it adds operational overhead with no
  new capability.
- No attempt to solve "information pollution" at ecosystem scale (source credibility,
  coordinated misinformation, provenance) — this is scoped to claim-level grounding
  against a corpus you control. State that scope honestly in the README; it's a stronger,
  more defensible claim than overreaching.

## Evaluation plan — and a trick to keep it cheap

The elegant part: you don't need to hand-label a large "is this claim true" dataset.
1. Generate a batch of AI summaries about entities in your corpus.
2. Programmatically create **corrupted variants** — swap a date, swap an attributed
   person, invert a relationship — so you have gold labels for free (you know exactly
   which claims you corrupted and how).
3. Mix corrupted and uncorrupted claims, run the pipeline, and score precision/recall
   on catching the corrupted ones plus false-positive rate on the genuine ones.
4. Layer in a small (~20-item) manually-checked set of real, naturally-occurring model
   hallucinations for a sanity check against the synthetic set — the synthetic corruption
   catches "planted" errors cleanly, but real hallucinations aren't always injected as
   cleanly as a synthetic swap, so this set confirms the pipeline generalizes.
5. Report precision/recall by claim type (relational vs. descriptive) and by corruption
   type (date swap, attribution swap, invented relationship) — this breakdown is the
   benchmark table, same role as the hop-count table in the original guide.

## UI spec (kept deliberately simple)

- **FastAPI backend:** one endpoint (`POST /verify`) taking raw text, returning a list of
  extracted claims each with verdict, confidence, and citation — this is the piece that
  actually demonstrates API design skill and is reusable/testable independent of any UI.
- **Streamlit frontend:** single page, single text box. Paste text → hit submit → see the
  text re-rendered with each claim highlighted (color-coded by verdict) and a headline
  "X% of claims supported / Y% contradicted / Z% unverifiable" score at the top. Streamlit
  calls the FastAPI endpoint over HTTP — two small services, not one tangled app.
- Deliberately no auth, no persistence, no multi-user handling — this is a demo surface,
  not a product. If deployment has friction (as with RightsDE), a short screen recording
  of the paste-and-verify flow is the fallback, same pattern as before.

## Phased build with checkpoints (full-time pace, target ~7–9 working days)

Each phase ends with a concrete checkpoint you can verify yourself before moving on —
matching your usual pattern of one completed, testable piece at a time.

**Phase 0 — Environment setup** *(~half day)*
- New repo, Python 3.11 virtual env, Neo4j AuraDB free-tier instance, Chroma installed
  locally, Langfuse account (cloud free tier) set up, `.env` for API keys, basic project
  structure (`ingestion/`, `verification/`, `api/`, `ui/`, `eval/`).
- ✅ **Checkpoint:** can connect to AuraDB from a script and run a trivial Cypher query;
  Claude API key confirmed working with a test call; a trivial Langfuse trace shows up
  in the dashboard.

**Phase 1 — Corpus ingestion & knowledge graph**
- Pull Wikipedia articles on mythology (constrained ontology: ~5–10 entity types —
  Deity, Hero, Artifact, Realm, Tradition; ~8–15 relationship types — parent_of,
  rival_of, wields, ruler_of, equivalent_to, member_of_pantheon).
- Extraction pass into Neo4j with entity resolution (normalization, embedding-similarity
  threshold, alias lists on nodes) — including the cross-tradition `equivalent_to`
  relationship as its own deliberate resolution challenge, idempotent writes via `MERGE`.
- Wire extraction calls through Langfuse so every extraction pass is traced from the start.
- ✅ **Checkpoint:** you can open Neo4j Browser, run a 2-hop query by hand, and get a
  sensible result. Known alias collisions (epithets, alternate spellings) visibly
  collapsed to one node; at least one cross-tradition equivalence correctly linked.

**Phase 2 — Vector layer**
- Embed the same chunks into Chroma, shared chunk ids with the graph.
- ✅ **Checkpoint:** a manual similarity search for a known fact returns the right chunk
  in the top few results.

**Phase 3 — Claim extraction & resolution**
- Atomic-claim decomposition (LLM call, strict schema), entity resolution of claim
  entities against the graph.
- ✅ **Checkpoint:** feed it one hand-written paragraph with 3–4 known claims; it
  correctly extracts each as a separate claim with entities resolved.

**Phase 4 — Grounding & verdicts**
- Router: relational claims → graph traversal, descriptive claims → vector search.
- Verdict logic (SUPPORTED/CONTRADICTED/UNVERIFIABLE) + citation validation +
  reject-and-regenerate on unresolved citations.
- Trace routing decisions and verdicts through Langfuse — this is the data you'll want
  for Phase 5, and you won't be able to reconstruct it after the fact.
- ✅ **Checkpoint:** the same test paragraph from Phase 3 now returns correct verdicts —
  including at least one you've deliberately made wrong, to confirm CONTRADICTED works.
  Langfuse dashboard shows the full trace for that request end-to-end.

**Phase 5 — Evaluation**
- Generate AI summaries, build the corrupted-variant set (programmatic gold labels),
  run the pipeline, score by claim type and corruption type. Add the small
  manually-checked real-hallucination set.
- ✅ **Checkpoint:** benchmark table exists with precision/recall broken out by claim
  type and corruption type — this is the artifact that goes in the README.

**Phase 6 — API + UI**
- FastAPI `/verify` endpoint. Streamlit single-page UI calling it over HTTP.
- ✅ **Checkpoint:** you can paste a real AI-generated paragraph into the running UI and
  see it annotated end-to-end, live.

**Phase 7 — Write-up & polish**
- README leading with the benchmark table and an honestly-scoped problem statement.
- Push to GitHub, short demo video/gif as a deployment fallback.
- ✅ **Checkpoint:** a stranger reading only the README top-to-bottom understands what
  it does, why it matters, and how well it works — in under 2 minutes.

## Definition of done

- A FastAPI endpoint that takes arbitrary text and returns claim-by-claim verdicts with
  validated citations.
- A benchmark table (precision/recall by claim type and corruption type) at the top of
  the README.
- A demo showing a real AI-generated passage annotated end-to-end.
- A one-paragraph, honestly-scoped explanation of what problem this solves (hallucination/
  grounding) and what it doesn't (broader information pollution).
