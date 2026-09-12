"""LLM extraction pass: chunk text -> validated relationship triples.

Strict tool use (Anthropic `strict: true`) constrains the model's output to
match the ontology's entity/relationship enums structurally; every response
is still re-validated with pydantic before being trusted, and every attempt
is traced through Langfuse. Results are cached on disk keyed by chunk id
(which itself embeds a content hash) so re-running ingestion doesn't
re-spend API budget on unchanged chunks.
"""
import json
import pathlib

from anthropic import Anthropic
from langfuse import Langfuse
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, EXTRACTION_CACHE_DIR, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
from ingestion.chunking import Chunk
from ingestion.ontology import ENTITY_TYPES, RELATIONSHIP_TYPES
from ingestion.schema import ExtractionResult

client = Anthropic(api_key=ANTHROPIC_API_KEY)
langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

CACHE_DIR = pathlib.Path(EXTRACTION_CACHE_DIR)

RELATION_DESCRIPTIONS = "\n".join(
    f"- {name} ({', '.join(src)} -> {', '.join(tgt)}): {desc}"
    for name, src, tgt, desc in RELATIONSHIP_TYPES
)
RELATIONSHIP_NAMES = [r[0] for r in RELATIONSHIP_TYPES]

SYSTEM_PROMPT = f"""You extract factual relationships between mythological entities from a \
Wikipedia excerpt, for a knowledge graph about mythology across cultures.

Valid entity types: {", ".join(ENTITY_TYPES)}

Valid relationship types (source -> target):
{RELATION_DESCRIPTIONS}

Rules:
- Only extract relationships that are explicitly stated or very strongly implied by the text.
- Use canonical proper names (e.g. "Zeus", not "he" or "the god"). Resolve pronouns to the \
named entity they refer to using context from the passage.
- confidence should reflect how directly the text states the relationship: 1.0 for an explicit \
statement, lower for something inferred or hedged in the source text.
- If the passage contains no clear relationships from the ontology, return an empty list.
- Do not invent relationships not supported by the passage."""

EXTRACTION_TOOL = {
    "name": "extract_relationships",
    "description": "Record the mythological relationships found in the passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_name": {"type": "string"},
                        "source_type": {"type": "string", "enum": ENTITY_TYPES},
                        "relation": {"type": "string", "enum": RELATIONSHIP_NAMES},
                        "target_name": {"type": "string"},
                        "target_type": {"type": "string", "enum": ENTITY_TYPES},
                        "confidence": {
                            "type": "number",
                            "description": "0.0 to 1.0",
                        },
                    },
                    "required": [
                        "source_name",
                        "source_type",
                        "relation",
                        "target_name",
                        "target_type",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["relations"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Running cost tally for the process (approximate, based on published per-token rates).
INPUT_COST_PER_MTOK = 2.00
OUTPUT_COST_PER_MTOK = 10.00
_run_totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0}


def _cache_path(chunk: Chunk) -> pathlib.Path:
    return CACHE_DIR / f"{chunk.chunk_id.replace('::', '__')}.json"


class ExtractionValidationError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ExtractionValidationError),
)
def _call_claude(chunk: Chunk, trace) -> ExtractionResult:
    generation = trace.generation(
        name="extract_relationships",
        model=CLAUDE_MODEL,
        input=chunk.text,
        metadata={"chunk_id": chunk.chunk_id, "title": chunk.title, "tradition": chunk.tradition},
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_relationships"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Entity in focus: {chunk.title} (tradition: {chunk.tradition})\n\n"
                    f"Passage:\n{chunk.text}"
                ),
            }
        ],
    )

    _run_totals["input_tokens"] += response.usage.input_tokens
    _run_totals["output_tokens"] += response.usage.output_tokens
    _run_totals["calls"] += 1

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    generation.end(output=tool_block.input if tool_block else None, usage={
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
        "unit": "TOKENS",
    })

    if tool_block is None:
        raise ExtractionValidationError(f"No tool_use block in response for {chunk.chunk_id}")

    try:
        return ExtractionResult.model_validate(tool_block.input)
    except ValidationError as e:
        raise ExtractionValidationError(f"Schema validation failed for {chunk.chunk_id}: {e}")


def extract_chunk(chunk: Chunk) -> ExtractionResult:
    """Extract relationships from one chunk, using the on-disk cache when available."""
    cache_file = _cache_path(chunk)
    if cache_file.exists():
        _run_totals["cache_hits"] += 1
        return ExtractionResult.model_validate_json(cache_file.read_text(encoding="utf-8"))

    trace = langfuse.trace(
        name="ingestion-extraction",
        input=chunk.text,
        metadata={"chunk_id": chunk.chunk_id, "title": chunk.title, "tradition": chunk.tradition},
    )
    try:
        result = _call_claude(chunk, trace)
    except ExtractionValidationError as e:
        trace.update(output=None, metadata={"error": str(e)})
        langfuse.flush()
        print(f"[FAIL] {chunk.chunk_id}: {e}")
        return ExtractionResult(relations=[])

    trace.update(output=result.model_dump())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result.model_dump_json(), encoding="utf-8")
    return result


def print_cost_summary() -> None:
    t = _run_totals
    cost = (
        t["input_tokens"] / 1_000_000 * INPUT_COST_PER_MTOK
        + t["output_tokens"] / 1_000_000 * OUTPUT_COST_PER_MTOK
    )
    print(
        f"\nExtraction run: {t['calls']} API calls, {t['cache_hits']} cache hits, "
        f"{t['input_tokens']} input tokens, {t['output_tokens']} output tokens, "
        f"~${cost:.4f} estimated cost"
    )
