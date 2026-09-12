"""Batch extraction: same schema/prompt/tool as extract.py, but submits all
chunks through the Message Batches API (50% cheaper, async) with prompt
caching on the shared system prompt. Recommended entry point for a full
ingestion run; extract.py's synchronous extract_chunk() remains available
for single-chunk debugging.
"""
import time

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from langfuse import Langfuse
from pydantic import ValidationError

from config import CLAUDE_MODEL, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
from ingestion.chunking import Chunk
from ingestion.extract import CACHE_DIR, EXTRACTION_TOOL, SYSTEM_PROMPT, _cache_path, client
from ingestion.schema import ExtractionResult

langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

# Batch API pricing is 50% of standard rates.
INPUT_COST_PER_MTOK = 1.00
OUTPUT_COST_PER_MTOK = 5.00
CACHE_WRITE_COST_PER_MTOK = 1.25  # 50% of the usual 1.25x-of-input cache-write rate
CACHE_READ_COST_PER_MTOK = 0.10  # 50% of the usual 0.1x-of-input cache-read rate


def _custom_id(chunk: Chunk) -> str:
    return chunk.chunk_id.replace("::", "__")


def _build_request(chunk: Chunk) -> Request:
    return Request(
        custom_id=_custom_id(chunk),
        params=MessageCreateParamsNonStreaming(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
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
        ),
    )


def run_batch_extraction(chunks: list[Chunk], poll_interval: int = 20) -> list[tuple[Chunk, ExtractionResult]]:
    """Extract relationships for all chunks, using cache where available and
    submitting the rest as one Anthropic Message Batch."""
    by_custom_id = {_custom_id(c): c for c in chunks}
    results: dict[str, ExtractionResult] = {}

    to_submit: list[Chunk] = []
    for chunk in chunks:
        cache_file = _cache_path(chunk)
        if cache_file.exists():
            results[_custom_id(chunk)] = ExtractionResult.model_validate_json(
                cache_file.read_text(encoding="utf-8")
            )
        else:
            to_submit.append(chunk)

    print(f"{len(results)} chunks already cached, {len(to_submit)} to submit as a batch.")

    totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "errors": 0}

    if to_submit:
        batch = client.messages.batches.create(requests=[_build_request(c) for c in to_submit])
        print(f"Batch {batch.id} submitted with {len(to_submit)} requests.")

        while True:
            batch = client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            print(
                f"  status={batch.processing_status} "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored}"
            )
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for item in client.messages.batches.results(batch.id):
            chunk = by_custom_id[item.custom_id]
            trace = langfuse.trace(
                name="ingestion-extraction-batch",
                input=chunk.text,
                metadata={"chunk_id": chunk.chunk_id, "title": chunk.title, "tradition": chunk.tradition},
            )

            if item.result.type != "succeeded":
                totals["errors"] += 1
                trace.update(output=None, metadata={"error": f"batch result: {item.result.type}"})
                print(f"[FAIL] {chunk.chunk_id}: batch result {item.result.type}")
                results[item.custom_id] = ExtractionResult(relations=[])
                continue

            msg = item.result.message
            usage = msg.usage
            totals["input"] += usage.input_tokens
            totals["output"] += usage.output_tokens
            totals["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
            totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0

            tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
            trace.generation(
                name="extract_relationships",
                model=CLAUDE_MODEL,
                input=chunk.text,
                output=tool_block.input if tool_block else None,
                usage={"input": usage.input_tokens, "output": usage.output_tokens, "unit": "TOKENS"},
            )

            try:
                parsed = ExtractionResult.model_validate(tool_block.input) if tool_block else ExtractionResult(relations=[])
            except ValidationError as e:
                totals["errors"] += 1
                print(f"[FAIL] {chunk.chunk_id}: schema validation failed: {e}")
                parsed = ExtractionResult(relations=[])

            results[item.custom_id] = parsed
            _cache_path(chunk).write_text(parsed.model_dump_json(), encoding="utf-8")
            trace.update(output=parsed.model_dump())

        langfuse.flush()

    cost = (
        totals["input"] / 1_000_000 * INPUT_COST_PER_MTOK
        + totals["output"] / 1_000_000 * OUTPUT_COST_PER_MTOK
        + totals["cache_write"] / 1_000_000 * CACHE_WRITE_COST_PER_MTOK
        + totals["cache_read"] / 1_000_000 * CACHE_READ_COST_PER_MTOK
    )
    print(
        f"\nBatch extraction: {len(to_submit)} requests, {totals['errors']} errors, "
        f"{totals['input']} input tokens, {totals['output']} output tokens, "
        f"{totals['cache_write']} cache-write tokens, {totals['cache_read']} cache-read tokens, "
        f"~${cost:.4f} estimated cost (batch pricing)"
    )

    return [(c, results[_custom_id(c)]) for c in chunks]
