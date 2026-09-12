"""End-to-end verification pipeline: text -> claims -> resolution -> routing
-> verdicts, traced through Langfuse end-to-end (extraction -> resolution ->
routing -> verdict)."""
from dataclasses import asdict

from langfuse import Langfuse
from neo4j import GraphDatabase

from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from verification.claim_extract import extract_claims
from verification.graph_lookup import EntityIndex, load_entity_index
from verification.resolve_claims import resolve_claims
from verification.verdict import Verdict, get_verdict

langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

_entity_index_cache: EntityIndex | None = None


def _get_entity_index() -> EntityIndex:
    global _entity_index_cache
    if _entity_index_cache is None:
        _entity_index_cache = load_entity_index()
    return _entity_index_cache


def verify_text(text: str) -> dict:
    trace = langfuse.trace(name="verify-text", input=text)

    extraction_span = trace.span(name="claim-extraction", input=text)
    extraction_result = extract_claims(text)
    extraction_span.end(output=[c.model_dump() for c in extraction_result.claims])

    resolution_span = trace.span(name="entity-resolution", input=[c.model_dump() for c in extraction_result.claims])
    index = _get_entity_index()
    resolved_claims = resolve_claims(extraction_result.claims, index)
    resolution_span.end(output=[c.model_dump() for c in resolved_claims])

    routing_span = trace.span(
        name="routing",
        input=[c.claim_type for c in resolved_claims],
        output={
            "relational": sum(1 for c in resolved_claims if c.claim_type == "relational"),
            "descriptive": sum(1 for c in resolved_claims if c.claim_type == "descriptive"),
        },
    )
    routing_span.end()

    verdict_span = trace.span(name="verdicts")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    verdicts: list[Verdict] = []
    try:
        for claim in resolved_claims:
            v = get_verdict(claim, driver)
            verdicts.append(v)
    finally:
        driver.close()
    verdict_span.end(output=[asdict(v) for v in verdicts])

    counts = {label: sum(1 for v in verdicts if v.verdict == label) for label in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]}
    total = len(verdicts) or 1
    summary = {
        "total_claims": len(verdicts),
        "supported_pct": round(100 * counts["SUPPORTED"] / total, 1),
        "contradicted_pct": round(100 * counts["CONTRADICTED"] / total, 1),
        "unverifiable_pct": round(100 * counts["UNVERIFIABLE"] / total, 1),
    }
    trace.update(output=summary)
    langfuse.flush()

    return {"verdicts": [asdict(v) for v in verdicts], "summary": summary}
