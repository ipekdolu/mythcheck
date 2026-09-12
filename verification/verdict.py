"""Phase 4: ground each resolved claim and produce a verdict.

Relational claims are checked by deterministic graph traversal - no LLM call,
no hallucination risk, and the citation is always a real chunk_id pulled
straight off the matching edge. Descriptive claims have no clean ontology
triple to traverse, so they're grounded via vector retrieval and judged by
Haiku 4.5 against the retrieved passages; if the model cites a chunk id that
wasn't actually retrieved, the request is rejected and regenerated once.
"""
from dataclasses import dataclass, field

from anthropic import Anthropic
from neo4j import Driver
from pydantic import BaseModel, ValidationError

from config import ANTHROPIC_API_KEY
from ingestion.vector_store import get_collection
from verification.claim_schema import ResolvedClaim

client = Anthropic(api_key=ANTHROPIC_API_KEY)
VERDICT_MODEL = "claude-haiku-4-5"

VERDICTS = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

# Relation-type pairs that cannot both hold between the same two entities.
# Kept deliberately small and conservative - mythology has real edge cases
# (e.g. Zeus and Hera are both siblings and spouses), so only pairs that are
# unambiguously exclusive are included.
MUTUALLY_EXCLUSIVE_RELATIONS = {frozenset({"rival_of", "ally_of"})}

ASYMMETRIC_RELATIONS = {
    "parent_of",
    "wields",
    "created",
    "rules",
    "resides_in",
    "guards",
    "defeats",
    "member_of_pantheon",
}


@dataclass
class Verdict:
    claim_text: str
    claim_type: str
    verdict: str
    citations: list[str] = field(default_factory=list)
    reasoning: str = ""


def ground_relational(claim: ResolvedClaim, driver: Driver) -> Verdict:
    if claim.source_key is None or claim.target_key is None:
        return Verdict(
            claim_text=claim.text,
            claim_type=claim.claim_type,
            verdict="UNVERIFIABLE",
            reasoning="One or both entities not found in the knowledge graph.",
        )

    with driver.session() as session:
        exact = session.run(
            "MATCH (a {key: $src})-[r]->(b {key: $tgt}) WHERE type(r) = $rel "
            "RETURN r.chunk_id AS chunk_id",
            src=claim.source_key,
            tgt=claim.target_key,
            rel=claim.relation,
        ).data()
        if exact:
            return Verdict(
                claim_text=claim.text,
                claim_type=claim.claim_type,
                verdict="SUPPORTED",
                citations=sorted({r["chunk_id"] for r in exact if r["chunk_id"]}),
                reasoning=f"Graph has a direct {claim.relation} edge between the two entities.",
            )

        if claim.relation in ASYMMETRIC_RELATIONS:
            reversed_match = session.run(
                "MATCH (a {key: $tgt})-[r]->(b {key: $src}) WHERE type(r) = $rel "
                "RETURN r.chunk_id AS chunk_id",
                src=claim.source_key,
                tgt=claim.target_key,
                rel=claim.relation,
            ).data()
            if reversed_match:
                return Verdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="CONTRADICTED",
                    citations=sorted({r["chunk_id"] for r in reversed_match if r["chunk_id"]}),
                    reasoning=f"Graph states the reverse {claim.relation} relationship.",
                )

        other_edges = session.run(
            "MATCH (a {key: $src})-[r]-(b {key: $tgt}) RETURN DISTINCT type(r) AS rel_type, "
            "r.chunk_id AS chunk_id",
            src=claim.source_key,
            tgt=claim.target_key,
        ).data()
        for row in other_edges:
            if frozenset({claim.relation, row["rel_type"]}) in MUTUALLY_EXCLUSIVE_RELATIONS:
                return Verdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="CONTRADICTED",
                    citations=[row["chunk_id"]] if row["chunk_id"] else [],
                    reasoning=f"Graph states {row['rel_type']}, which is incompatible with claimed {claim.relation}.",
                )

    return Verdict(
        claim_text=claim.text,
        claim_type=claim.claim_type,
        verdict="UNVERIFIABLE",
        reasoning="No matching or contradicting edge found between these entities.",
    )


class DescriptiveJudgment(BaseModel):
    verdict: str
    citation_chunk_id: str | None = None
    reasoning: str


JUDGE_TOOL = {
    "name": "judge_claim",
    "description": "Record the verdict for a claim given retrieved passages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "citation_chunk_id": {
                "type": "string",
                "description": "The chunk id of the passage that supports/contradicts the claim, or empty string if UNVERIFIABLE.",
            },
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "citation_chunk_id", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}

JUDGE_SYSTEM_PROMPT = """You judge whether a claim about mythology is SUPPORTED, CONTRADICTED, \
or UNVERIFIABLE, using ONLY the retrieved passages provided - not your own knowledge.

- SUPPORTED: a passage directly confirms the claim.
- CONTRADICTED: a passage directly states something incompatible with the claim.
- UNVERIFIABLE: the passages don't clearly confirm or contradict the claim.

You MUST cite the exact chunk id of the passage you used as citation_chunk_id (copy it exactly \
as given). If UNVERIFIABLE, set citation_chunk_id to an empty string."""


def _judge_descriptive(claim_text: str, passages: list[tuple[str, str]]) -> DescriptiveJudgment:
    passages_block = "\n\n".join(f"[{cid}]\n{text}" for cid, text in passages)
    response = client.messages.create(
        model=VERDICT_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "judge_claim"},
        messages=[
            {
                "role": "user",
                "content": f"Claim: {claim_text}\n\nRetrieved passages:\n{passages_block}",
            }
        ],
    )
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    return DescriptiveJudgment.model_validate(tool_block.input)


def ground_descriptive(claim: ResolvedClaim, n_results: int = 3) -> Verdict:
    collection = get_collection()
    results = collection.query(query_texts=[claim.text], n_results=n_results)
    ids = results["ids"][0]
    docs = results["documents"][0]
    passages = list(zip(ids, docs))

    if not passages:
        return Verdict(
            claim_text=claim.text, claim_type=claim.claim_type, verdict="UNVERIFIABLE",
            reasoning="No passages retrieved.",
        )

    valid_ids = set(ids)
    for attempt in range(2):  # reject-and-regenerate once on an invalid citation
        try:
            judgment = _judge_descriptive(claim.text, passages)
        except ValidationError:
            continue
        if judgment.verdict == "UNVERIFIABLE" or not judgment.citation_chunk_id:
            return Verdict(
                claim_text=claim.text, claim_type=claim.claim_type, verdict="UNVERIFIABLE",
                reasoning=judgment.reasoning,
            )
        if judgment.citation_chunk_id in valid_ids:
            return Verdict(
                claim_text=claim.text,
                claim_type=claim.claim_type,
                verdict=judgment.verdict,
                citations=[judgment.citation_chunk_id],
                reasoning=judgment.reasoning,
            )
        # citation didn't resolve to an actually-retrieved chunk - retry once
        continue

    return Verdict(
        claim_text=claim.text, claim_type=claim.claim_type, verdict="UNVERIFIABLE",
        reasoning="Judge could not produce a citation that resolved to a retrieved passage.",
    )


def get_verdict(claim: ResolvedClaim, driver: Driver) -> Verdict:
    if claim.claim_type == "relational":
        return ground_relational(claim, driver)
    return ground_descriptive(claim)
