"""Phase 3: decompose arbitrary text into atomic, checkable claims.

Uses Haiku 4.5 (cheap, and this decomposition task is much simpler than the
ingestion-time ontology extraction) with a strict tool schema. Every claim is
classified relational (checkable via graph traversal) or descriptive
(checkable via vector retrieval).
"""
from anthropic import Anthropic
from pydantic import ValidationError

from config import ANTHROPIC_API_KEY
from ingestion.ontology import ENTITY_TYPES, RELATIONSHIP_NAMES
from verification.claim_schema import ClaimExtractionResult

client = Anthropic(api_key=ANTHROPIC_API_KEY)

CLAIM_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = f"""You decompose a passage of text about mythology into atomic, independently \
checkable factual claims.

Rules:
- Each claim should be a single, standalone factual assertion (no compound sentences).
- Classify each claim as "relational" or "descriptive":
  - "relational": the claim states a specific relationship between two named entities that maps \
onto one of these relation types: {", ".join(RELATIONSHIP_NAMES)}. For these, also fill in \
source_name, source_type, relation, target_name, target_type. Valid entity types: \
{", ".join(ENTITY_TYPES)}.
  - "descriptive": everything else (epithets, domains, general characterization, e.g. \
"associated with wisdom", "known as the trickster"). For these, fill in only subject_name.
- Use the exact proper name as it appears in the text for source_name/target_name/subject_name.
- CRITICAL - source_name/target_name are defined by SEMANTIC ROLE in the relation, not by \
word order in the sentence. Each relation type has a fixed direction: source is always the one \
performing/holding the role named by the relation (e.g. for parent_of, source is the PARENT; for \
rules, source is the RULER; for wields, source is the WIELDER). A sentence can state the same \
fact with the roles in either grammatical order:
  - "Odin is the father of Thor" -> source_name=Odin, relation=parent_of, target_name=Thor
  - "Thor is the son of Odin" -> ALSO source_name=Odin, relation=parent_of, target_name=Thor \
(Odin is still the parent, even though "Thor" appears first in the sentence)
  - "Egypt is ruled by Ra" -> source_name=Ra, relation=rules, target_name=Egypt (Ra is the ruler)
  Always identify who holds the source role semantically before filling in the fields - never \
just copy the sentence's subject into source_name by default.
- Skip claims that are pure opinion, or too vague to check (e.g. "is very powerful").
- If the text contains no checkable claims, return an empty list."""

EXTRACTION_TOOL = {
    "name": "extract_claims",
    "description": "Record the atomic claims found in the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "claim_type": {"type": "string", "enum": ["relational", "descriptive"]},
                        "source_name": {"type": "string"},
                        "source_type": {"type": "string", "enum": ENTITY_TYPES},
                        "relation": {"type": "string", "enum": RELATIONSHIP_NAMES},
                        "target_name": {"type": "string"},
                        "target_type": {"type": "string", "enum": ENTITY_TYPES},
                        "subject_name": {"type": "string"},
                    },
                    "required": ["text", "claim_type"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
    "strict": True,
}


def extract_claims(text: str) -> ClaimExtractionResult:
    response = client.messages.create(
        model=CLAIM_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_claims"},
        messages=[{"role": "user", "content": text}],
    )
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        return ClaimExtractionResult(claims=[])
    try:
        return ClaimExtractionResult.model_validate(tool_block.input)
    except ValidationError:
        return ClaimExtractionResult(claims=[])
