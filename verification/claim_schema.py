"""Schema for atomic claims decomposed from AI-generated text to verify."""
from pydantic import BaseModel, Field

from ingestion.ontology import ENTITY_TYPES, RELATIONSHIP_NAMES

CLAIM_TYPES = ["relational", "descriptive"]


class RawClaim(BaseModel):
    """One atomic, checkable claim extracted from input text, before entity
    resolution against the graph."""

    text: str = Field(..., description="The claim in a short standalone sentence.")
    claim_type: str = Field(..., description="'relational' or 'descriptive'.")
    # For relational claims: an (source, relation, target) triple mapped onto
    # the ontology, so it can be checked via graph traversal.
    source_name: str | None = None
    source_type: str | None = None
    relation: str | None = None
    target_name: str | None = None
    target_type: str | None = None
    # For descriptive claims (no clean ontology triple, e.g. "associated with
    # wisdom"): checked via vector retrieval instead of graph traversal.
    subject_name: str | None = None


class ClaimExtractionResult(BaseModel):
    claims: list[RawClaim]


class ResolvedClaim(BaseModel):
    """A RawClaim after entity resolution against the knowledge graph."""

    text: str
    claim_type: str
    source_key: str | None = None
    relation: str | None = None
    target_key: str | None = None
    subject_key: str | None = None
    unresolved_entities: list[str] = Field(default_factory=list)
