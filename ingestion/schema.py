"""Pydantic schema for validating LLM extraction output."""
from pydantic import BaseModel, Field, field_validator

from ingestion.ontology import ENTITY_TYPES, RELATIONSHIP_NAMES


class ExtractedRelation(BaseModel):
    source_name: str = Field(..., min_length=1)
    source_type: str
    relation: str
    target_name: str = Field(..., min_length=1)
    target_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("source_type", "target_type")
    @classmethod
    def check_entity_type(cls, v: str) -> str:
        if v not in ENTITY_TYPES:
            raise ValueError(f"{v!r} is not a valid entity type; must be one of {ENTITY_TYPES}")
        return v

    @field_validator("relation")
    @classmethod
    def check_relation_type(cls, v: str) -> str:
        if v not in RELATIONSHIP_NAMES:
            raise ValueError(f"{v!r} is not a valid relation type; must be one of {RELATIONSHIP_NAMES}")
        return v


class ExtractionResult(BaseModel):
    relations: list[ExtractedRelation]
