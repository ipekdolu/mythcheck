"""Resolve a RawClaim's entity names to graph node keys, reusing the same
matching approach (exact alias lookup, then embedding-similarity fallback)
built for ingestion-time entity resolution."""
from verification.claim_schema import RawClaim, ResolvedClaim
from verification.graph_lookup import EntityIndex


def resolve_claim(claim: RawClaim, index: EntityIndex) -> ResolvedClaim:
    unresolved: list[str] = []

    if claim.claim_type == "relational":
        source = index.resolve(claim.source_name, claim.source_type) if claim.source_name else None
        target = index.resolve(claim.target_name, claim.target_type) if claim.target_name else None
        if claim.source_name and source is None:
            unresolved.append(claim.source_name)
        if claim.target_name and target is None:
            unresolved.append(claim.target_name)
        return ResolvedClaim(
            text=claim.text,
            claim_type=claim.claim_type,
            source_key=source.key if source else None,
            relation=claim.relation,
            target_key=target.key if target else None,
            unresolved_entities=unresolved,
        )

    # descriptive
    subject = index.resolve_any_type(claim.subject_name) if claim.subject_name else None
    if claim.subject_name and subject is None:
        unresolved.append(claim.subject_name)
    return ResolvedClaim(
        text=claim.text,
        claim_type=claim.claim_type,
        subject_key=subject.key if subject else None,
        unresolved_entities=unresolved,
    )


def resolve_claims(claims: list[RawClaim], index: EntityIndex) -> list[ResolvedClaim]:
    return [resolve_claim(c, index) for c in claims]
