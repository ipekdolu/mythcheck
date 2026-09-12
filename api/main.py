"""Phase 6: FastAPI backend exposing the verification pipeline.

Run with: uvicorn api.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from verification.pipeline import verify_text

app = FastAPI(title="MythCheck", description="Hallucination detection for AI-generated mythology text.")


class VerifyRequest(BaseModel):
    text: str


class ClaimVerdict(BaseModel):
    claim_text: str
    claim_type: str
    verdict: str
    citations: list[str]
    reasoning: str


class VerifySummary(BaseModel):
    total_claims: int
    supported_pct: float
    contradicted_pct: float
    unverifiable_pct: float


class VerifyResponse(BaseModel):
    verdicts: list[ClaimVerdict]
    summary: VerifySummary


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    result = verify_text(request.text)
    return VerifyResponse(**result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
