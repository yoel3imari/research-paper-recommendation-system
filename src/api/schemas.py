"""Pydantic v2 response models for the arXiv recommendation API.

These models mirror ``docs/architecture.md`` §5. Every recommendation
response is a candidate paper plus a cosine similarity ``score``. The
response wraps the paper with the query metadata and the serving latency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ARXIV_URL_TEMPLATE = "https://arxiv.org/abs/{arxiv_id}"


class Paper(BaseModel):
    """Bibliographic metadata for a single arXiv paper (no full text)."""

    id: str = Field(..., description="arXiv identifier, for example 2006.00001")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    authors: list[str] = Field(..., description="Author names")
    categories: list[str] = Field(..., description="Category tokens, for example cs.LG")
    update_date: str = Field(..., description="Last update date (YYYY-MM-DD)")
    arxiv_url: str = Field(..., description="Link to the paper on arXiv")


class Recommendation(BaseModel):
    """A single recommendation: a paper and its cosine similarity score."""

    paper: Paper
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity to the query")


class RecommendResponse(BaseModel):
    """Ranked recommendation result for a query or seed paper."""

    query: str = Field(description="Query text that produced the recommendations")
    k: int = Field(description="Number of recommendations in the response")
    took_ms: float = Field(description="Serving latency in milliseconds")
    results: list[Recommendation] = Field(description="Recommendations ranked from best to worst")


class PaperSearchResponse(BaseModel):
    """Title-picker results used by the dashboard to select a seed paper."""

    results: list[Paper]
    total: int = Field(description="Total number of matching papers")


# ── Paper construction ─────────────────────────────────────────────────────
def paper_from_meta(meta: dict) -> Paper:
    """Build a ``Paper`` schema from a ``RecommendationService.metadata`` dict."""
    return Paper(
        id=meta["id"],
        title=meta["title"],
        abstract=meta["abstract"],
        authors=meta["authors"],
        categories=meta["categories"],
        update_date=meta["update_date"],
        arxiv_url=ARXIV_URL_TEMPLATE.format(arxiv_id=meta["id"]),
    )