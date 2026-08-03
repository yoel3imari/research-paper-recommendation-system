"""Pydantic v2 response models for the arXiv recommendation API.

Mirrors ``docs/architecture.md`` §5. Every recommendation response is a
candidate paper plus a cosine similarity ``score``, wrapped with the query
metadata and serving latency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Bibliographic metadata for a single arXiv paper (no full text)."""

    id: str = Field(..., description="arXiv identifier, e.g. 2006.00001")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    authors: list[str] = Field(..., description="Author names")
    categories: list[str] = Field(..., description="Category tokens, e.g. cs.LG")
    update_date: str = Field(..., description="Last update date (YYYY-MM-DD)")
    arxiv_url: str = Field(..., description="Link to the paper on arXiv")


class Recommendation(BaseModel):
    """A single recommendation: paper + cosine similarity score."""

    paper: Paper
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity to the query")


class RecommendResponse(BaseModel):
    """Ranked recommendation result for a query or seed paper."""

    query: str = Field(description="The text/query the recommendations were derived from")
    k: int = Field(description="Number of recommendations returned")
    took_ms: float = Field(description="Serving latency in milliseconds")
    results: list[Recommendation] = Field(description="Ranked recommendations, best first")


class PaperSearchResponse(BaseModel):
    """Title-picker results used by the dashboard to select a seed paper."""

    results: list[Paper]
    total: int = Field(description="Total number of matching papers")