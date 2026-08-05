"""API router for the arXiv recommendation endpoints (docs/architecture.md §5).

Endpoints are plain ``def`` functions (FastAPI threadpool). FAISS ``search``
is thread-safe for concurrent reads. ONNX encode uses a single
``threading.Lock`` inside the service. Do not use ``async def`` for blocking
work. The heavyweight service loads the snapshot, FAISS index, and ONNX
encoder once and caches them.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from .schemas import (
    Paper,
    PaperSearchResponse,
    Recommendation,
    RecommendResponse,
    paper_from_meta,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Fetch this many candidates from FAISS before post-filtering.
FETCH_K = 100


# ── Service dependency ─────────────────────────────────────────────────────
def get_service():
    """Return the cached ``RecommendationService`` (overridable in tests).

    Tests inject a stub service through ``app.dependency_overrides``. The real
    model and index are never loaded in the test suite.
    """
    from .app import cached_service

    return cached_service()


# Module-level dependency sentinel (avoids ruff B008 on `= Depends(...)`).
_service = Depends(get_service)


# ── Response builders (pure, testable) ─────────────────────────────────────
def _paper_from_row(service, row_label: int) -> Paper:
    """Build a ``Paper`` schema for the given snapshot row (FAISS label)."""
    return paper_from_meta(service.metadata[row_label])


def _recommend(
    service, query_text: str, k: int, year_min: int | None, categories: set[str]
) -> RecommendResponse:
    """Embed the query, ANN-search ``FETCH_K``, post-filter, rank, and return the top ``k``."""
    t0 = time.perf_counter()
    scores, labels = service.search_query(query_text, FETCH_K)

    scored: list[tuple[float, int]] = []
    for score, label in zip(scores[0], labels[0], strict=True):
        if label < 0 or label >= service.n_papers:
            continue
        meta = service.metadata[label]
        if year_min is not None and meta.get("year", 0) < year_min:
            continue
        if categories and not categories.intersection(meta["categories"]):
            continue
        scored.append((float(score), int(label)))

    ranked = sorted(scored, key=lambda t: t[0], reverse=True)[:k]
    results = [
        Recommendation(paper=_paper_from_row(service, label), score=score)
        for score, label in ranked
    ]
    took_ms = (time.perf_counter() - t0) * 1000
    return RecommendResponse(query=query_text, k=k, took_ms=took_ms, results=results)


# ── Endpoints ──────────────────────────────────────────────────────────────
@router.get("/health")
def health(service=_service) -> dict:
    """Liveness probe (used by the compose healthcheck)."""
    return {"status": "ok", "papers": service.n_papers}


@router.get("/papers/search", response_model=PaperSearchResponse)
def search_papers(
    q: str = Query(..., min_length=1, description="Paper title substring or keyword"),
    limit: int = Query(10, ge=1, le=100),
    service=_service,
) -> PaperSearchResponse:
    """Title picker for the dashboard.

    Return papers whose title contains ``q`` (case-insensitive).
    """
    labels = service.title_matches(q, limit)
    return PaperSearchResponse(
        results=[_paper_from_row(service, label) for label in labels],
        total=len(labels),
    )


# NOTE: register this route BEFORE the ``/papers/{arxiv_id}`` route below.
# The literal path ``/papers/search`` wins over the path-parameter match.
# FastAPI matches the routes in registration order.
@router.get("/papers/{arxiv_id}", response_model=Paper)
def get_paper(arxiv_id: str, service=_service) -> Paper:
    """Full metadata for one paper by arXiv id."""
    paper = service.by_id.get(arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not found")
    return paper


@router.get("/recommend/query", response_model=RecommendResponse)
def recommend_query(
    q: str = Query(..., min_length=1, description="Free-text research interest query"),
    k: int = Query(10, ge=1, le=100),
    year_min: int | None = Query(None, ge=1991, le=2030),
    categories: str | None = Query(None, description="Comma-separated category prefixes"),
    service=_service,
) -> RecommendResponse:
    """User-to-item. Embed the free text and return the top ``k`` similar papers."""
    cat_list = _parse_categories(categories)
    return _recommend(service, q, k, year_min, cat_list)


@router.get("/recommend/paper/{arxiv_id}", response_model=RecommendResponse)
def recommend_paper(
    arxiv_id: str,
    k: int = Query(10, ge=1, le=100),
    year_min: int | None = Query(None, ge=1991, le=2030),
    categories: str | None = Query(None, description="Comma-separated category prefixes"),
    service=_service,
) -> RecommendResponse:
    """Item-to-item recommendations.

    Embed the paper title and abstract, then return the top ``k`` similar
    papers.
    """
    target = service.by_id.get(arxiv_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not found")
    cat_list = _parse_categories(categories)
    query_text = f"{target.title} {target.abstract}"
    return _recommend(service, query_text, k, year_min, cat_list)


def _parse_categories(raw: str | None) -> set[str]:
    """Parse comma-separated category prefixes and ids into exact tokens (or an empty set)."""
    if not raw:
        return set()
    return {c.strip() for c in raw.split(",") if c.strip()}