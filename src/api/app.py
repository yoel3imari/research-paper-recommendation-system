"""FastAPI application and cached recommendation service (docs/architecture.md §5).

The ``RecommendationService`` is the single heavyweight object. It loads the
snapshot (metadata and title-search index), the FAISS vector index, and the
ONNX encoder once. Routes receive the service through ``get_service``. Tests
override this dependency for offline tests.

The lifespan loads the service at startup. If the artifacts are missing (for
example, the embed has not run), the app logs a warning. The metadata and
/health endpoints stay available. They fail cleanly on request.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
import polars as pl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import ModelConfig, Paths
from ..embed import OnnxEncoder
from ..index import load_index
from .router import router
from .schemas import Paper, paper_from_meta

logger = logging.getLogger(__name__)


class RecommendationService:
    """Owns the snapshot, FAISS index, and ONNX encoder for the API.

    Thread-safety: FAISS ``search`` is documented thread-safe for concurrent
    reads (no lock); ONNX ``encode`` is guarded by a single ``threading.Lock``
    because concurrent forward passes are not guaranteed safe.
    """

    def __init__(
        self,
        snapshot_path=Paths.SNAPSHOT,
        index_path=None,
        ids_path=Paths.IDS,
        encoder=None,
    ) -> None:
        self._encode_lock = threading.Lock()
        self._encoder = encoder
        self._df = pl.read_parquet(snapshot_path)
        self._index = (
            load_index(index_path) if index_path is not None else load_index(Paths.INDEX)
        )
        self._ids = np.load(ids_path)

        # Snapshot column layouts (architecture §4.1).
        ids_column = self._df["id"].to_list()
        titles = self._df["title"].to_list()
        abstracts = self._df["abstract"].to_list()
        authors = self._df["authors"].to_list()
        categories = self._df["categories"].to_list()
        dates = self._df["update_date"].to_list()

        self.metadata: list[dict] = []
        self.by_id: dict[str, Paper] = {}
        for i in range(len(ids_column)):
            cat_tokens = _split_categories(categories[i])
            meta = {
                "id": ids_column[i],
                "title": titles[i],
                "abstract": abstracts[i],
                "authors": _split_authors(authors[i]),
                "categories": cat_tokens,
                "update_date": dates[i],
                "year": int(dates[i][:4]) if dates[i] else 0,
            }
            self.metadata.append(meta)
            self.by_id[meta["id"]] = paper_from_meta(meta)
        self.n_papers = len(ids_column)
        self._titles_lower = [(t or "").lower() for t in titles]

    @property
    def dim(self) -> int:
        return ModelConfig.DIM

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = OnnxEncoder()
        return self._encoder

    def encode(self, text: str) -> np.ndarray:
        """L2-normalized ``(1, dim)`` embedding for a query (locked)."""
        with self._encode_lock:
            vectors = self._get_encoder().encode_texts([text], batch_size=1)
        return vectors[:1]

    def search_query(self, text: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Embed ``text`` and ANN-search; returns ``(scores, labels)`` (1, k)."""
        query = self.encode(text)
        return self._index.search(query, k)

    def title_matches(self, q: str, limit: int) -> list[int]:
        """Row labels whose title contains ``q`` (case-insensitive), up to ``limit``."""
        q = q.lower()
        hits: list[int] = []
        for i, title in enumerate(self._titles_lower):
            if q in title:
                hits.append(i)
                if len(hits) >= limit:
                    break
        return hits


# ── Builders ───────────────────────────────────────────────────────────────
def _split_categories(raw) -> list[str]:
    """Split a space-separated category string into tokens."""
    if raw is None:
        return []
    return [c for c in str(raw).split() if c]


def _split_authors(raw) -> list[str]:
    if raw is None:
        return []
    return [a.strip() for a in str(raw).split(",") if a.strip()]


@lru_cache(maxsize=1)
def cached_service() -> RecommendationService:
    """Load (and cache) the singleton service; used by ``get_service``/lifespan."""
    return RecommendationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the service at startup. If the artifacts are missing, degrade gracefully."""
    try:
        svc = cached_service()
        logger.info("RecommendationService loaded (papers=%d)", svc.n_papers)
    except Exception:
        logger.warning(
            "Could not load RecommendationService on startup — "
            "metadata/search endpoints may fail until artifacts exist",
            exc_info=True,
        )
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="arXiv Recommendation API",
        description=(
            "Item-to-item and query-to-item arXiv paper recommendations for the "
            "cs.AI, cs.LG, and cs.CL corpus. It uses ONNX embeddings and FAISS."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the single-page dashboard (same origin as the API)."""
        return FileResponse(Paths.ROOT / "dashboard" / "index.html", media_type="text/html")

    return app


app = create_app()