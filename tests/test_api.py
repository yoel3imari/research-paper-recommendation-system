"""This module tests the arXiv recommendation API. See docs/architecture.md §8, test_api.

The tests run fully offline. They do not download a model, and they use no
real embeddings. A tiny FAISS index uses synthetic L2-normalized vectors.
The encoder is a stub (``FakeEmbed``) that maps a fixed vocabulary to
deterministic vectors. Therefore, every assertion is pinned and deterministic.
"""

from __future__ import annotations

import threading

import numpy as np
import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import RecommendationService, _split_authors, _split_categories
from src.api.router import get_service, router
from src.api.schemas import paper_from_meta

DIM = 4

_SAMPLE_TEXTS = [
    "Attention is all you need A self-attention architecture for sequence.",
    "Transformer advances A self-attention architecture for sequence.",
    "Graph networks primer Message passing on graphs.",
    "Neural NLP Neural models for natural language.",
    "RAG Retrieval augmented generation.",
]


def _vec_for(text: str) -> np.ndarray:
    """Build a deterministic unit vector from the text to make the ranking reproducible."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _corpus_embeddings() -> np.ndarray:
    """Build deterministic unit embeddings for the five synthetic papers."""
    return np.stack([_vec_for(t) for t in _SAMPLE_TEXTS])


class FakeEmbed:
    """This deterministic stub maps a query string to a fixed unit vector."""

    dim = DIM

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.stack([_vec_for(t) for t in texts])


class FakeVectorIndex:
    """This fake index returns row labels ordered by the embedding inner product."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self._emb = embeddings

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self._emb.T
        order = np.argsort(-scores[0])[:k]
        return scores[:, order], order.reshape(1, -1).astype(np.int64)


@pytest.fixture
def corpus() -> pl.DataFrame:
    """This synthetic snapshot has five papers and includes category and year edge cases."""
    return pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4],
            "id": ["2001.00001", "2001.00002", "2015.00003", "2020.00004", "2020.00005"],
            "title": [
                "Attention is all you need",
                "Transformer advances",
                "Graph networks primer",
                "Deep learning for NLP",
                "Retrieval augmented generation",
            ],
            "abstract": [
                "A self-attention architecture for transduction.",
                "Extensions of the transformer architecture.",
                "Message passing on graphs.",
                "Neural models for natural language.",
                "Augmenting language models with retrieval.",
            ],
            "categories": ["cs.CL", "cs.CL cs.LG", "cs.LG", "cs.CL", "cs.AI"],
            "authors": ["Vaswani", "Devlin", "Scarselli", "Wang", "Lewis"],
            "update_date": [
                "2017-06-12",
                "2019-05-15",
                "2015-04-01",
                "2020-11-02",
                "2023-02-20",
            ],
        }
    )


def _build_service(df: pl.DataFrame) -> RecommendationService:
    """Build a RecommendationService with synthetic data and stub services."""
    svc = RecommendationService.__new__(RecommendationService)
    svc._encode_lock = threading.Lock()
    svc._encoder = FakeEmbed()
    svc.metadata = []
    svc.by_id = {}
    svc.n_papers = 0
    svc._titles_lower = []
    svc._index = FakeVectorIndex(_corpus_embeddings())
    for row in df.rows(named=True):
        meta = {
            "id": row["id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "authors": _split_authors(row["authors"]),
            "categories": _split_categories(row["categories"]),
            "update_date": row["update_date"],
            "year": int(row["update_date"][:4]),
        }
        svc.metadata.append(meta)
        svc.by_id[meta["id"]] = paper_from_meta(meta)
        svc.n_papers += 1
        svc._titles_lower.append(meta["title"].lower())
    return svc


@pytest.fixture
def service(corpus) -> RecommendationService:
    return _build_service(corpus)


@pytest.fixture
def client(service):
    """This app resolves the ``get_service`` dependency to the stub service."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["papers"] == 5


class TestPaperLookup:
    def test_known_paper(self, client):
        r = client.get("/api/v1/papers/2015.00003")
        assert r.status_code == 200
        body = r.json()
        assert body["title"] == "Graph networks primer"
        assert body["arxiv_url"] == "https://arxiv.org/abs/2015.00003"
        assert body["categories"] == ["cs.LG"]

    def test_unknown_paper_404(self, client):
        r = client.get("/api/v1/papers/9999.99999")
        assert r.status_code == 404

    def test_old_style_id_with_slash(self, client, service):
        # Older arXiv ids contain a slash (e.g. ``comp-gas/9305001``).
        service.by_id["comp-gas/9305001"] = paper_from_meta(
            {
                "id": "comp-gas/9305001",
                "title": "Old style paper",
                "abstract": "An older arXiv identifier.",
                "authors": ["A. Author"],
                "categories": ["cond-mat"],
                "update_date": "1993-05-01",
            }
        )

        lookup = client.get("/api/v1/papers/comp-gas/9305001")
        recommend = client.get("/api/v1/recommend/paper/comp-gas/9305001", params={"k": 2})

        assert lookup.status_code == 200
        assert lookup.json()["id"] == "comp-gas/9305001"
        assert recommend.status_code == 200


class TestPaperSearch:
    def test_substring_match(self, client):
        r = client.get("/api/v1/papers/search", params={"q": "attention"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["results"][0]["title"] == "Attention is all you need"

    def test_no_match(self, client):
        r = client.get("/api/v1/papers/search", params={"q": "zzzz"})
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_limit_respected(self, client):
        r = client.get("/api/v1/papers/search", params={"q": "a", "limit": 2})
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 2


class TestRecommendQuery:
    def test_returns_k_recommendations(self, client):
        r = client.get("/api/v1/recommend/query", params={"q": "attention", "k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 3
        assert body["k"] == 3
        assert all("score" in rec and "paper" in rec for rec in body["results"])
        scores = [rec["score"] for rec in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_year_min_filters(self, client):
        r = client.get(
            "/api/v1/recommend/query",
            params={"q": "attention", "k": 10, "year_min": 2019},
        )
        assert r.status_code == 200
        for rec in r.json()["results"]:
            assert int(rec["paper"]["update_date"][:4]) >= 2019

    def test_category_filter(self, client):
        r = client.get(
            "/api/v1/recommend/query",
            params={"q": "attention", "k": 10, "categories": "cs.LG"},
        )
        assert r.status_code == 200
        for rec in r.json()["results"]:
            assert "cs.LG" in rec["paper"]["categories"]


class TestRecommendPaper:
    def test_recommend_from_seed(self, client):
        r = client.get("/api/v1/recommend/paper/2001.00001", params={"k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 3
        assert body["query"]  # The query text is non-empty and comes from the seed paper.
        assert all(rec["paper"]["id"] for rec in body["results"])

    def test_unknown_seed_404(self, client):
        r = client.get("/api/v1/recommend/paper/9999.99999")
        assert r.status_code == 404
