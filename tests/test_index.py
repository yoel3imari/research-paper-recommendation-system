"""This module tests src.index.py. See docs/architecture.md §8, test_index.

The tests build and search tiny synthetic FAISS indexes. They use no real
model, no network, and no ``src.embed`` encoder. The tests write L2-normalized
vectors directly as chunk ``.npy`` files and a manifest. This format mirrors
the output layout of ``src.embed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.index import (
    FaissVectorIndex,
    VectorIndex,
    build_index,
    clamp_nlist,
    load_index,
    make_index,
)

DIM = 64


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _write_chunks(tmp_path: Path, rows_per_chunk: int, n_chunks: int) -> Path:
    """Write synthetic L2-normalized chunk .npy files and a manifest, then return the directory."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest: dict[str, dict] = {}
    start = 0
    for chunk_index in range(n_chunks):
        vectors = _normalize(rng.standard_normal((rows_per_chunk, DIM)))
        np.save(embeddings_dir / f"chunk_{chunk_index:05d}.npy", vectors)
        manifest[str(chunk_index)] = {
            "start": start,
            "end": start + rows_per_chunk,
            "rows": rows_per_chunk,
            "shape": [rows_per_chunk, DIM],
        }
        start += rows_per_chunk
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return embeddings_dir


class TestClampNlist:
    def test_clamps_to_quarter_of_train(self):
        assert clamp_nlist(8192, 100) == 25

    def test_never_below_one(self):
        assert clamp_nlist(8192, 2) == 1

    def test_keeps_small_nlist(self):
        assert clamp_nlist(4, 100) == 4


class TestMakeIndex:
    def test_default_profile_is_ivf_sq8(self):
        index = make_index("ivf-sq8", DIM, nlist=8)
        assert isinstance(index, faiss.IndexIVFScalarQuantizer)
        assert index.nprobe == 64

    def test_hnsw_profile(self):
        index = make_index("hnsw", DIM)
        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.hnsw.efConstruction == 200
        assert index.hnsw.efSearch == 64

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            make_index("nope", DIM)


class TestBuildSearchRoundTrip:
    def test_builds_and_searches_self_hit(self, tmp_path):
        embeddings_dir = _write_chunks(tmp_path, rows_per_chunk=20, n_chunks=3)
        meta = build_index(
            embeddings_dir=embeddings_dir,
            manifest_path=tmp_path / "manifest.json",
            index_path=tmp_path / "arxiv.index",
            meta_path=tmp_path / "index_meta.json",
            profile="ivf-sq8",
            nlist=8,
        )
        assert meta["n"] == 60
        assert meta["d"] == DIM
        assert meta["profile"] == "ivf-sq8"

        index = load_index(tmp_path / "arxiv.index")
        assert index.ntotal == 60
        query = np.load(embeddings_dir / "chunk_00000.npy")[0]
        scores, labels = index.search(query.reshape(1, -1), 5)
        assert labels[0][0] == 0
        assert scores[0][0] > 0.9


class TestCosineCorrectness:
    def test_inner_product_equals_cosine_for_normalized(self):
        v1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        v2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        index = faiss.IndexFlatIP(3)
        index.add(v1)
        index.add(v2)
        vi = FaissVectorIndex(index)
        scores, labels = vi.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), k=2)
        np.testing.assert_allclose(scores[0], [1.0, 0.0], atol=1e-6)
        assert labels[0][0] == 0


class TestIdAlignment:
    def test_labels_map_through_ids(self):
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        vi = FaissVectorIndex(index, ids=np.array([42]))
        _, labels = vi.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), k=1)
        assert labels[0][0] == 42


class TestVectorIndexContract:
    def test_faiss_vector_index_satisfies_protocol(self):
        vi = FaissVectorIndex(faiss.IndexFlatIP(3))
        assert isinstance(vi, VectorIndex)

    def test_search_returns_scores_and_labels(self):
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        vi = FaissVectorIndex(index)
        scores, labels = vi.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), k=1)
        assert scores.shape == (1, 1)
        assert labels.shape == (1, 1)


class TestHNSWBuild:
    def test_builds_hnsw_profile(self, tmp_path):
        embeddings_dir = _write_chunks(tmp_path, rows_per_chunk=20, n_chunks=2)
        meta = build_index(
            embeddings_dir=embeddings_dir,
            manifest_path=tmp_path / "manifest.json",
            index_path=tmp_path / "arxiv.index",
            meta_path=tmp_path / "index_meta.json",
            profile="hnsw",
        )
        assert meta["profile"] == "hnsw"
        assert meta["n"] == 40
        assert isinstance(load_index(tmp_path / "arxiv.index"), faiss.IndexHNSWFlat)


class TestMeta:
    def test_ivf_meta_records_params(self, tmp_path):
        embeddings_dir = _write_chunks(tmp_path, rows_per_chunk=20, n_chunks=1)
        build_index(
            embeddings_dir=embeddings_dir,
            manifest_path=tmp_path / "manifest.json",
            index_path=tmp_path / "arxiv.index",
            meta_path=tmp_path / "index_meta.json",
            profile="ivf-sq8",
            nlist=8,
        )
        meta = json.loads((tmp_path / "index_meta.json").read_text())
        assert meta["d"] == DIM
        assert meta["n"] == 20
        assert meta["params"] == {"nprobe": 64}


class TestCheckpoint:
    def test_checkpoints_during_build(self, tmp_path):
        embeddings_dir = _write_chunks(tmp_path, rows_per_chunk=20, n_chunks=3)
        index_path = tmp_path / "arxiv.index"
        build_index(
            embeddings_dir=embeddings_dir,
            manifest_path=tmp_path / "manifest.json",
            index_path=index_path,
            meta_path=tmp_path / "index_meta.json",
            profile="ivf-sq8",
            nlist=8,
            checkpoint_every=20,
        )
        assert index_path.exists()
        assert load_index(index_path).ntotal == 60


class TestMissingEmbeddings:
    def test_raises_when_no_chunks(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(
                embeddings_dir=tmp_path / "embeddings",
                manifest_path=tmp_path / "manifest.json",
                index_path=tmp_path / "arxiv.index",
                meta_path=tmp_path / "index_meta.json",
            )