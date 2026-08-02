"""Tests for src.embed.py (docs/architecture.md §8, test_embed).

Uses a stub ``Encoder`` — the real ONNX model is never constructed here (it
would download ``model_quint8_avx2.onnx`` + tokenizer), so these tests run
fully offline with no network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.embed import (
    build_texts,
    chunk_is_complete,
    encode_chunk,
    iter_chunks,
    l2_normalize,
    load_manifest,
    run,
    save_manifest,
    write_chunk,
)

DIM = 384


class StubEncoder:
    """Deterministic L2-normalized stub standing in for the ONNX model."""

    dim = DIM

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vectors = rng.standard_normal((len(texts), self.dim))
        return l2_normalize(vectors)


@pytest.fixture
def texts():
    return ["paper one abstract", "paper two abstract", "paper three abstract"]


class TestL2Normalize:
    def test_returns_unit_norm_rows(self):
        vectors = np.array([[3.0, 0.0], [0.0, 4.0]])
        result = l2_normalize(vectors)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0]])

    def test_shape_preserved(self):
        vectors = np.random.default_rng(0).standard_normal((5, DIM))
        assert l2_normalize(vectors).shape == (5, DIM)


class TestBuildTexts:
    def test_composes_title_and_abstract(self):
        df = pl.DataFrame({"title": ["T1"], "abstract": ["A1"]})
        assert build_texts(df) == ["T1 A1"]


class TestIterChunks:
    def test_splits_into_fixed_chunks(self):
        df = pl.DataFrame({"v": list(range(250))})
        chunks = list(iter_chunks(df, 100))
        assert [start for start, _ in chunks] == [0, 100, 200]

    def test_last_chunk_is_partial(self):
        df = pl.DataFrame({"v": list(range(250))})
        last_start, last_chunk = list(iter_chunks(df, 100))[-1]
        assert last_chunk.height == 50


class TestEncodeChunk:
    def test_returns_normalized_vectors(self, texts):
        result = encode_chunk(texts, StubEncoder())
        assert result.shape == (len(texts), DIM)
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1), np.ones(len(texts)), rtol=1e-6
        )


class TestManifest:
    def test_load_returns_empty_dict_when_absent(self, tmp_path):
        assert load_manifest(tmp_path / "manifest.json") == {}

    def test_save_and_load_round_trip(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest = {"0": {"start": 0, "end": 10, "rows": 10, "shape": [10, DIM]}}
        save_manifest(manifest, manifest_path)
        assert load_manifest(manifest_path) == manifest

    def test_writes_then_renames_no_leftover_tmp(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        save_manifest({"0": {"rows": 10, "shape": [10, DIM]}}, manifest_path)
        assert manifest_path.exists()
        assert not (tmp_path / "manifest.json.tmp").exists()


class TestChunkIsComplete:
    @staticmethod
    def entry(rows: int) -> dict:
        return {"0": {"rows": rows, "shape": [rows, DIM]}}

    def test_complete_when_shape_matches(self):
        assert chunk_is_complete(self.entry(150), 0, 150, DIM) is True

    def test_incomplete_when_chunk_absent(self):
        assert chunk_is_complete({}, 0, 150, DIM) is False

    def test_incomplete_when_row_count_mismatches(self):
        assert chunk_is_complete(self.entry(150), 0, 151, DIM) is False

    def test_incomplete_when_dim_mismatches(self):
        assert chunk_is_complete(self.entry(150), 0, 150, DIM + 1) is False


class TestWriteChunk:
    def test_writes_npy_and_manifest_entry(self, tmp_path):
        embeddings_dir = tmp_path / "embeddings"
        manifest_path = tmp_path / "manifest.json"
        vectors = np.ones((10, DIM))
        manifest: dict = {}

        write_chunk(vectors, embeddings_dir, 0, 0, manifest, manifest_path)

        artifact = embeddings_dir / "chunk_00000.npy"
        assert artifact.exists()
        np.testing.assert_allclose(np.load(artifact), vectors)
        assert manifest["0"] == {
            "start": 0,
            "end": 10,
            "rows": 10,
            "shape": [10, DIM],
        }
        assert load_manifest(manifest_path)["0"] == manifest["0"]


def _snapshot(n_rows: int, tmp_path) -> Path:
    """Write a minimal snapshot parquet; return its path."""
    df = pl.DataFrame(
        {
            "id": [str(i) for i in range(n_rows)],
            "title": [f"title {i}" for i in range(n_rows)],
            "abstract": [f"abstract {i}" for i in range(n_rows)],
        }
    )
    path = tmp_path / "snapshot.parquet"
    df.write_parquet(path)
    return path


class TestRun:
    def test_writes_all_chunks_with_correct_extents(self, tmp_path):
        embeddings_dir = tmp_path / "embeddings"
        manifest_path = tmp_path / "manifest.json"

        summary = run(
            StubEncoder(),
            snapshot_path=_snapshot(250, tmp_path),
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            chunk_size=100,
        )

        assert summary == {"chunks_written": 3, "chunks_skipped": 0}
        chunks = sorted(p.name for p in embeddings_dir.glob("chunk_*.npy"))
        assert chunks == ["chunk_00000.npy", "chunk_00001.npy", "chunk_00002.npy"]
        manifest = load_manifest(manifest_path)
        assert set(manifest) == {"0", "1", "2"}
        assert manifest["0"]["start"] == 0
        assert manifest["0"]["rows"] == 100
        assert manifest["2"]["start"] == 200
        assert manifest["2"]["rows"] == 50


class TestRunResume:
    def test_skips_chunks_already_complete(self, tmp_path):
        embeddings_dir = tmp_path / "embeddings"
        manifest_path = tmp_path / "manifest.json"
        snapshot = _snapshot(250, tmp_path)

        run(
            StubEncoder(),
            snapshot_path=snapshot,
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            chunk_size=100,
        )
        second = run(
            StubEncoder(),
            snapshot_path=snapshot,
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            chunk_size=100,
        )

        assert second == {"chunks_written": 0, "chunks_skipped": 3}

    def test_no_resume_writes_everything_again(self, tmp_path):
        embeddings_dir = tmp_path / "embeddings"
        manifest_path = tmp_path / "manifest.json"
        snapshot = _snapshot(250, tmp_path)

        run(
            StubEncoder(),
            snapshot_path=snapshot,
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            chunk_size=100,
        )
        fresh = run(
            StubEncoder(),
            snapshot_path=snapshot,
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            chunk_size=100,
            resume=False,
        )

        assert fresh == {"chunks_written": 3, "chunks_skipped": 0}