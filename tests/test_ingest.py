"""Tests for src.ingest.py (docs/architecture.md §8, test_ingest)."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import Paths
from src.ingest import SNAPSHOT_COLUMNS, category_pattern, clean_batch, merge_snapshot

CS_FILTER = r"^cs\..*"


class TestCleanBatch:
    def test_selects_only_snapshot_columns(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert result.columns == SNAPSHOT_COLUMNS

    def test_drops_empty_title(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert "1002.0001" not in result["id"].to_list()

    def test_drops_whitespace_only_title(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert "1005.0001" not in result["id"].to_list()

    def test_drops_whitespace_only_abstract(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert "1006.0001" not in result["id"].to_list()

    def test_keeps_papers_with_missing_categories_when_unfiltered(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert "1007.0001" in result["id"].to_list()

    def test_keeps_rows_with_valid_title_and_abstract(self, synthetic_papers):
        result = clean_batch(synthetic_papers, None)
        assert result.height == 5  # 8 rows minus 3 dropped


class TestCategoryPattern:
    def test_none_for_empty_filter(self):
        assert category_pattern(None) is None
        assert category_pattern("") is None
        assert category_pattern("  ") is None

    def test_compiles_nonempty_filter(self):
        pattern = category_pattern(CS_FILTER)
        assert pattern is not None
        assert pattern.pattern == CS_FILTER


class TestCategoryFilter:
    def test_keeps_only_matching_categories(self, synthetic_papers):
        result = clean_batch(synthetic_papers, category_pattern(CS_FILTER))
        kept = result["categories"].to_list()
        assert all(cat is not None and cat.startswith("cs.") for cat in kept)

    def test_drops_nonmatching_category(self, synthetic_papers):
        result = clean_batch(synthetic_papers, category_pattern(CS_FILTER))
        assert "1004.0001" not in result["id"].to_list()

    def test_drops_null_category_when_filtered(self, synthetic_papers):
        result = clean_batch(synthetic_papers, category_pattern(CS_FILTER))
        assert "1007.0001" not in result["id"].to_list()

    def test_matches_any_token_in_multi_category_string(self):
        raw = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "title": ["t1", "t2", "t3"],
                "abstract": ["a1", "a2", "a3"],
                "categories": ["cs.LG", "math.OC", "cs.CL stat.ML"],
                "authors": ["x", "y", "z"],
                "update_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            }
        )
        result = clean_batch(raw, category_pattern(CS_FILTER))
        assert result["id"].to_list() == ["a", "c"]


class TestMergeSnapshot:
    def _stage_batches(self, tmp_path, batches, filename="batch_00000.parquet"):
        for i, batch in enumerate(batches):
            path = tmp_path / f"batch_{i:05d}.parquet"
            batch.write_parquet(path)
        return tmp_path

    def test_dedupes_on_id_across_batches(self, tmp_path, synthetic_papers):
        first = synthetic_papers.head(2)   # contains id 1001.0001
        second = synthetic_papers.tail(6)  # contains id 1001.0001 again
        staging = self._stage_batches(tmp_path, [first, second])
        n = merge_snapshot(staging, tmp_path / "snapshot.parquet", tmp_path / "ids.npy")
        snapshot = pl.read_parquet(tmp_path / "snapshot.parquet")
        assert n == snapshot.height
        assert snapshot["id"].is_unique().all()

    def test_snapshot_sorted_by_id(self, tmp_path, synthetic_papers):
        staging = self._stage_batches(tmp_path, [synthetic_papers])
        merge_snapshot(staging, tmp_path / "snapshot.parquet", tmp_path / "ids.npy")
        snapshot = pl.read_parquet(tmp_path / "snapshot.parquet")
        assert snapshot["id"].to_list() == sorted(snapshot["id"].to_list())

    def test_dense_integer_row_index(self, tmp_path, synthetic_papers):
        staging = self._stage_batches(tmp_path, [synthetic_papers])
        n = merge_snapshot(staging, tmp_path / "snapshot.parquet", tmp_path / "ids.npy")
        snapshot = pl.read_parquet(tmp_path / "snapshot.parquet")
        assert snapshot["row_index"].to_list() == list(range(n))

    def test_ids_npy_matches_snapshot_rows(self, tmp_path, synthetic_papers):
        staging = self._stage_batches(tmp_path, [synthetic_papers])
        n = merge_snapshot(staging, tmp_path / "snapshot.parquet", tmp_path / "ids.npy")
        ids = np.load(tmp_path / "ids.npy")
        assert ids.dtype == np.int64
        assert ids.shape == (n,)
        assert ids.tolist() == list(range(n))

    def test_snapshot_is_written_to_configured_path(self, tmp_path, synthetic_papers):
        staging = self._stage_batches(tmp_path, [synthetic_papers])
        snapshot_path = tmp_path / "out" / "snapshot.parquet"
        ids_path = tmp_path / "out" / "ids.npy"
        merge_snapshot(staging, snapshot_path, ids_path)
        assert snapshot_path.exists()
        assert ids_path.exists()


def test_ingest_module_paths_match_config():
    assert SNAPSHOT_COLUMNS == ["id", "title", "abstract", "categories", "authors", "update_date"]
    assert Paths.SNAPSHOT.name == "snapshot.parquet"
    assert Paths.IDS.name == "ids.npy"
