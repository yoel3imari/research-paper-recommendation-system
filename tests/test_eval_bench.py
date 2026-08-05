"""This module tests src.bench_index.py and src.evaluate.py. See architecture §8.

The tests cover pure helper functions only. They use tiny synthetic data and
no real index, no snapshot, and no network. The tests mirror the style of
``test_index.py`` and use ``tmp_path`` for outputs.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.bench_index import p95, recall_at_k
from src.evaluate import (
    category_base_rates,
    category_purity,
    primary_cs_category,
    random_baseline,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        ann = np.array([[0, 1, 2, 3]])
        truth = np.array([[0, 1, 2, 3]])
        assert recall_at_k(ann, truth, 4) == 1.0

    def test_partial_recall(self):
        ann = np.array([[0, 1, 2, 3]])
        truth = np.array([[4, 5, 0, 1]])
        assert recall_at_k(ann, truth, 4) == 0.5

    def test_zero_relevant_returns_zero(self):
        ann = np.array([[0, 1]])
        truth = np.array([[-1, -1]])
        assert recall_at_k(ann, truth, 2) == 0.0

    def test_multiple_queries_averaged(self):
        ann = np.array([[0, 1], [2, 3]])
        truth = np.array([[0, 1], [9, 8]])
        # Query 0 has two of two hits. Query 1 has zero of two hits. The mean is 0.5.
        assert recall_at_k(ann, truth, 2) == 0.5

    def test_ignores_negative_labels(self):
        ann = np.array([[0, -1, -1, -1]])
        truth = np.array([[0, 1, 2, 3]])
        assert recall_at_k(ann, truth, 4) == 0.25


class TestP95:
    def test_basic_percentile(self):
        values = list(range(1, 101))  # The p95 is 95.05 with linear interpolation.
        assert 94.0 < p95(values) < 96.0

    def test_empty(self):
        assert p95([]) == 0.0


class TestPrimaryCsCategory:
    def test_first_cs_token_wins(self):
        assert primary_cs_category(["math.OC", "cs.LG", "stat.ML"]) == "cs.LG"

    def test_no_cs_token_returns_none(self):
        assert primary_cs_category(["math.OC", "q-bio.NC"]) is None

    def test_empty_list(self):
        assert primary_cs_category([]) is None

    def test_handles_none(self):
        assert primary_cs_category(None) is None


class TestCategoryBaseRates:
    def test_rates_sum_to_one(self):
        df = pl.DataFrame(
            {
                "categories": [
                    "cs.AI cs.LG",
                    "cs.LG",
                    "cs.CL",
                    "cs.AI",
                    "math.OC cs.LG",
                ]
            }
        )
        rates = category_base_rates(df)
        assert abs(sum(rates.values()) - 1.0) < 1e-9
        # cs.LG is primary in 2 of 5 rows, cs.AI in 2, and cs.CL in 1.
        assert abs(rates["cs.LG"] - 0.4) < 1e-9
        assert abs(rates["cs.AI"] - 0.4) < 1e-9
        assert abs(rates["cs.CL"] - 0.2) < 1e-9


class TestCategoryPurity:
    def test_all_matching(self):
        recs = [(0.9, "cs.LG"), (0.8, "cs.LG"), (0.7, "cs.LG")]
        assert category_purity(recs, "cs.LG", 3) == 1.0

    def test_mixed(self):
        recs = [(0.9, "cs.LG"), (0.8, "cs.AI"), (0.7, "cs.LG")]
        assert category_purity(recs, "cs.LG", 3) == 2 / 3

    def test_seed_without_category_is_zero(self):
        recs = [(0.9, "cs.LG")]
        assert category_purity(recs, None, 1) == 0.0

    def test_truncates_to_k(self):
        recs = [(0.9, "cs.LG"), (0.8, "cs.AI"), (0.7, "cs.AI")]
        assert category_purity(recs, "cs.LG", 1) == 1.0


class TestRandomBaseline:
    def test_uses_corpus_rate(self):
        rates = {"cs.LG": 0.4, "cs.AI": 0.35, "cs.CL": 0.25}
        assert random_baseline(rates, "cs.LG") == 0.4

    def test_missing_category_defaults_zero(self):
        assert random_baseline({"cs.LG": 0.4}, "cs.CL") == 0.0


class TestRender:
    def test_benchmark_report_round_trip(self, tmp_path):
        from src.bench_index import _render_markdown, write_report

        summary = {
            "profile": "ivf-sq8",
            "n": 1000,
            "recall_at_10": 0.95,
            "latency_ms_p95": 2.5,
            "latency_ms_mean": 1.2,
            "n_queries": 1000,
            "index_bytes": 4_000_000,
        }
        text = _render_markdown(summary)
        assert "Recall@10" in text and "0.95" in text
        path = write_report(summary, tmp_path / "bench.md")
        assert path.read_text() == text

    def test_eval_report_contains_queries(self, tmp_path):
        from src.evaluate import _render_eval_markdown, write_evaluation_report

        summary = {
            "n_seeds": 5,
            "purity": {1: 0.8, 5: 0.6, 10: 0.5},
            "baseline": {1: 0.3, 5: 0.3, 10: 0.3},
        }
        hits = [
            (
                "graph neural networks",
                [{"id": "2001.00001", "title": "GNN paper", "score": 0.9}],
            )
        ]
        text = _render_eval_markdown(summary, hits)
        assert "graph neural networks" in text
        assert "2001.00001" in text
        path = write_evaluation_report(summary, hits, tmp_path / "eval.md")
        assert path.read_text() == text
