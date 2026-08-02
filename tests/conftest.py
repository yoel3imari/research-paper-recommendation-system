"""Shared fixtures for the test suite (docs/architecture.md §8)."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def synthetic_papers() -> pl.DataFrame:
    """Tiny synthetic corpus (3 categories) including ingest edge cases.

    Covers: one cross-shard duplicate id, one empty title, one whitespace-only
    abstract, one null-category row, one non-matching category, and out-of-order
    ids so the sort invariant is actually exercised.
    """
    rows = [
        # (id, title, abstract, categories)
        (
            "1001.0001",
            "Attention is all you need",
            "An attention mechanism for sequence transduction.",
            "cs.CL",
        ),
        ("1002.0001", "", "An orphan with no title but a long abstract.", "math.OC"),
        ("1001.0001", "Duplicate paper", "Same paper as id 1001.0001.", "cs.CL"),
        ("1003.0001", "BERT paper", "Pretraining of deep bidirectional transformers.", "cs.LG"),
        ("1004.0001", "Graph networks", "Message passing on graphs.", "math.OC"),
        ("1005.0001", "   ", "Whitespace-only title.", "cs.CL"),
        ("1006.0001", "Null abstract", "   ", "cs.LG"),
        ("1007.0001", "No categories", "A paper with a missing category field.", None),
    ]
    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "title": [r[1] for r in rows],
            "abstract": [r[2] for r in rows],
            "categories": [r[3] for r in rows],
            "authors": [f"Author {i}" for i in range(len(rows))],
            "update_date": [f"2024-0{i % 9 + 1}-15" for i in range(len(rows))],
        }
    )
