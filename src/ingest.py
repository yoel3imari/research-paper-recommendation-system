"""Streaming ingest of the arXiv corpus (docs/architecture.md §4.1).

Streams the HuggingFace ``open-index/open-arxiv`` Parquet mirror once, in
batches, never holding the full corpus in RAM. Produces the canonical
snapshot:

* ``data/processed/snapshot.parquet`` — deduplicated, filtered, sorted by
  ``id``, with a dense integer ``row_index`` column.
* ``data/processed/ids.npy`` — ``np.arange(N, dtype=np.int64)``, the alignment
  key: row ``i`` of every downstream artifact corresponds to snapshot row
  ``ids[i]``.

The pipeline is two-pass by design:

1. ``_stream_cleaned_batches`` — one network pass; each batch is filtered
   (empty title/abstract dropped, optional ``CORPUS_FILTER``) and written to a
   small staging parquet under ``data/raw/staging/``.
2. ``merge_snapshot`` — scans the staged parquets lazily, deduplicates on
   ``id``, sorts by ``id`` and sinks the snapshot out-of-core (polars
   streaming engine), so peak memory stays bounded regardless of corpus size.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
from datasets import load_dataset

from .config import Paths, Settings

logger = logging.getLogger(__name__)

# ── Ingest parameters ────────────────────────────────────────────────────────
SNAPSHOT_COLUMNS: list[str] = [
    "id",
    "title",
    "abstract",
    "categories",
    "authors",
    "update_date",
]
BATCH_SIZE: int = 8192
STAGING_DIR: Path = Paths.DATA_RAW / "staging"


# ── Batch cleaning (pure, unit-testable) ────────────────────────────────────
def category_pattern(corpus_filter: str | None) -> re.Pattern[str] | None:
    """Compile the ``CORPUS_FILTER`` regex, or return ``None`` to keep all.

    The pattern is matched with ``re.search`` against each whitespace-separated
    category token (a paper is kept if any of its categories matches).
    """
    if not corpus_filter or not corpus_filter.strip():
        return None
    return re.compile(corpus_filter.strip())


def clean_batch(df: pl.DataFrame, pattern: re.Pattern[str] | None) -> pl.DataFrame:
    """Reduce a raw batch to the snapshot fields and drop unusable rows.

    Drops rows whose ``title`` or ``abstract`` is null/empty/whitespace-only,
    and (when ``pattern`` is not ``None``) rows where no category token matches
    the ``CORPUS_FILTER`` regex. Null categories are dropped only when a filter
    is active (we cannot verify their category).
    """
    df = df.select(SNAPSHOT_COLUMNS)
    df = df.filter(
        pl.col("title").str.strip_chars().str.len_chars() > 0,
        pl.col("abstract").str.strip_chars().str.len_chars() > 0,
    )
    if pattern is not None:
        category_match = (
            pl.col("categories")
            .fill_null("")
            .str.split(by=" ", literal=True)
            .list.eval(pl.element().str.contains(pattern.pattern))
            .list.any()
        )
        df = df.filter(category_match)
    return df


# ── Streaming pass ───────────────────────────────────────────────────────────
def _iter_raw_batches(dataset_name: str) -> Iterator[pl.DataFrame]:
    """Stream the HF mirror once, yielding raw polars batches.

    Columns are selected at load time (``columns=``) so only the snapshot
    fields cross the network. ``iter(batch_size=...)`` advances a single stream
    sequentially — unlike per-index ``shard()`` calls, this never re-reads
    earlier shards.
    """
    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        columns=SNAPSHOT_COLUMNS,
    )
    for batch in ds.iter(batch_size=BATCH_SIZE, drop_last_batch=False):
        yield pl.from_dict(batch)


def _stream_cleaned_batches(
    dataset_name: str,
    pattern: re.Pattern[str] | None,
    staging_dir: Path,
    max_batches: int | None = None,
) -> tuple[int, int]:
    """Single network pass: clean each batch and stage it as parquet.

    Returns ``(batches_written, rows_written)``. ``max_batches`` bounds the
    pass for smoke runs; ``None`` means the full corpus.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    batches_written = 0
    rows_written = 0
    for i, raw in enumerate(_iter_raw_batches(dataset_name)):
        if max_batches is not None and i >= max_batches:
            break
        cleaned = clean_batch(raw, pattern)
        if cleaned.height == 0:
            continue
        cleaned.write_parquet(staging_dir / f"batch_{i:05d}.parquet")
        batches_written += 1
        rows_written += cleaned.height
    logger.info("staged %d batches / %d rows", batches_written, rows_written)
    return batches_written, rows_written


# ── Snapshot merge (out-of-core) ─────────────────────────────────────────────
def merge_snapshot(
    staging_dir: Path,
    snapshot_path: Path,
    ids_path: Path,
) -> int:
    """Merge staged batches into the canonical snapshot. Returns row count.

    Lazily scans all staged parquets, deduplicates on ``id`` (cross-shard
    duplicates exist in the HF mirror), sorts by ``id``, assigns the dense
    integer ``row_index`` and sinks ``snapshot.parquet`` via the polars
    streaming engine (out-of-core). ``ids.npy`` is the dense index array.
    """
    lazy = (
        pl.scan_parquet(staging_dir / "batch_*.parquet")
        .select(SNAPSHOT_COLUMNS)
        .unique(subset=["id"], keep="first")
        .sort("id")
        .with_row_index("row_index")
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lazy.sink_parquet(snapshot_path)
    n = int(pl.scan_parquet(snapshot_path).select(pl.len()).collect().item())
    np.save(ids_path, np.arange(n, dtype=np.int64))
    logger.info("snapshot: %d rows -> %s", n, snapshot_path)
    return n


# ── Entrypoint ───────────────────────────────────────────────────────────────
def run(
    dataset_name: str | None = None,
    corpus_filter: str | None = None,
    staging_dir: Path | None = None,
    snapshot_path: Path | None = None,
    ids_path: Path | None = None,
    max_batches: int | None = None,
) -> int:
    """Run the full ingest pipeline. Returns the snapshot row count.

    All paths default to ``src.config.Paths`` / ``Settings``; ``corpus_filter``
    falls back to the ``CORPUS_FILTER`` env var (``None`` = keep all).
    """
    settings = Settings()
    dataset_name = dataset_name or settings.dataset_name
    corpus_filter = settings.corpus_filter if corpus_filter is None else corpus_filter
    staging_dir = staging_dir or STAGING_DIR
    snapshot_path = snapshot_path or Paths.SNAPSHOT
    ids_path = ids_path or Paths.IDS

    Paths.ensure()
    pattern = category_pattern(corpus_filter)
    _stream_cleaned_batches(dataset_name, pattern, staging_dir, max_batches)
    return merge_snapshot(staging_dir, snapshot_path, ids_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the arXiv snapshot (docs/architecture.md §4.1)."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after this many batches (smoke runs); default: full corpus",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = run(max_batches=args.max_batches)
    print(f"ingest complete: {n} rows in snapshot")


if __name__ == "__main__":
    main()
