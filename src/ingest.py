"""Ingests the arXiv corpus (docs/architecture.md §4.1).

The module streams the HuggingFace ``open-index/open-arxiv`` Parquet mirror
once, in batches. It never holds the full corpus in RAM. It produces the
canonical snapshot:

* ``data/processed/snapshot.parquet`` — the deduplicated, filtered snapshot,
  sorted by ``id``. It contains a dense integer ``row_index`` column.
* ``data/processed/ids.npy`` — ``np.arange(N, dtype=np.int64)``. It is the
  alignment key: row ``i`` of every downstream artifact corresponds to
  snapshot row ``ids[i]``.

The pipeline has two passes:

1. ``_stream_cleaned_batches`` — one network pass. It filters each batch and
   writes it to a small staging parquet under ``data/raw/staging/``. The
   filter drops empty titles and abstracts and applies the optional
   ``CORPUS_FILTER``.
2. ``merge_snapshot`` — scans the staged parquets lazily. Deduplicates on
   ``id``, sorts by ``id``, and sinks the snapshot out-of-core with the polars
   streaming engine. Peak memory stays bounded regardless of corpus size.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
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

# ── Ingest scope selection ───────────────────────────────────────────────────
SCOPE_FULL = "full"
SCOPE_CATEGORY = "category"
SCOPE_SUBCATEGORY = "subcategory"

# Examples shown in the interactive prompt. arXiv archives are the dotted
# prefix of a category id; subcategories are the full id, e.g. ``cs.AI``.
EXAMPLE_CATEGORIES = "cs, math, stat, q-bio, q-fin, eess, econ, astro-ph, cond-mat, hep-th"


def scope_pattern(
    scope: str, category: str | None = None, subcategory: str | None = None
) -> str | None:
    r"""Converts an ingest scope into a ``CORPUS_FILTER`` regex.

    * ``SCOPE_FULL`` → ``None`` (keep every paper).
    * ``SCOPE_CATEGORY`` → ``^<category>\.`` — keeps every token under an
      archive, e.g. ``cs`` retains ``cs.AI``, ``cs.LG``, ``cs.CL``, ...
    * ``SCOPE_SUBCATEGORY`` → ``^<subcategory>$`` — keeps the exact arXiv
      category id, e.g. ``cs.AI`` retains only ``cs.AI`` papers.

    ``clean_batch`` ``re.search``-es the pattern against each
    whitespace-separated category token; a paper survives when any token
    matches. Raises ``ValueError`` for an unknown scope or a missing value.
    """
    if scope == SCOPE_FULL:
        return None
    if scope == SCOPE_CATEGORY:
        if not category or not category.strip():
            raise ValueError(f"a category name is required for scope={scope!r}")
        return rf"^{re.escape(category.strip())}\."
    if scope == SCOPE_SUBCATEGORY:
        if not subcategory or not subcategory.strip():
            raise ValueError(f"a subcategory name is required for scope={scope!r}")
        return rf"^{re.escape(subcategory.strip())}$"
    raise ValueError(f"unknown ingest scope: {scope!r}")


# ── Category vocabulary (discovery) ──────────────────────────────────────────
SAMPLE_BATCHES: int = 50


def collect_category_counts(
    batches: Iterator[pl.DataFrame],
    sample_batches: int | None = SAMPLE_BATCHES,
) -> dict[str, int]:
    """Counts distinct category tokens over a bounded sample of the mirror.

    ``batches`` is the raw-stream iterator (e.g. ``_iter_raw_batches``);
    iteration stops after ``sample_batches`` batches. Returns
    ``{token: paper_count}`` for every non-empty whitespace token in the
    ``categories`` column; a paper listing several categories contributes to
    each token.
    """
    counts: Counter[str] = Counter()
    for i, raw in enumerate(batches):
        if sample_batches is not None and i >= sample_batches:
            break
        tokens = raw["categories"].fill_null("").str.split(by=" ", literal=True).explode()
        counts.update(t for t in tokens.to_list() if t)
    return dict(counts)


def summarize_categories(
    counts: dict[str, int],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Splits counted category tokens into ``(archives, subcategories)``.

    Each list is sorted by count descending. An archive is the dotted prefix
    of a token (``cs`` for ``cs.AI``; the token itself when it has no dot);
    a subcategory is the full token.
    """
    archives: Counter[str] = Counter()
    for token, n in counts.items():
        archives[token.split(".", 1)[0]] += n
    return archives.most_common(), Counter(counts).most_common()


def print_category_summary(dataset_name: str, sample_batches: int, limit: int = 40) -> None:
    """Streams a bounded sample and prints the category vocabulary with counts."""
    counts = collect_category_counts(_iter_raw_batches(dataset_name), sample_batches)
    archives, subcategories = summarize_categories(counts)
    print(f"category vocabulary from a {sample_batches}-batch sample of {dataset_name}:")
    print(f"\narchives (use with --category), {len(archives)} found:")
    for name, n in archives:
        print(f"  {name:<20} {n:>10,} papers")
    print(f"\ntop {limit} subcategories (use with --subcategory):")
    for name, n in subcategories[:limit]:
        print(f"  {name:<24} {n:>10,}")
    if len(subcategories) > limit:
        print(f"  ... and {len(subcategories) - limit} more")


def _load_category_vocab() -> tuple[list[tuple[str, int]], list[tuple[str, int]]] | None:
    """Samples the dataset for the category vocabulary; ``None`` when offline."""
    print("  loading the category list from a sample of the dataset...")
    try:
        counts = collect_category_counts(_iter_raw_batches(Settings().dataset_name))
    except Exception as exc:
        print(f"  warning: could not load the category list ({exc}); type names manually.")
        return None
    return summarize_categories(counts)


def _pick_category_value(
    kind: str,
    entries: list[tuple[str, int]],
    examples: str,
    limit: int = 40,
) -> str | None:
    """Numbered picker over a vocabulary list; returns the chosen name or ``None``."""
    if entries:
        print(f"  available {kind}s from the sample ({min(len(entries), limit)} shown):")
        for i, (name, n) in enumerate(entries[:limit], start=1):
            print(f"    {i:>3}. {name:<20} {n:>10,}")
    raw = input(
        f"  pick a number, type a {kind} name (e.g. {examples}), or Enter to cancel: "
    ).strip()
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(entries):
            return entries[index - 1][0]
        print(f"  invalid number {raw!r}")
        return None
    return raw or None


# ── Batch cleaning (pure, unit-testable) ────────────────────────────────────
def category_pattern(corpus_filter: str | None) -> re.Pattern[str] | None:
    """Compiles the ``CORPUS_FILTER`` regex or returns ``None`` to keep all.

    ``re.search`` matches the pattern against each whitespace-separated category
    token. The function keeps the paper when any category matches.
    """
    if not corpus_filter or not corpus_filter.strip():
        return None
    return re.compile(corpus_filter.strip())


def clean_batch(df: pl.DataFrame, pattern: re.Pattern[str] | None) -> pl.DataFrame:
    """Reduces a raw batch to the snapshot fields and drops unusable rows.

    Drops rows whose ``title`` or ``abstract`` is null, empty, or whitespace-only.
    When ``pattern`` is not ``None``, also drops rows where no category token
    matches the ``CORPUS_FILTER`` regex. Drops null categories only when a filter
    is active; a null category does not match the pattern.
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
    """Streams the HF mirror once and yields raw polars batches.

    ``load_dataset`` selects columns at load time through the ``columns=``
    argument. Only the snapshot fields cross the network.
    ``iter(batch_size=...)`` advances a single stream sequentially. Unlike
    per-index ``shard()`` calls, it never re-reads earlier shards.
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
    """Cleans each batch and stages it as parquet in a single network pass.

    Returns ``(batches_written, rows_written)``. ``max_batches`` bounds the
    pass for smoke runs. ``None`` means the full corpus.
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
    """Merges staged batches into the canonical snapshot. Returns the row count.

    Lazily scans all staged parquets. Deduplicates on ``id``; cross-shard
    duplicates exist in the HF mirror. Sorts by ``id`` and assigns the dense
    integer ``row_index``. Sinks ``snapshot.parquet`` through the polars
    streaming engine, out-of-core. ``ids.npy`` is the dense index array.
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


# ── Scope prompt ──────────────────────────────────────────────────────────────
def _prompt_scope() -> tuple[str, str | None]:
    """Interactively asks which part of the corpus to ingest.

    Returns ``(label, corpus_filter)``. Only called when stdin is a TTY and no
    scope flag or ``CORPUS_FILTER`` is set. Choosing category/subcategory loads
    the vocabulary from a bounded sample of the dataset and offers a numbered
    picker; invalid input re-prompts.
    """
    print("Ingest scope:")
    print("  1. Full dataset (all papers)")
    print("  2. A category (one archive, e.g. cs, math, stat)")
    print("  3. A subcategory (one arXiv id, e.g. cs.AI, stat.ML)")
    while True:
        raw = input("Choose 1, 2, or 3 [1]: ").strip()
        choice = raw or "1"
        if choice == "1":
            return "full dataset", None
        if choice not in ("2", "3"):
            print(f"  invalid input {raw!r}; try again.")
            continue
        kind = "category" if choice == "2" else "subcategory"
        vocab = _load_category_vocab()
        entries = (vocab[0] if vocab else []) if choice == "2" else (vocab[1] if vocab else [])
        value = _pick_category_value(kind, entries, "cs" if choice == "2" else "cs.AI")
        if not value:
            print("  cancelled; choose again.")
            continue
        if choice == "3" and "." not in value:
            print(
                f"  invalid subcategory {value!r}: expected 'archive.sub' "
                "(e.g. cs.AI); choose again."
            )
            continue
        known = [name for name, _ in entries]
        if known and value not in known:
            reply = input(
                f"  {value!r} was not seen in the sample; "
                "the filter may match 0 papers. Continue? [y/N]: "
            ).strip()
            if reply.lower() not in ("y", "yes"):
                print("  cancelled; choose again.")
                continue
        if choice == "2":
            return f"category {value}", scope_pattern(SCOPE_CATEGORY, category=value)
        return f"subcategory {value}", scope_pattern(SCOPE_SUBCATEGORY, subcategory=value)


# ── Entrypoint ───────────────────────────────────────────────────────────────
def run(
    dataset_name: str | None = None,
    corpus_filter: str | None = None,
    staging_dir: Path | None = None,
    snapshot_path: Path | None = None,
    ids_path: Path | None = None,
    max_batches: int | None = None,
) -> int:
    """Runs the full ingest pipeline. Returns the snapshot row count.

    All paths default to ``src.config.Paths`` and ``Settings``. ``corpus_filter``
    falls back to the ``CORPUS_FILTER`` environment variable. ``None`` keeps all
    rows; pass ``""`` to keep all rows while *bypassing* the environment filter.
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
        description="Build the arXiv snapshot (docs/architecture.md §4.1).",
        epilog=(
            "Scope selection: pass --full, --category, or --subcategory to run "
            "non-interactively. With no scope flag and no CORPUS_FILTER set, the "
            "command prompts interactively for one. Use --list-categories to see "
            "which categories actually exist in the dataset."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--full",
        action="store_true",
        help="ingest the full dataset (default when --category/--subcategory are absent)",
    )
    action.add_argument(
        "--category",
        metavar="NAME",
        help="ingest one archive, e.g. 'cs' keeps every cs.* subcategory",
    )
    action.add_argument(
        "--subcategory",
        metavar="NAME",
        help="ingest one arXiv category, e.g. 'cs.AI' keeps only cs.AI papers",
    )
    action.add_argument(
        "--list-categories",
        action="store_true",
        help="stream a bounded sample and print the available archives/subcategories, then exit",
    )
    parser.add_argument(
        "--sample-batches",
        type=int,
        default=SAMPLE_BATCHES,
        help="batches to sample for --list-categories (default: %(default)s)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after this many batches (smoke runs); default: full corpus",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list_categories:
        if args.sample_batches <= 0:
            parser.error("--sample-batches requires a positive value")
        print_category_summary(Settings().dataset_name, args.sample_batches)
        return

    try:
        if args.category:
            label = f"category {args.category}"
            corpus_filter = scope_pattern(SCOPE_CATEGORY, category=args.category)
        elif args.subcategory:
            label = f"subcategory {args.subcategory}"
            corpus_filter = scope_pattern(SCOPE_SUBCATEGORY, subcategory=args.subcategory)
        elif args.full:
            label, corpus_filter = "full dataset", None
        else:
            env_filter = Settings().corpus_filter
            if env_filter:
                label, corpus_filter = "CORPUS_FILTER env", env_filter
            elif sys.stdin.isatty():
                label, corpus_filter = _prompt_scope()
            else:
                parser.error(
                    "no scope selected and stdin is not a terminal; pass "
                    "--full, --category NAME, or --subcategory NAME"
                )
    except ValueError as exc:
        parser.error(str(exc))

    if corpus_filter:
        print(f"ingest scope: {label} (category filter {corpus_filter!r})")
    else:
        print(f"ingest scope: {label}")
    # ``run`` falls back to CORPUS_FILTER only when corpus_filter is None, so
    # an explicit full-dataset choice is passed as "".
    n = run(corpus_filter=corpus_filter or "", max_batches=args.max_batches)
    print(f"ingest complete: {n} rows in snapshot")


if __name__ == "__main__":
    main()
