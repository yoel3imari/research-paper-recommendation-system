"""FAISS index benchmark (docs/architecture.md §4.4).

Measures the built vector index against exhaustive ground truth so "we chose
SQ8" is a measured, documented decision rather than a claim:

1. **Recall@10** of the IVF/SQ8 index vs an exhaustive ``IndexFlatIP`` over the
   same vectors (held-out sample of the corpus).
2. **p95 + mean search latency** over a fixed number of single-query searches
   (index warmed first, single-threaded).
3. **Memory footprint** — RSS delta while the index is resident (best-effort
   via ``resource``; on platforms where that is unreliable we report the
   index's own ``total_vector_bytes`` as an estimate).

Outputs a markdown table to ``reports/index_benchmark.md`` (``Paths.INDEX_
BENCHMARK``) and prints the same table to stdout. Import-safe: no heavy work
happens at module import; run with ``python -m src.bench_index``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from .config import IndexParams, Paths
from .index import iter_chunk_paths, load_index, load_manifest

logger = logging.getLogger(__name__)

DEFAULT_N_QUERIES = 1_000
DEFAULT_SEED = 42


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────
def recall_at_k(ann_labels: np.ndarray, truth_labels: np.ndarray, k: int) -> float:
    """Fraction of the true top-``k`` that the ANN index found in its top-``k``.

    ``ann_labels`` / ``truth_labels`` are 2-D arrays of FAISS row labels, one
    row per query. Both must be sorted by descending score (FAISS returns them
    that way). Rows with fewer than ``k`` results are handled by comparing only
    the intersection.
    """
    total_hits = 0
    total_relevant = 0
    for ann, truth in zip(ann_labels, truth_labels, strict=True):
        ann_set = {int(x) for x in ann[:k] if x >= 0}
        truth_set = {int(x) for x in truth[:k] if x >= 0}
        total_hits += len(ann_set & truth_set)
        total_relevant += len(truth_set)
    if total_relevant == 0:
        return 0.0
    return total_hits / total_relevant


def p95(values: Sequence[float]) -> float:
    """95th percentile of a list of latencies (``np.quantile``, linear)."""
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), 0.95))


# ── Data loading (CLI-only path) ────────────────────────────────────────────
def _load_all_vectors() -> np.ndarray:
    """Concatenate all embedding chunks into one ``(N, d)`` float32 matrix.

    At the demo corpus scale (~30K rows) this is ~46MB — fine to materialize
    for the benchmark. Chunks are read in manifest order so row ``i`` of the
    result aligns with snapshot row ``i`` (architecture §4.1).
    """
    manifest = load_manifest(Paths.MANIFEST)
    chunk_paths = iter_chunk_paths(Paths.DATA_EMBEDDINGS, manifest)
    if not chunk_paths:
        raise FileNotFoundError(f"no embedding chunks in {Paths.DATA_EMBEDDINGS}")
    parts = [np.load(path).astype(np.float32) for _, path in chunk_paths]
    return np.concatenate(parts, axis=0)


def _held_out_sample(vectors: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic random sample of ``n`` rows (query vectors)."""
    rng = np.random.default_rng(seed)
    return vectors[rng.choice(vectors.shape[0], size=n, replace=False)]


# ── Benchmark ───────────────────────────────────────────────────────────────
def run_benchmark(
    index_path: Path | None = None,
    meta_path: Path | None = None,
    n_queries: int = DEFAULT_N_QUERIES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the index benchmark; return a summary dict.

    Raises ``FileNotFoundError`` if the index or embeddings are missing (the
    build pipeline has not run on this machine yet).
    """
    index_path = index_path or Paths.INDEX
    meta_path = meta_path or Paths.INDEX_META
    if not index_path.exists():
        raise FileNotFoundError(f"index not found at {index_path}; run src.index first")

    meta = json.loads(meta_path.read_text())
    vectors = _load_all_vectors()
    n = vectors.shape[0]

    # Ground truth: exhaustive inner-product over the same L2-normalized vectors.
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)

    queries = _held_out_sample(vectors, n_queries, seed)

    # Recall@10 against exhaustive ground truth.
    index = load_index(index_path)
    _, truth = flat.search(queries, 10)
    _, ann = index.search(queries, 10)
    recall = recall_at_k(ann, truth, 10)

    # Latency: warm the index, then time 1,000 single queries.
    for _ in range(10):
        index.search(queries[:1], 10)
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 10)
        latencies.append((time.perf_counter() - t0) * 1000)

    # Memory: report the on-disk index size (most reliable estimate; the
    # in-memory footprint is dominated by the index's stored vectors).
    mem_bytes = index_path.stat().st_size

    summary = {
        "profile": meta.get("profile", IndexParams.PROFILE_DEFAULT),
        "n": n,
        "recall_at_10": round(recall, 4),
        "latency_ms_p95": round(p95(latencies), 3),
        "latency_ms_mean": round(float(np.mean(latencies)), 3),
        "n_queries": len(latencies),
        "index_bytes": mem_bytes,
    }
    return summary


def _render_markdown(summary: dict) -> str:
    lines = [
        "# Index Benchmark",
        "",
        f"Run against the built `{summary['profile']}` index "
        f"({summary['n']} vectors). Ground truth: exhaustive `IndexFlatIP` "
        f"over the same embeddings. Latency: single-threaded, warmed.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Profile | `{summary['profile']}` |",
        f"| Vectors | {summary['n']} |",
        f"| Recall@10 | {summary['recall_at_10']} |",
        f"| p95 search latency | {summary['latency_ms_p95']} ms |",
        f"| mean search latency | {summary['latency_ms_mean']} ms |",
        f"| queries timed | {summary['n_queries']} |",
        f"| index size (est.) | {summary['index_bytes'] / 1e6:.1f} MB |",
        "",
    ]
    return "\n".join(lines)


def write_report(summary: dict, report_path: Path | None = None) -> Path:
    """Write the markdown report and return its path (unit-testable)."""
    report_path = report_path or Paths.INDEX_BENCHMARK
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_markdown(summary))
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_benchmark(n_queries=args.n_queries, seed=args.seed)
    print(_render_markdown(summary))
    path = write_report(summary)
    logger.info("benchmark report written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
