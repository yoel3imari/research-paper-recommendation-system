"""Recommendation quality evaluation (docs/architecture.md §7).

The dataset has no citation graph. The evaluation uses two defensible
proxies:

1. **Category-purity@k** — for a deterministic sample of seed papers, measure
   what fraction of the top-``k`` recommendations share the seed's primary
   sub-category (``cs.AI``/``cs.LG``/``cs.CL``). Compare against a random
   baseline (the corpus base rate for that category). Purity above baseline
   indicates semantic signal.

2. **Hand-judged query set** — a fixed list of curated research-interest
   queries; the top-5 hits per query go to ``reports/evaluation.md`` with
   empty ``relevance (0-3)`` cells for a human pass.

Import-safe: no heavy work at import time; run with ``python -m src.evaluate``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from .config import Paths
from .index import FaissVectorIndex, load_index

logger = logging.getLogger(__name__)

DEFAULT_N_SEEDS = 200
DEFAULT_K = 10
DEFAULT_SEED = 7

HAND_JUDGED_QUERIES: list[str] = [
    "retrieval augmented generation for code",
    "efficient transformer inference",
    "graph neural networks for molecular property prediction",
    "reinforcement learning from human feedback",
    "vision transformers for medical imaging",
    "diffusion models for text generation",
    "federated learning privacy",
    "large language model evaluation benchmarks",
    "neural architecture search",
    "multi-agent language models",
]


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────
def primary_cs_category(categories: Sequence[str] | str | None) -> str | None:
    """First category token starting with ``cs.`` (or None).

    ``categories`` is a sequence of tokens or a whitespace-separated string.
    The snapshot stores the category field as a space-separated string.
    """
    tokens = categories.split() if isinstance(categories, str) else categories or ()
    for token in tokens:
        if str(token).startswith("cs."):
            return str(token)
    return None


def category_base_rates(df: pl.DataFrame) -> dict[str, float]:
    """Fraction of the corpus whose primary ``cs.`` token is each category."""
    primary = df["categories"].map_elements(primary_cs_category, return_dtype=pl.Utf8)
    counts = primary.value_counts()
    total = df.height
    rates = {
        str(row["categories"]): row["count"] / total
        for row in counts.to_dicts()
        if row["categories"] is not None
    }
    return rates


def category_purity(
    recs: Sequence[tuple[float, str | None]], seed_cat: str | None, k: int
) -> float:
    """Fraction of the top-``k`` recommendation categories equal to ``seed_cat``.

    ``recs`` is a sequence of ``(score, category)`` tuples, best first. If the
    seed has no primary ``cs.`` category, the purity is 0 for that seed.
    """
    if not seed_cat:
        return 0.0
    top = recs[:k]
    if not top:
        return 0.0
    hits = sum(1 for _, cat in top if cat == seed_cat)
    return hits / len(top)


def random_baseline(rates: dict[str, float], seed_cat: str) -> float:
    """Expected purity under random selection: the corpus base rate."""
    return rates.get(seed_cat, 0.0)


# ── Snapshot helpers ────────────────────────────────────────────────────────
def load_snapshot_categories(snapshot_path: Path | None = None) -> pl.DataFrame:
    """Read ``id`` + ``categories`` columns from the snapshot."""
    snapshot_path = snapshot_path or Paths.SNAPSHOT
    return pl.read_parquet(snapshot_path, columns=["id", "categories"])


# ── Evaluation ──────────────────────────────────────────────────────────────
def run_evaluation(
    n_seeds: int = DEFAULT_N_SEEDS,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    index_path: Path | None = None,
    snapshot_path: Path | None = None,
) -> dict:
    """Compute category-purity@k against the random baseline; return a summary dict."""
    index_path = index_path or Paths.INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"index not found at {index_path}; run src.index first")

    df = load_snapshot_categories(snapshot_path)
    ids = df["id"].to_list()
    categories = df["categories"].to_list()
    n = len(ids)

    vectors = _load_vectors_for_eval()
    index = FaissVectorIndex(load_index(index_path), np.arange(n, dtype=np.int64))

    rng = np.random.default_rng(seed)
    seed_rows = rng.choice(n, size=min(n_seeds, n), replace=False)

    rates = category_base_rates(df)
    purities: dict[int, list[float]] = {1: [], 5: [], 10: []}
    baseline_purities: dict[int, list[float]] = {1: [], 5: [], 10: []}

    for row in seed_rows:
        seed_cat = primary_cs_category(categories[row])
        if seed_cat is None:
            continue
        _, labels = index.search(vectors[row : row + 1], k)
        rec_cats: list[tuple[float, str | None]] = [
            (1.0, primary_cs_category(categories[int(label)]))
            for label in labels[0]
            if 0 <= int(label) < n
        ]
        for kk in (1, 5, 10):
            purities[kk].append(category_purity(rec_cats, seed_cat, kk))
            baseline_purities[kk].append(random_baseline(rates, seed_cat))

    summary = {
        "n_seeds": len(purities[10]),
        "k": k,
        "purity": {kk: round(float(np.mean(v)), 4) for kk, v in purities.items() if v},
        "baseline": {
            kk: round(float(np.mean(v)), 4)
            for kk, v in baseline_purities.items()
            if v
        },
    }
    return summary


def _load_vectors_for_eval() -> np.ndarray:
    """Concatenate embedding chunks into one ``(N, d)`` matrix (row-aligned)."""
    from .bench_index import _load_all_vectors

    return _load_all_vectors()


# ── Report writing ──────────────────────────────────────────────────────────
def _render_eval_markdown(summary: dict, query_hits: list[tuple[str, list[dict]]]) -> str:
    lines = [
        "# Evaluation",
        "",
        "Proxies used (no citation graph in this dataset — architecture §7): "
        "category-purity@k against a random baseline, plus a hand-judged query set.",
        "",
        "## Category-purity@k",
        "",
        f"Seed papers sampled: {summary['n_seeds']}. Purity above the random",
        "baseline indicates the embeddings capture topical structure.",
        "",
        "| k | Purity@k | Random baseline | Lift |",
        "|---|---|---|---|",
    ]
    for kk in sorted(summary["purity"]):
        purity = summary["purity"][kk]
        baseline = summary["baseline"][kk]
        lift = purity - baseline
        lines.append(f"| {kk} | {purity:.4f} | {baseline:.4f} | {lift:+.4f} |")
    lines += ["", "## Hand-judged queries (relevance 0–3)", ""]
    for query, hits in query_hits:
        lines.append(f"### {query}")
        lines.append("")
        lines.append("| # | arXiv id | Title | Score | Relevance (0–3) |")
        lines.append("|---|---|---|---|---|")
        for i, hit in enumerate(hits, start=1):
            lines.append(
                f"| {i} | `{hit['id']}` | {hit['title'][:80]} | {hit['score']:.3f} |  |"
            )
        lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    summary: dict,
    query_hits: list[tuple[str, list[dict]]],
    report_path: Path | None = None,
) -> Path:
    """Write the evaluation markdown and return its path (unit-testable)."""
    report_path = report_path or Paths.EVALUATION
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_eval_markdown(summary, query_hits))
    return report_path


def run_hand_judged(snapshot_path: Path | None = None) -> list[tuple[str, list[dict]]]:
    """Top-5 hits per curated query; returns ``(query, [hit dicts])``."""
    from .api.app import RecommendationService

    service = RecommendationService(snapshot_path=snapshot_path or Paths.SNAPSHOT)
    hits: list[tuple[str, list[dict]]] = []
    for query in HAND_JUDGED_QUERIES:
        scores, labels = service.search_query(query, 5)
        query_hits = []
        for score, label in zip(scores[0], labels[0], strict=True):
            if int(label) < 0 or int(label) >= service.n_papers:
                continue
            meta = service.metadata[int(label)]
            query_hits.append(
                {"id": meta["id"], "title": meta["title"], "score": float(score)}
            )
        hits.append((query, query_hits))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_evaluation(n_seeds=args.seeds, k=args.k)
    logger.info("purity: %s baseline: %s", summary["purity"], summary["baseline"])
    query_hits = run_hand_judged()
    path = write_evaluation_report(summary, query_hits, args.out)
    logger.info("evaluation report written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
