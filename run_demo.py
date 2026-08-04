"""One-shot demo pipeline: subset snapshot -> embed -> FAISS index.

Run it once and walk away — everything is printed to stdout with live
progress, and the finished app is one command away.

    uv run python run_demo.py              # default 20K papers
    uv run python run_demo.py --rows 5000  # smaller/faster
    uv run python run_demo.py --rows 20000 --batch 256

It slices the FIRST ``--rows`` papers off the full snapshot (which is already
sorted by id) and writes a compact snapshot + matching ``ids.npy``, so the
embed chunks, the FAISS index, the API and ``smoke_api.py`` all stay
positionally aligned with zero manual wiring.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np
import polars as pl

from src.config import Paths
from src.embed import OnnxEncoder
from src.embed import run as embed_run
from src.index import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="number of papers to embed (default 20_000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="ONNX encode batch size (default 128)",
    )
    args = parser.parse_args()

    if args.rows <= 0:
        print(f"error: --rows must be positive, got {args.rows}", file=sys.stderr)
        return 1

    # 1/3 — subset the snapshot (full snapshot stays untouched on disk)
    full = pl.read_parquet(Paths.SNAPSHOT)
    total = full.height
    rows = min(args.rows, total)
    print(f"[1/3] full snapshot: {total} rows -> demo subset: {rows} rows")

    subset = full.head(rows)
    Paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    subset.write_parquet(Paths.SNAPSHOT)
    np.save(Paths.IDS, np.arange(rows, dtype=np.int64))
    print(f"[1/3] wrote demo snapshot ({rows} rows) + ids.npy to {Paths.DATA_PROCESSED}")

    # wipe stale chunks/index so this run is a clean build
    for directory in (Paths.DATA_EMBEDDINGS, Paths.DATA_INDEX):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    # 2/3 — embed with live progress
    t0 = time.time()
    print(f"[2/3] embedding {rows} rows (batch={args.batch})...")
    encoder = OnnxEncoder()
    encoder.progress = True
    summary = embed_run(
        encoder,
        snapshot_path=Paths.SNAPSHOT,
        embeddings_dir=Paths.DATA_EMBEDDINGS,
        manifest_path=Paths.MANIFEST,
        batch_size=args.batch,
    )
    elapsed = time.time() - t0
    print(f"[2/3] embed done: {summary} in {elapsed:.1f}s "
          f"({rows / elapsed:.1f} rows/s)")

    # 3/3 — build the FAISS index from the chunks
    t1 = time.time()
    print("[3/3] building FAISS index...")
    meta = build_index(
        embeddings_dir=Paths.DATA_EMBEDDINGS,
        manifest_path=Paths.MANIFEST,
        index_path=Paths.INDEX,
        meta_path=Paths.INDEX_META,
    )
    print(f"[3/3] index built: {meta} in {time.time() - t1:.1f}s")

    print()
    print("DONE. Run the API with:")
    print("  uv run uvicorn src.api.app:app --reload")
    print("or verify end-to-end with:")
    print("  uv run python smoke_api.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())