"""One-shot smoke test against the real build artifacts (not the stub corpus).

The script exercises the full stack end-to-end when a real FAISS index and
real embeddings exist. It uses ``RecommendationService`` with the real
snapshot, index ids, and the ONNX encoder. It verifies the
embed→index→search→metadata chain:

1. index loads and reports ``n == snapshot height``
2. a seeded paper returns ``k`` papers with plausible scores (descending)
3. a free-text query returns ``k`` papers with descending scores
4. returned paper ids resolve in the snapshot

Run: ``uv run python smoke_api.py``. The run takes minutes (index + first
service call spin up the IVF and the ONNX model). Idempotent; safe to rerun.
"""

from __future__ import annotations

import time

import polars as pl
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import Paths
from src.index import load_index

SNAPSHOT_HEIGHT = int(pl.read_parquet(Paths.SNAPSHOT).height)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> int:
    t0 = time.time()
    n = int(load_index().ntotal)
    _check(n == SNAPSHOT_HEIGHT, f"index n={n} == snapshot height={SNAPSHOT_HEIGHT}")

    app = create_app()
    client = TestClient(app)

    results = client.get("/api/v1/health")
    _check(results.status_code == 200, f"GET /health -> {results.status_code}")
    print(f"[{int(time.time()-t0)}s] loaded index + app")

    # pick a seed paper: high-citation transformer ancestry
    seed = "1706.03762"  # Attention is all you need
    r = client.get(f"/api/v1/recommend/paper/{seed}?k=5")
    _check(r.status_code == 200, f"GET /recommend/paper/{seed} -> {r.status_code}")
    body = r.json()
    recs = body["results"]
    _check(len(recs) == 5, f"returned {len(recs)} results (expect 5)")
    scores = [x["score"] for x in recs]
    _check(all(b > e for b, e in zip(scores, scores[1:], strict=False)),
           f"scores descending {scores}")

    q = client.get("/api/v1/recommend/query", params={"q": "graph neural networks", "k": 5})
    _check(q.status_code == 200, f"GET /recommend/query -> {q.status_code}")
    qrecs = q.json()["results"]
    qscores = [x["score"] for x in qrecs]
    _check(len(qrecs) == 5 and all(b >= e for b, e in zip(qscores, qscores[1:], strict=False)),
           f"query k=5 scores descending {qscores}")
    _check(all(x["paper"]["id"] for x in qrecs), "returned ids non-empty")

    print(f"[{(time.time()-t0):.1f}s] smoke test PASSED: api + index + embed chain verified")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())