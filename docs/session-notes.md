# Session Notes — arXiv Paper Recommendation

Living knowledge base for this project. Read at the start of every session to
re-orient: what's done, what's next, and — critically — what this machine can
and cannot do. Append new entries to the Session Log.

## 1. Project State

| Phase (architecture §11) | Status | Notes |
|---|---|---|
| 1. Scaffold (uv, pyproject, config, .gitignore) | ✅ Done | commit `0a62f75` |
| 2. Ingest + tests (`src/ingest.py`) | ✅ Done | commit `…` — streaming HF, sort-by-id snapshot; smoke-validated live |
| 3. Embed pipeline + tests | ⏭ Next | ONNX + manifest/resume |
| 4. Index + bench + tests | ⛔ | IVF8192,SQ8 + benchmark report |
| 5. FastAPI + tests | ⛔ | 5 endpoints, concurrency |
| 6. Streamlit dashboard | ⛔ | query mode + seed paper mode |
| 7. Eval + README + Docker | ⛔ | reports + docs + compose |

Full plan: `docs/architecture.md`. Code is Python 3.12 / uv, managed by `uv`.

## 2. Environment Constraints — READ BEFORE ADDING ANY DEPENDENCY

**Dev machine: Intel MacBook Pro 15,4 — macOS 12, x86_64, 8GB RAM.**

This is the #1 source of pain. Modern wheels have largely dropped Intel Macs:

| Package | Pinned version | Why |
|---|---|---|
| `faiss-cpu` | `==1.10.0` | 1.12+ ships only `macosx_13_0_x86_64` wheels (needs macOS 13) |
| `onnxruntime` | `==1.19.2` | 1.20+ has no x86_64 macOS wheel at all |
| `torch` | `==2.2.2` | 2.3+ has no x86_64 macOS wheel at all |
| `numpy` | `<2` | torch 2.2.2 is ABI-incompatible with numpy 2 |

`pyproject.toml` has a `[tool.uv] required-environments` entry constraining
resolution to this platform — without it, uv's universal lockfile silently
falls back to **source builds** (which fail: swig, missing headers). Comment in
`pyproject.toml` explains the why. Do not remove it.

**Rule for any new dependency**: check wheel availability first:

```bash
uv pip install --dry-run "pkg==VERSION"
```

If it errors with "doesn't have a source distribution or wheel for the current
platform", pick an older version or drop it. Do not `uv add` blindly.

**Memory**: 8GB total. Never run ingest + embed + index concurrently (~2GB peak
each). Close browser/editors during the full-corpus embed build (~5-6GB total).

## 3. Tooling & Commands

```bash
uv add "pkg==ver"            # add dependency (pin versions for ML packages!)
uv add --dev pytest httpx ruff
uv sync                      # install from lockfile
uv run python -c "..."       # run in project env
uv run ruff check src/       # lint (select: E,F,I,UP,B,SIM)
uv run ruff check --fix src/ # auto-fix
uv run pytest                # tests (testpaths = ["tests"])
```

Import smoke test after any dependency change:

```bash
uv run python -c "import numpy, torch, faiss, onnxruntime, transformers, optimum, tokenizers, datasets, polars, streamlit, fastapi, pydantic_settings; print('ok')"
```

Pytest quirks:
- `pytest` exits with code **5** when 0 tests are collected — breaks `&&` chains before tests exist.
- First `torch` import is slow (~30s) on this machine; don't mistake it for a hang.

## 4. Portfolio Conventions (match these)

Sibling projects in `/Users/yusef/projects/portfolio/` (`customer-churn-prediction` is the reference).

- `src/config.py` — class-constant style (`Paths.ROOT`, `IndexParams.NLIST`), box-drawing section separators (`# ── Paths ──`), module docstring pointing to the architecture doc.
- Config: deploy-time knobs via env vars; `pydantic-settings` `Settings(BaseSettings)`. Gotcha: init kwargs use **field names** (`index_path`), env vars match **case-insensitively** (`INDEX_PATH`).
- pytest: `[tool.pytest.ini_options] testpaths = ["tests"]`.
- ruff: `line-length = 100`, `target-version = "py312"`, select `E,F,I,UP,B,SIM`.
- FastAPI: `src/api/{app,router,schemas}.py`; Streamlit: single `dashboard/app.py`.
- `uv` + `uv.lock` everywhere.

## 5. Architecture Quick Reference

Full detail: `docs/architecture.md`.

- **Pipeline**: HF `open-index/open-arxiv` (streaming) → `snapshot.parquet` (sorted by id, +`ids.npy`) → chunked ONNX embed (`all-MiniLM-L6-v2`, dynamic UINT8, `model_quint8_avx2.onnx`) → FAISS `IVF8192,SQ8` (~1GB) → FastAPI (`/recommend/paper/{}`, `/recommend/query`, `/papers/search`) → Streamlit.
- **Key params** (in `src/config.py`): dim 384, chunk 100K rows, batch 64, max seq 256, nlist 8192, nprobe 64, train fraction 0.10, checkpoint every 500K.
- **Concurrency**: FAISS search is thread-safe; ONNX encode needs a `threading.Lock`; plain `def` endpoints (threadpool); cap OpenMP threads (`OMP_NUM_THREADS=2-4`).
- **Deferred**: specter2 A/B, BM25+RRF, vector-DB swap — all documented as Phase 3 in architecture §12.

## 6. Session Log

### 2026-08-02 — Scaffold + platform discovery
- Committed scaffold (`0a62f75`): project layout, deps, `src/config.py`, `.gitignore`, `uv.lock`.
- Discovered and pinned the Intel-Mac wheel constraints (§2). `faiss-cpu==1.10.0` is the last version that installs on macOS 12 x86_64; `onnxruntime==1.19.2`; `torch==2.2.2` + `numpy<2`.
- Added `required-environments` to `[tool.uv]` to stop universal-lockfile source-build fallbacks.
- Verified all imports, ruff clean, config env-override works (`INDEX_PATH`).
- Removed hello-world `main.py` stub.

### 2026-08-02 — Phase 2: ingest
- `src/ingest.py`: two-pass design — (1) single network pass via `ds.iter(batch_size=8192)` cleaning + staging each batch to `data/raw/staging/`, (2) out-of-core polars `scan_parquet → unique(id) → sort(id) → sink_parquet` snapshot. Columns selected at load time via `load_dataset(..., columns=...)`.
- **Why not `ds.shard(num_shards, index=i)` per shard**: each call re-reads the stream from the start (O(N²) network). `iter(batch_size=...)` advances one stream sequentially.
- Snapshot columns: `row_index` (dense int64), `id`, `title`, `abstract`, `categories`, `authors`, `update_date`. `ids.npy` = `np.arange(N, int64)` — the alignment key (row i of every artifact ↔ snapshot row ids[i]).
- `CORPUS_FILTER` env var: regex matched per whitespace-separated category token (a paper is kept if any token matches). Confirmed live: categories are space-separated multi-token strings (e.g. `acc-phys physics.acc-ph`).
- Tests (18): `tests/conftest.py` `synthetic_papers` fixture; `tests/test_ingest.py` classes `TestCleanBatch`, `TestCategoryPattern`, `TestCategoryFilter`, `TestMergeSnapshot`.
- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` (needed for `from src.ingest import ...`).
- Polars 1.43 gotchas: no `str.split_whitespace()` (use `str.split(by=" ", literal=True)`); `sink_parquet` does NOT create parent dirs (`mkdir` first).
- **Live smoke test passed** (`python -m src.ingest --max-batches 10`): 81,920 rows in ~3min, peak RSS ~796MB — HF streaming works on this machine (the Phase-2 risk is retired). Artifacts cleaned up afterward; full run is `uv run python -m src.ingest`.
- **Next**: Phase 3 embed — suggested 15-min ONNX smoke test first (download `model_quint8_avx2.onnx`, encode ~1K papers, measure papers/sec) to validate the architecture's 300-950 paps/sec claim before the ~2h full-corpus embed.
