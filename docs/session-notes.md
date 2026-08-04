# Session Notes — arXiv Paper Recommendation

Living knowledge base for this project. Read at the start of every session to
re-orient: what's done, what's next, and — critically — what this machine can
and cannot do. Append new entries to the Session Log.

## 1. Project State

| Phase (architecture §11) | Status | Notes |
|---|---|---|
| 1. Scaffold (uv, pyproject, config, .gitignore) | ✅ Done | commit `0a62f75` |
| 2. Ingest + tests (`src/ingest.py`) | ✅ Done | commit `a48622a` — streaming HF, sort-by-id snapshot; smoke-validated live |
| 3. Embed pipeline + tests | 🚧 Coded, **not run live** | commit `f5d8f34` — offline-only (mobile data); ONNX smoke test deferred to WiFi |
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

### 2026-08-02 — Phase 3: embed (coded offline, mobile-data session)
- `src/embed.py`: `Encoder` protocol + `OptimumEncoder` (dynamic UINT8 ONNX). `run()` iterates the snapshot in fixed `CHUNK_SIZE` rows, encodes `title + " " + abstract`, L2-normalizes (cosine via inner product), writes `data/embeddings/chunk_{n:05d}.npy` + a `manifest.json` entry `{n: {start, end, rows, shape}}`. Resume semantics: skip chunks whose manifest records match the actual `rows`/`shape` (torn/truncated `.npy` → re-encode); atomic manifest writes via tmp+rename.
- **Offline by design**: heavy `optimum`/`transformers` imports live inside `OptimumEncoder.__init__` (instantiation), never at module import. Tests use a `StubEncoder` — the real model is never constructed, so no download, no network. `OptimumEncoder.main()`/CLI is what runs the real job (deferred).
- Tests (17): `TestL2Normalize`, `TestBuildTexts`, `TestIterChunks`, `TestEncodeChunk`, `TestManifest`, `TestChunkIsComplete`, `TestWriteChunk`, `TestRun`, `TestRunResume`. All offline, no model.
- **Deferred to WiFi (this is the next action when off mobile data)**: 15-min ONNX smoke test — download `model_quint8_avx2.onnx` + AutoTokenizer, encode ~1K papers, measure papers/sec — before the ~2h full-corpus embed (`uv run python -m src.embed`).
- Full suite now 35 tests green (18 ingest + 17 embed), ruff clean — run with `HF_HUB_OFFLINE=1` as a guard against accidental network.

### Session context (mobile data — DO NOT network)
- This entire phase ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. No downloads, no smoke test, no full run.
- LSP shows "polars/optimum/transformers cannot resolve" in `src/embed.py` and tests — venv-path artifact of the language server only; all verified importable via `uv run`.

### 2026-08-03 — Scope change: demo corpus (CS only)
- **Decision**: the demo uses **computer science papers only** — not the full ~2.4M-row open-arxiv corpus. This is a portfolio demo; the smaller corpus makes every downstream phase (embed, index, API, dashboard) faster and the artifacts smaller.
- Mechanism: the existing `CORPUS_FILTER` env var (include-regex on category tokens). CS-only pattern: `CORPUS_FILTER='^cs\.'` (matches categories like `cs.AI`, `cs.LG`; a paper is kept if any of its category tokens match).
- Important: ingest still **streams the entire mirror once** (~90+ min) — the filter drops rows, it does not skip shards. Expected snapshot size: roughly 300–500K rows.
- Run: `CORPUS_FILTER='^cs\.' uv run python -m src.ingest`
- Network is back (no more `HF_HUB_OFFLINE=1`). Next after ingest: ONNX model download + throughput smoke test, then full embed.

### 2026-08-03 — Dropped `optimum` for the runtime embed path ⚠️
- **Blocker + fix**: `optimum[onnxruntime]` does not import on this machine. `optimum-onnx 0.1.0`'s `modeling.py` does `from optimum.exporters.onnx import main_export` at top level, but `optimum 2.1.0` only exposes `main_export` in the `__main__` submodule — a genuine upstream version mismatch. Fixing `optimum`'s version doesn't help (only 2.1.0 exists; optimum-onnx is capped). Additionally, importing `optimum.exporters.onnx`'s lazy chain drags in `onnx 1.19.0` which needs `ml_dtypes.float4_e2m1fn` (not present in the pinned `ml-dtypes 0.4.1`).
- **Fix**: replaced `OptimumEncoder` with `OnnxEncoder` in `src/embed.py` — a bare `onnxruntime.InferenceSession` (`CPUExecutionProvider`) + `transformers.AutoTokenizer`. The BERT ONNX graph already outputs mean-pooled `last_hidden_state`, so no exporter machinery is needed. Removed `optimum`/`optimum-onnx`/`onnx`/`ml-dtypes` (4 pkgs). Runtime route is functionally identical, 17 embed tests still pass, ruff clean.
- Smoke bench on this machine (realistic ~1000-char abstracts, 256-token truncation): **~17–20 pap/sec** (batch 64/128/256; model load 3.9s). **This contradicts the architecture's 300–950 pap/sec claim by ~10–20×.** Implication: a 300–500K-row CS snapshot will take roughly **4–7h to embed**, not ~2h. (Verify once the real snapshot exists; the number may drift — and long-format throughput scales with tokens, so exact abstract lengths matter.)
- Model files downloaded to `models/all-MiniLM-L6-v2/` (config.json, tokenizer + vocab, `model_quint8_avx2.onnx` @23MB). `optimum` no longer needed at runtime; remove any remaining mention from docs.

### 2026-08-03 — Corpus sized to cs.AI/LG/CL (smaller demo)
- **After further scoping**: full-CS was still ~408K rows (cs.LG + cs.AI are enormous) — too big for a quick demo embed (~5-6h at ~20 pap/sec). Tightened the filter to **cs.AI/LG/CL** via `CORPUS_FILTER='^cs\.(AI|LG|CL)$'`. Verified on staged batches: zero leakage (every kept row has ≥1 target token; cross-listed categories like `q-bio` are expected metadata, not leakage).
- **Ingest ran to completion**: `data/processed/snapshot.parquet` = **408,067 rows** (2007-05-23 → 2026-03-20), sorted by id, `ids.npy` contiguous, 0 null title/abstract. (cs.AI+LG+CL is genuinely ~400K — the demos-cope estimate was too low; cs.LG alone is huge.)
- **Phase 4 built**: `src/index.py` (FAISS build/search per §4.3: IVF-SQ8 default + HNSW profile, memory-bounded train-saples, `faiss.write_index` checkpoints every `CHECKPOINT_EVERY`, `index_meta.json`) + `tests/test_index.py` (15 tests). ruff + pytest green. Generated by a delegated `deep` agent; it stalled in a thinking loop before reporting, but the written files verify correct. NOT yet committed by me — see below.
- **Phase 3 embed kicked off live** (network available): `OMP_NUM_THREADS=4 nohup uv run python -m src.embed` → 408,067 rows in 5 chunks of 100K (last ~8K). Predicted ~5-6h at ~20 pap/sec; first chunk writes to `data/embeddings/` after ~50min.
- Cleaned `data/raw/staging` after ingest (189M freed).

### 2026-08-03 — Phase 4 committed + Phase 5 (FastAPI) coded, offline-green
- **Phase 4 committed** (`8ef67f4`): `src/index.py` (`build_index`, `load_index`, `FaissVectorIndex` with protocol `VectorIndex`, `make_index`, `clamp_nlist`, mmap-bounded `_train_sample`) + `tests/test_index.py`. THE ROUTE-ORDERING BUG: 3 of 11 API tests initially failed (404 on `/papers/search`) because FastAPI matches routes in registration order and `/papers/{arxiv_id}` was declared BEFORE `/papers/search`, so the path-param swallowed the literal path. **Fix**: reordered the decorators — `/papers/search` declared first. `# NOTE` comment documents WHY the order matters (prevents a future refactor from re-breaking it).
- **Phase 5 committed** (`ae593c2`): `src/api/{app,router,schemas}.py`. `RecommendationService` owns snapshot polars df + `FaissVectorIndex` (id-aligned) + `OnnxEncoder`; constructor reads real artifacts, endpoints are plain `def` (threadpool), FAISS search lock-free + encode guarded by one `threading.Lock`. `create_app()` factory + `@lru_cache`-wrapped `cached_service` for `--reload` reuse. Five GETs: `/health`, `/papers/search` (substring title picker), `/papers/{arxiv_id}`, `/recommend/query`, `/recommend/paper/{arxiv_id}`. `src/api/router.py` uses a `_service` sentinel default to sidestep ruff B008 (request-scope param annotation) — mirrors the reference `customer-churn-prediction` convention.
- Test philosophy (per architecture §8): offline unit tests have NO real model/index — `StubEncoder` + a hand-built `FakeVectorIndex` in `tests/conftest.py`; `TestClient` injects the router under a fixture instead of using the global `app` in module scope. `tests/test_api.py`: 11 tests (health, paper lookup 200/404, search matched/no-match/limit, recommend query with filters, recommend-from-seed, error for missing seed). Full suite **61 green** + ruff clean.
- Smoke script staged at repo root: `smoke_api.py` — gets the heap height dynamically from the snapshot, runs real `create_app()` against real artifacts once the index exists, asserting index-n == snapshot-height, seeded and query recommendations return `k` with descending scores, and returned ids resolve. Ruff-clean. NOT run yet (waits on the index).

### 2026-08-03 — embedding in progress (slow but healthy)
- Embed PID 8114, `nohup`ed, `OMP_NUM_THREADS=4`: 408,067 rows / chunks-of-100K, batch 64, dim 384. Monitored since kickoff: ~2h27m elapsed, ~124 CPU-min across ~3.5 cores, RSS ~855MB stable (no thrash). **Still on chunk 0** — `data/embeddings/` is empty because a chunk artifact only writes after the ENTIRE 100K-row chunk completes (no intermediate output).
- **Throughput reality check**: consistently **lower than the ~20 pap/s smoke bench** — at 124 CPU-min with no 100K write, the first chunk is costing well over the ~83 CPU-min the smoke forecast implied. It is NOT failing (cores busy, RSS flat), just slower. Projected full run (5 chunks): likely **8–10h wall on this Intel 8GB box**, not the 5–6h originally estimated. Per user decision ("don't worry about the speed — just want it to work") I am NOT killing/retuning it.
- **Unattended handoff**: `/tmp/watch_build.sh` watcher PID 9699 polls every 60s; on embed exit it (1) verifies `manifest.json` sum(rows) == snapshot height (408,067 rows) to avoid building from a truncated/crashed embed, (2) runs `from src.index import build_index; build_index()` logging to `/tmp/build_index.log`, writing `data/index/arxiv.index` + `index_meta.json`. Detached via `nohup`, safe to rerun.
- **Next (once the index exists)**: `uv run python smoke_api.py` → expect PASS; then full `uv run pytest -q` + `uv run ruff check`; record measured figures; final commit.

### 2026-08-04 — MVP push: demo corpus + phases 6–7 (parallel)
- **State on arrival**: the 408K-embed job from 08-03 died silently (empty `/tmp/embed_cs.log`, no PID, no chunks in `data/embeddings/`, watcher still waiting on dead PID 8114). Snapshot (408,067 rows) intact. Per user directive ("make it work, don't optimize"), pivoted to the fast demo path: `run_demo.py --rows 30000` (subset → embed → index). **Full 408K snapshot + ids backed up** to `data/processed/full/` before the subset overwrote `data/processed/snapshot.parquet`.
- **Demo corpus**: 30,000 rows (ids sorted; includes seed `1706.03762` at row 25610). Embed running with live progress (`encoder.progress = True` flag added to `src/embed.py`, uncommitted), batch 128, `OMP_NUM_THREADS=4`. Measured ~6-11 pap/sec on this box → ~50-90 min wall for 30K.
- **Phase 6 (Streamlit dashboard)**: `dashboard/app.py` — two modes (interest query + seed paper picker), `@st.cache_data` API helpers, result cards (title link, similarity badge, category chips, authors, abstract expander, arXiv link), sidebar k/year/category filters. Ruff-clean. One real bug fixed: `categories` was passed as a `list[str]` to the API which expects a comma-separated string — joined with `",".join(...)`.
- **Phase 7 (eval + bench + Docker + README)**: `src/bench_index.py` (recall@10 vs exhaustive `IndexFlatIP`, p95/mean latency, index size) + `src/evaluate.py` (category-purity@k vs corpus base-rate baseline + 10 hand-judged queries → `reports/evaluation.md`) + `tests/test_eval_bench.py` (20 offline tests). `Dockerfile` (multi-stage, `--no-environments` to bypass the darwin-pinned `required-environments`, non-root) + `docker-compose.yml` (app + dashboard, healthchecks, 4GB-VM-cap gotcha documented) + `.dockerignore`. README rewritten with measured-figures placeholders to fill post-bench.
- **Next**: wait for embed+index → `smoke_api.py` → run bench + eval → fill README figures → full pytest + ruff → final commit.
