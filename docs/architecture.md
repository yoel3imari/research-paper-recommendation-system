# Architecture — arXiv Paper Recommendation System

> **Status**: Proposed (pre-implementation)
> **Last updated**: 2026-08-02

---

## 1. Overview

A recommendation system for arXiv research papers. Given either a **seed paper** or a **free-text research interest**, the system returns ranked similar papers from the full arXiv corpus (~2.5M papers).

The system is designed for a portfolio-quality end-to-end demonstration: streaming data ingestion, a checkpointed batch embedding pipeline, a quantized vector index that fits an 8GB laptop, a production-style FastAPI service, a Streamlit demo UI, and measurable evaluation.

### Goals

| Goal | Priority |
|---|---|
| Item-to-item recommendations ("papers like paper X") | P0 |
| User-to-item recommendations ("papers matching this interest") | P0 |
| Full arXiv corpus (~2.5M papers), running on an 8GB laptop (MacBook Pro 15,4) | P0 |
| Reproducible, resumable pipeline (crash-safe batch jobs) | P0 |
| Measured quality (benchmark + evaluation, not vibes) | P1 |
| Production-grade serving (FastAPI + Docker + tests) | P1 |
| specter2 (science-trained) model A/B — **blocked at full corpus scale** | P2 (documented) |

### Non-Goals

- No full-text / PDF processing. We embed **title + abstract only** (standard practice; specter2-family models are trained on exactly this signal). Full text stays on arXiv — the app links out via `https://arxiv.org/abs/{id}`.
- No collaborative filtering. arXiv has no implicit/explicit user feedback signal in this dataset; content-based embedding similarity is the correct paradigm.
- No streaming/real-time corpus updates. The corpus is a static snapshot; refresh = rerun the pipeline. Documented, not built.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language / runtime | Python 3.12 (uv) | Matches portfolio siblings; `>=3.12` already pinned |
| Dependency mgmt | uv + `uv.lock` | Portfolio convention |
| Data ingestion | HuggingFace `datasets` streaming (`open-index/open-arxiv`, Parquet) | ~1.4GB compressed vs 5.4GB Kaggle JSONL; streams per-shard, no full-file load |
| Embedding model | `all-MiniLM-L6-v2` — dynamic UINT8 ONNX (384-dim, 22.7M params, **23MB**) | Fastest proven general model; ~300-950 papers/sec with quantized ONNX (M-series at the high end; Intel i5 dev laptop ~2× faster than fp32) |
| Inference runtime | ONNX backend (`optimum`/onnxruntime), **dynamic UINT8** (`model_quint8_avx2.onnx`) | ~3× CPU speedup vs raw PyTorch, plus ~2× more from UINT8 (23MB vs 90MB, <1% STS-B loss). Quantized configs show ≈equal speedups across arm64/avx2/avx512 (corrects an earlier "int8 is a wash on ARM NEON" claim). Non-VNNI Intel → unsigned variant; static QDQ + `reduce_range` is the accuracy-max alternative (see §12). |
| Vector index | FAISS `IVF8192,SQ8` (default) / `HNSW32` (benchmark profile) | Quantized index = ~1GB for 2.5M vectors, ~95% recall, ~1-10ms search (M-series at the low end; Intel i5 dev laptop at the high end) |
| Similarity metric | Cosine (L2-normalized + inner product) | Standard for text embeddings |
| API | FastAPI + Pydantic v2 (`src/api/`) | Portfolio convention (matches customer-churn project) |
| Demo UI | Streamlit (`dashboard/app.py`) | Portfolio convention |
| Tests | pytest + `conftest.py` fixtures | Portfolio convention |
| Linting | ruff | Portfolio convention (`.ruff_cache` present in siblings) |
| Containerization | Dockerfile + docker-compose (multi-service, healthchecks) | Portfolio convention |

---

## 3. System Architecture

```
                    ┌────────────────────────────────────────────────────────────┐
                    │                    OFFLINE PIPELINE (batch)                │
                    │                                                            │
 HF Parquet ──────► │  1. INGEST     2. EMBED     3. INDEX     4. BENCH/EVAL     │
 (open-index/       │  streaming →    chunked ONNX  IVF8192,    recall@10,       │
  open-arxiv)       │  snapshot.parq  chunk_{n}    SQ8 →        purity@k,        │
                    │  + ids.npy      .npy +       arxiv.index  latency, mem     │
                    │                 manifest.json + benchmark                 │
                    └──────┬──────────────────────────────┬─────────────────────┘
                           │                              │
                           ▼                              ▼
                    ┌───────────────┐             ┌──────────────────────┐
                    │    ARTIFACTS  │             │  ONLINE SERVING      │
                    │  snapshot.parq│             │  (FastAPI)           │
                    │  arxiv.index  │  loaded at  │                      │
                    │  metadata     │  lifespan   │  /recommend/paper/{} │
                    │               ├────────────►│  /recommend/query    │
                    │               │             │  /papers/search      │
                    └───────────────┘             │  /papers/{id}        │
                                                  └──────────┬───────────┘
                                                             │ HTTP
                                                    ┌────────▼───────────┐
                                                    │  STREAMLIT DASHBOARD │
                                                    │  query mode + seed   │
                                                    │  paper picker        │
                                                    └─────────────────────┘
```

**Key architectural decisions** (with rationale):

| Decision | Choice | Why |
|---|---|---|
| Vector store | **In-process FAISS** (no vector DB) | `read_index` loads 1GB in seconds; a vector DB (Qdrant/pgvector) adds a service, network hop, and id-alignment ceremony that teaches nothing at this scale. `VectorIndex` ABC keeps the swap-in story without building it. |
| Index profile | **IVF8192,SQ8** default | 2.5M × 384-dim fp32 = 3.66GB; SQ8 → ~1GB. Fits 8GB if memory-hungry apps are closed during the build (~2GB peak), ~95% recall, ~1-10ms search. HNSW (4.3GB, ~8-10GB build peak) **does not fit 8GB at full corpus** — kept only as a CS-subset benchmark profile. |
| k-NN precompute | **No** | Request-time embed of one paper is 50-100ms — no latency problem. Precompute adds build time + artifact + staleness for identical results (encode is deterministic). |
| Hybrid BM25+RRF | **No** (Phase 3 option) | Adds a build artifact + 100-300ms full-corpus scoring for a benefit reviewers rarely notice. The senior move is *evaluation*, not more machinery. |
| Signals | **Pure cosine score**; year/category as post-filters | Score weights are numerology ("why 0.2?"); post-filters are engineering (two-stage: candidate gen → filter → rank). |
| Corpus scope | **Full corpus** (all categories); `CORPUS_FILTER` config flip exists | User requirement. Cost: specter2 A/B hard-locked out at full scale (768-dim ≈ 2× memory). Documented tradeoff. |

---

## 4. Data Pipeline

### 4.1 Ingest (`src/ingest.py`)

Streams the HuggingFace Parquet mirror shard-by-shard (never loads the full corpus into RAM).

1. Stream shards; select fields: `id`, `title`, `abstract`, `categories`, `authors`, `update_date`.
2. Drop rows with empty `abstract` or `title` (expect ~1-5% loss).
3. Deduplicate on `id` (the HF mirror has cross-shard duplicates).
4. Filter categories against `CORPUS_FILTER` (regex, `None` = all — default all).
5. Write **canonical snapshot**: `data/processed/snapshot.parquet` **sorted by `id`**, with a dense integer row index.

**Why sort-once-at-ingest**: row `i` of every downstream artifact ↔ `ids[i]` of the snapshot. Alignment is free and immutable; nothing downstream re-sorts.

**Outputs**: `data/processed/snapshot.parquet`, `data/processed/ids.npy` (int64, N).

### 4.2 Embed (`src/embed.py`)

Batch encoding with crash-safe checkpoints.

1. Iterate the snapshot in fixed chunks of **100K rows** (~153MB float32 per chunk).
2. Encode `title + " " + abstract` with `all-MiniLM-L6-v2` — **dynamic UINT8 ONNX** (`model_quint8_avx2.onnx`, 23MB, the AVX2 config for the dev laptop's ISA), `batch_size=64`.
3. Truncate to the model's context window (**256 wordpieces**) — deterministic truncation, documented.
4. L2-normalize each vector (cosine via inner product).
5. Per chunk: write `data/embeddings/chunk_{n:05d}.npy` + append `{n: {"start", "end", "rows", "shape"}}` to `data/embeddings/manifest.json`.

**Resume semantics**: a chunk is *atomic* — present in the manifest with a matching shape ⇒ done. Resume skips completed chunks. Verify shape on resume to catch torn `.npy` writes. No per-id bookkeeping.

**Expected throughput** (ONNX dynamic UINT8): ~300-950 papers/sec depending on hardware → **~1.5-3h for 2.5M papers** on the Intel i5 dev laptop (≈2× faster than fp32; faster still on M-series).

### 4.3 Index (`src/index.py`)

FAISS index build with its own checkpointing.

1. Train on a representative **10% sample** (~250K vectors; FAISS rule: ≥30×nlist).
2. Stream chunk `.npy` files into `index.add()` — **never concatenate the full matrix into RAM** (that would cost 3.66GB).
3. Profile A (default): `IVF8192,SQ8`, `METRIC_INNER_PRODUCT`, `nprobe=64`.
4. Profile B (benchmark): `HNSW32` (`efConstruction=200`, `efSearch=64`).
5. `faiss.write_index()` after every 500K rows → crash-resume for the build step (index inserts are deterministic; resumes are exact).
6. Save id alignment alongside: `data/index/arxiv.index` + `data/index/index_meta.json` (profile, d, n, nprobe/ef params).

**Memory math (8GB laptop — MacBook Pro 15,4, Intel i5-8257U)**:

| Resource | IVF8192,SQ8 | HNSW32 (full corpus) |
|---|---|---|
| Index (2.5M × 384-dim) | ~1.0 GB | ~4.3 GB |
| Model (ONNX UINT8) | ~0.03 GB | ~0.03 GB |
| Embed chunk buffer | ~0.15 GB | ~0.15 GB |
| Peak during build | ~2 GB | ~8-10 GB |
| Baseline (macOS + editor/browser) | ~3-4 GB | ~3-4 GB |
| **Total on 8GB?** | **~5-6 GB — fits, tight** | **~11-14 GB — does not fit** |

- **IVF8192,SQ8 (default)**: fits 8GB only if memory-hungry apps (browser, heavy editors) are closed during the build. No headroom — never run the ingest/embed/index pipelines concurrently.
- **HNSW32 (benchmark profile)**: impossible at full corpus on 8GB — build peak alone exceeds total RAM. The HNSW benchmark runs on the **CS-subset scope** (`CORPUS_FILTER = ("cs.*",)`, ~800K papers): index ~1.4GB, build peak ~3GB — fits. See §12.
- **Serving** (index resident + model + FastAPI + Streamlit): ~1.5-2GB — comfortable alongside normal apps.

### 4.4 Benchmark (`src/bench_index.py`)

The measurable proof of the optimization. On a **100K-paper held-out subset**:

- **Recall@10** against exhaustive `IndexFlatIP` ground truth (the same subset)
- **p95 / mean search latency** (single-threaded, 1K queries)
- **Memory footprint** (RSS delta while index is resident)
- Output: table to `reports/index_benchmark.md` + JSON for the README

This converts "we chose SQ8" from a claim into a measured, documented decision.

---

## 5. Serving Layer (FastAPI)

Structure mirrors the customer-churn portfolio convention: `src/api/{app,router,schemas}.py`.

### Endpoints

| Endpoint | Method | Purpose | Cost |
|---|---|---|---|
| `/health` | GET | Liveness/readiness (used by compose healthcheck) | instant |
| `/papers/{arxiv_id}` | GET | Full metadata lookup for a paper | instant |
| `/papers/search?q=&limit=` | GET | **Title picker** — find a seed paper by title substring/keyword (needed by the dashboard) | ~10-50ms |
| `/recommend/paper/{arxiv_id}?k=&year_min=&categories=` | GET | **Item-to-item**: embed that paper's title+abstract → ANN → filter → rank | ~50-150ms |
| `/recommend/query?q=&k=&year_min=&categories=` | GET | **User-to-item**: embed free text → ANN → filter → rank | ~50-150ms |

### Request/Response flow (both recommend routes)

```
query text / paper text ──► embed (locked, ONNX) ──► FAISS top-100 (thread-safe)
                                                         │
                                              post-filter: year_min, categories
                                                         │
                                              top-k + metadata join + arxiv links
```

### Concurrency rules (from research)

- FAISS `search` is **documented thread-safe** for concurrent reads — no lock.
- `torch`/ONNX `encode` is **not** guaranteed safe for concurrent forward passes → guard with a single `threading.Lock`.
- Plain `def` endpoints (FastAPI threadpool) — never `async def` with blocking encode/search inside.
- `OMP_NUM_THREADS=2-4` and `faiss.omp_set_num_threads(...)`: torch and FAISS both use OpenMP; without capping, cores thrash (8 threads on the dev laptop, more on M-series).
- Model + index loaded once at lifespan startup; `read_index` of 1GB ≈ 2-5s.

### Schemas (Pydantic v2)

```python
class Paper(BaseModel):          # id, title, abstract, authors, categories, update_date, arxiv_url
class Recommendation(BaseModel): # paper: Paper, score: float
class RecommendResponse(BaseModel): # query, k, took_ms, results: list[Recommendation]
class PaperSearchResponse(BaseModel): # results: list[Paper], total
```

---

## 6. Demo Dashboard (Streamlit)

Single `dashboard/app.py` (churn-project pattern), two modes via sidebar:

1. **Interest query mode**: free-text box → `/recommend/query` → ranked cards with title, score, category chips, abstract expander, "View on arXiv" link.
2. **Seed paper mode**: title search (`/papers/search`) to pick a paper → `/recommend/paper/{id}` → ranked cards.

`@st.cache_data` on API calls. `API_BASE_URL` from env (default `http://localhost:8000/api/v1`). Streamlit runs on `:8501`.

---

## 7. Evaluation (`src/evaluate.py`)

Two cheap, defensible proxies (no citation graph exists in this dataset):

1. **Category-purity@k**: for a held-out sample of seed papers, measure what fraction of top-k recommendations share a top-level category (e.g. `cs.*`). Baseline: random selection. Purity above baseline = semantic signal present.
2. **Hand-judged query set**: ~10 curated queries (e.g. "retrieval augmented generation for code", "efficient transformers inference"), judged relevance of top-5 on a 0-3 scale, averaged and reported in README.

Reports → `reports/evaluation.md`.

---

## 8. Testing Strategy

| Test file | Scope | Fixtures |
|---|---|---|
| `tests/conftest.py` | Tiny synthetic corpus (50 fake papers, 3 categories), mini FAISS index, FastAPI `TestClient` | `synthetic_papers`, `mini_index`, `client` |
| `tests/test_ingest.py` | Dedup, empty-abstract drop, category filter, sort invariant | synthetic raw rows |
| `tests/test_embed.py` | Chunk shape, normalization (‖v‖≈1), manifest + resume correctness (simulated partial failure) | 2 chunks of synthetic text |
| `tests/test_index.py` | Build/search round-trip, id alignment, `VectorIndex` ABC contract | mini index |
| `tests/test_api.py` | All 5 endpoints: status codes, response models, filter params, 404 for unknown paper | `client` |

Unit tests use the synthetic corpus (no model download — the model can be mocked or skipped via `@pytest.mark.embedding`). Integration tests are fast.

---

## 9. Deployment

### Docker

- `Dockerfile` — multi-stage: builder (uv sync) → runtime (uv, non-root).
- `docker-compose.yml` — two services with healthchecks:

```yaml
services:
  app:
    build: .                    # uvicorn src.api.app:app --port 8000
    ports: ["8000:8000"]
    environment:
      INDEX_PATH: /app/data/index/arxiv.index
      API_BASE_URL: http://app:8000/api/v1
    volumes: ["./data:/app/data"]
    healthcheck: { test: ["CMD", "curl", "-f", "http://localhost:8000/health"], start_period: 60s }

  dashboard:
    build: .                    # streamlit run dashboard/app.py --port 8501
    ports: ["8501:8501"]
    environment:
      API_BASE_URL: http://app:8000/api/v1
    depends_on: { app: { condition: service_healthy } }
```

**Gotcha (documented)**: on an 8GB host, Docker Desktop's VM must be capped well below the host total — allocate ~4GB. The ~1GB IVF/SQ8 index + app (~2GB total in the VM) fits; **HNSW profile does not** in any VM on this machine. If HNSW serving is ever desired, it requires a larger host. `start_period: 60s` covers index load.

### Config (`src/config.py`)

Centralized `Paths` / `IndexParams` / `ModelConfig` classes + env overrides (`INDEX_PATH`, `API_BASE_URL`, `OMP_NUM_THREADS`), matching sibling conventions.

---

## 10. Project Structure

```
research-paper-recommendation/
├── pyproject.toml            # uv, deps, [tool.pytest.ini_options], [tool.ruff]
├── uv.lock
├── README.md                 # tech stack table, architecture, quick start, API docs, eval results
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── embed.py
│   ├── index.py
│   ├── bench_index.py
│   ├── evaluate.py
│   ├── search.py             # VectorIndex ABC + FAISS impl + two-stage filter
│   └── api/
│       ├── __init__.py
│       ├── app.py            # create_app() factory, lifespan (load index+model)
│       ├── router.py
│       └── schemas.py
├── dashboard/
│   └── app.py                # Streamlit
├── tests/
│   ├── conftest.py
│   └── test_{ingest,embed,index,api}.py
├── data/
│   ├── raw/                  # (HF cache — gitignored)
│   └── processed/            # snapshot.parquet, ids.npy, embeddings/, index/  (gitignored)
├── models/                   # model cache (gitignored)
├── notebooks/                # exploratory/visualization
├── docs/
│   └── architecture.md       # this file
├── reports/
│   ├── index_benchmark.md
│   └── evaluation.md
├── Dockerfile
├── docker-compose.yml
└── .gitignore                # data/, models/, .venv, __pycache__, reports artifacts
```

---

## 11. Effort & Timeline

| Phase | Effort | Notes |
|---|---|---|
| 1. Scaffold (uv, pyproject, config, .gitignore) | Small | 1 session |
| 2. Ingest + tests | Small | streaming from HF |
| 3. Embed pipeline + tests | Medium | ONNX + manifest/resume |
| 4. Index + bench + tests | Medium | IVF/SQ8 + benchmark report |
| 5. FastAPI + tests | Medium | 5 endpoints, concurrency |
| 6. Streamlit dashboard | Small | 2 modes |
| 7. Eval + README + Docker | Medium | reports + docs + compose |

**Total**: ~2-3 weeks part-time. The 2.5M-paper embed job (~2h) runs once in the background during phase 4-5 development.

---

## 12. Open Tradeoffs & Follow-ups

1. **specter2 A/B** — blocked at full-corpus scale on 8GB (768-dim ≈ 2× index memory, plus a ~1.7GB model). Path forward: rerun pipeline with `CORPUS_FILTER = ("cs.*",)` (~800K papers, ~1.3GB even at 768-dim) and compare purity@k. Documented as Phase 3.
2. **BM25 + Reciprocal Rank Fusion** — hybrid lexical channel (catches author names, exact identifiers like "GPT-4", out-of-vocabulary terms that embeddings mangle). Self-contained (~`rank_bm25`, zero deps). Phase 3 option.
3. **Vector DB swap** — `VectorIndex` ABC makes Qdrant/pgvector a ~1-day swap if scale or multi-service optics are ever wanted.
4. **mmap/on-disk index** — if full corpus + specter2 ever both matter, switch to an on-disk IVFFlat/mmap profile instead of HNSW.
5. **Corpus refresh** — static snapshot by design; refresh = rerun pipeline.
6. **Static INT8 (QDQ) vs dynamic UINT8** — dynamic UINT8 is the default (no calibration, ~2× on Intel, <1% STS-B loss). Static QDQ calibrated on the corpus abstracts (`quantize_static` with `reduce_range=True` for non-VNNI x86) can shave additional latency at the cost of an offline calibration pass. Revisit only if encode throughput becomes a bottleneck.
