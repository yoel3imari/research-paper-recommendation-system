# arXiv Paper Recommendation System

End-to-end recommendation engine for arXiv research papers: stream an open snapshot,
embed titles + abstracts with a quantized ONNX model, build a FAISS vector index,
serve recommendations through a FastAPI service, and explore them in a static
HTML/Tailwind/Alpine.js dashboard.

No full-text or PDF processing — the system stores only bibliographic metadata
(`title`, `abstract`, `authors`, `categories`) and links out to the original paper
on arXiv.

## Highlights

- **Streaming ingest** — HuggingFace `open-index/open-arxiv` mirror is streamed
  shard-by-shard (never loaded into RAM), deduplicated, category-filtered, and
  written as an id-sorted canonical snapshot.
- **Checkpointed embedding** — `all-MiniLM-L6-v2` (dynamic UINT8 ONNX, 23MB,
  384-dim) encodes `title + abstract` in atomic 100K-row chunks with a manifest for
  crash-safe resume.
- **Quantized FAISS index** — `IVF8192,SQ8` (inner product over L2-normalized
  vectors) fits in ~1GB at full corpus scale; nprobe=64.
- **Production-style serving** — FastAPI + Pydantic v2, threadpool endpoints,
  thread-safe FAISS reads, locked ONNX encode.
- **Measured quality** — index benchmark (recall@10 vs exhaustive, latency, memory)
  and category-purity evaluation, reported below.
- **Dockerized** — multi-stage image + compose stack with healthchecks.

## Tech Stack

| Layer | Choice |
|---|---|
| Language / deps | Python 3.12 · uv + `uv.lock` |
| Data ingestion | HuggingFace `datasets` streaming (`open-index/open-arxiv`, Parquet) |
| Embedding model | `all-MiniLM-L6-v2` — dynamic UINT8 ONNX (`model_quint8_avx2.onnx`, 23MB, 384-dim) |
| Inference runtime | `onnxruntime` 1.19.2 (CPU) |
| Vector index | FAISS `IVF8192,SQ8` (default) · `HNSW32` (benchmark profile) |
| Similarity | Cosine (L2-normalized + inner product) |
| API | FastAPI + Pydantic v2 (`src/api/`) |
| Demo UI | HTML + Tailwind CSS + Alpine.js (`dashboard/index.html`) |
| Tests / lint | pytest · ruff |
| Containerization | Dockerfile + docker-compose |

## Architecture

```
                     ┌────────────────────────────────────────────────────────────┐
                     │                    OFFLINE PIPELINE (batch)                │
                     │                                                            │
 HF Parquet ──────►  │  1. INGEST     2. EMBED     3. INDEX     4. BENCH/EVAL     │
 (open-index/        │  streaming →    chunked ONNX  IVF8192,    recall@10,       │
  open-arxiv)        │  snapshot.parq  chunk_{n}    SQ8 →        purity@k,        │
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
                                                     │  HTML + TAILWIND UI  │
                                                     │  served by FastAPI   │
                                                     │  query + seed modes  │
                                                     └─────────────────────┘
```

**Alignment invariant**: row `i` of every artifact (snapshot, `ids.npy`, embedding
chunk, FAISS index) refers to the same paper. The snapshot is sorted by id once at
ingest; nothing downstream re-sorts.

Full design rationale, tradeoffs, and deferred work: [`docs/architecture.md`](docs/architecture.md).

## Project Layout

```
├── pyproject.toml            # uv deps, pytest/ruff config
├── setup.sh                  # interactive setup (deps + optional corpus)
├── src/
│   ├── config.py             # paths, index params, env overrides
│   ├── ingest.py             # streaming snapshot builder (Phase 2)
│   ├── embed.py              # chunked ONNX encoder + manifest/resume (Phase 3)
│   ├── index.py              # FAISS build/search, id alignment (Phase 4)
│   ├── bench_index.py        # recall@10, latency, memory report (Phase 7)
│   ├── evaluate.py           # category-purity + hand-judged queries (Phase 7)
│   └── api/                  # FastAPI app/router/schemas (Phase 5)
├── dashboard/
│   └── index.html            # static Tailwind + Alpine.js demo (Phase 6)
├── tests/                    # pytest suite (offline, synthetic corpus)
├── data/                     # gitignored artifacts (snapshot, embeddings, index)
├── models/                   # gitignored ONNX model cache
├── docs/architecture.md      # full design doc
├── reports/                  # index_benchmark.md, evaluation.md
├── Dockerfile
└── docker-compose.yml
```

## Quick Start

Requires: Python 3.12, `uv`, an Intel-Mac or similar platform (see
[docs/architecture.md §2](docs/architecture.md) for pinned wheel constraints).

### Automated setup (recommended)

```bash
./setup.sh
```

The script installs dependencies with `uv`, prints the bandwidth, disk, RAM, and
time cost of each corpus scope, and **asks for permission before downloading any
paper data**. It can also embed the snapshot and build the FAISS index in the
same run, or stop after ingest so you can do that later.

```bash
./setup.sh --help                              # all options
./setup.sh --scope smoke                       # quick pipeline check
./setup.sh --scope category --category cs      # real corpus, no build
./setup.sh --yes --scope skip --no-build       # CI: deps + tests only
```

Declining the download is safe: setup still verifies the install and prints the
commands to run later. The manual steps below are equivalent.

### 1. Install

```bash
uv sync
```

### 2. Build the pipeline (once)

The pipeline is a one-shot batch job: ingest → embed → index.

Ingest lets you choose what to index: **the full dataset, one category**
(entire archive, e.g. `cs` = every `cs.*` paper), or **one subcategory**
(single arXiv id, e.g. `cs.AI`). Run it with a flag, or with no flag the CLI
prompts interactively:

```bash
# See which categories/subcategories are ACTUALLY in the dataset (streams a
# bounded 50-batch sample and prints each with a paper-count estimate):
uv run python -m src.ingest --list-categories
uv run python -m src.ingest --list-categories --sample-batches 10   # smaller sample

uv run python -m src.ingest                  # prompts: full / category / subcategory
uv run python -m src.ingest --full           # full corpus (~90 min on open-arxiv)
uv run python -m src.ingest --category cs     # everything under cs.* (e.g. cs.AI, cs.LG)
uv run python -m src.ingest --subcategory cs.AI  # only cs.AI papers
```

Picking category/subcategory in the interactive prompt loads that same list
and offers a numbered picker, so you don't have to know a category id in
advance; it also warns before accepting a name never seen in the sample.

> Scope flags are mutually exclusive. With no flag and an interactive terminal
> the CLI prompts for a scope; in a non-interactive shell (CI) a flag is
> required. The legacy `CORPUS_FILTER` regex env var is still honored when set,
> e.g. `CORPUS_FILTER='^cs\.(AI|LG|CL)$' uv run python -m src.ingest`.
> `--max-batches N` limits the run to N batches for a fast smoke test.

Embed the snapshot (chunked, resumable). On a 30K demo subset this takes
minutes; on the full ~400K CS corpus expect hours at ~15-20 papers/sec.
uv run python -m src.embed

# Build the FAISS index from the embedding chunks.
uv run python -c "from src.index import build_index; print(build_index())"
```

**Fast demo path** — subset the snapshot, embed, and index in one command
(recommended for the first run):

```bash
uv run python run_demo.py --rows 30000 --batch 128
```

### 3. Run the API

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>. Health probe at
<http://localhost:8000/api/v1/health>.

### 4. Open the dashboard

The dashboard is a static page served by the API — no Node.js build step and
no second process:

- UI: <http://localhost:8000/>
- Interactive API docs: <http://localhost:8000/docs>

### 5. Verify end-to-end

```bash
uv run python smoke_api.py          # real artifacts: index n == snapshot height,
                                    # seeded + query recommendations work
uv run pytest -q                    # offline unit tests (no model/index needed)
uv run ruff check src/ tests/
```

### 6. Docker

```bash
docker compose up --build
# app + UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

Note: on an 8GB Intel Mac, cap the Docker VM at ~4GB (see
[docs/architecture.md §9](docs/architecture.md)).

## API Reference

Base path `/api/v1`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe (compose healthcheck) |
| `/papers/search?q=&limit=` | GET | Title picker — find a seed paper by title substring |
| `/papers/{arxiv_id}` | GET | Full metadata for one paper |
| `/recommend/query?q=&k=&year_min=&categories=` | GET | Free-text interest → top-k similar papers |
| `/recommend/paper/{arxiv_id}?k=&year_min=&categories=` | GET | Seed paper → top-k similar papers |

`k` (1–100, default 10), `year_min` (1991–2030), `categories` (comma-separated
category tokens). Response shape:

```json
{
  "query": "graph neural networks",
  "k": 5,
  "took_ms": 48.2,
  "results": [
    {
      "paper": {
        "id": "1706.03762",
        "title": "Attention is All you Need",
        "abstract": "...",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "..."],
        "categories": ["cs.CL", "cs.LG"],
        "update_date": "2023-08-02",
        "arxiv_url": "https://arxiv.org/abs/1706.03762"
      },
      "score": 0.74
    }
  ]
}
```

## Measured Results

> Figures below are produced by `src/bench_index.py` and `src/evaluate.py` against
> the real built artifacts. Regenerate with `uv run python -m src.bench_index` and
> `uv run python -m src.evaluate`.

### Index benchmark (IVF-SQ8)

Measured against the built 30,000-vector `ivf-sq8` index (nprobe=64) on an Intel
MacBook (macOS 12, x86_64). Ground truth: exhaustive `IndexFlatIP`. Full report:
[`reports/index_benchmark.md`](reports/index_benchmark.md).

| Metric | Value |
|---|---|
| Recall@10 | 0.9667 |
| p95 search latency | 0.871 ms |
| mean search latency | 0.534 ms |
| Index size (on disk) | 12.9 MB |

### Recommendation quality (category purity@k)

200 seed papers sampled; purity = fraction of top-k that share the seed's
primary sub-category (`cs.AI`/`cs.LG`/`cs.CL`). Baseline = corpus base rate
(0.23). ~3× lift over random = the embeddings capture topical structure.
Full report: [`reports/evaluation.md`](reports/evaluation.md).

| k | Purity@k | Random baseline | Lift |
|---|---|---|---|
| 1 | 1.0000 | 0.2298 | +0.77 |
| 5 | 0.7070 | 0.2298 | +0.48 |
| 10 | 0.6500 | 0.2298 | +0.42 |

### Hand-judged queries (relevance 0–3, top-5)

10 curated research-interest queries; top-5 hits per query are listed in
[`reports/evaluation.md`](reports/evaluation.md) with blank relevance cells for
hand-judging. Example top hits from the live API:

| Query | Top hit | Score |
|---|---|---|
| "retrieval augmented generation for code" | *Code Attention: Translating Code to Comments by Exploiting Domain Features* | 0.538 |
| "graph neural networks" | *Residual Gated Graph ConvNets* | 0.721 |
| seed `1706.03762` (Attention is All You Need) | 5 transformer/attention papers | 0.989 → 0.787 |

## Session Notes

Running notes on constraints, decisions, and phase history live in
[`docs/session-notes.md`](docs/session-notes.md).
