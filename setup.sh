#!/usr/bin/env bash
# =============================================================================
# arXiv Paper Recommendation System — interactive setup
#
# Installs the Python environment, explains what each corpus size costs in
# bandwidth, disk, RAM, and time, and asks for explicit permission before
# downloading/ingesting any paper data.
#
# Quick start:
#   ./setup.sh                     # interactive; safe defaults
#   ./setup.sh --help              # all options
#
# Non-interactive examples (only use these if you know what you want):
#   ./setup.sh --yes --scope skip --no-build           # deps + offline tests only
#   ./setup.sh --yes --scope smoke                     # download <=2 batches, no build
#   ./setup.sh --yes --scope category --category cs --build
#
# Paper data source: https://huggingface.co/datasets/open-index/open-arxiv
# =============================================================================

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET="open-index/open-arxiv"
MODEL_REPO="sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR="$ROOT/models/all-MiniLM-L6-v2"
MODEL_FILE="$MODEL_DIR/model_quint8_avx2.onnx"

SCOPE=""                 # full | category | subcategory | smoke | skip
CATEGORY=""
SUBCATEGORY=""
MAX_BATCHES=2
BUILD=""                 # yes | no | "" (ask)
ASSUME_YES=0
DO_INSTALL=1
DO_TEST=1

MONITOR_PID=""

# ── Output helpers ───────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m'
    BLUE=$'\033[34m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    RESET=$'\033[0m'
else
    BOLD="" BLUE="" GREEN="" YELLOW="" RED="" RESET=""
fi

section() {
    printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"
    printf '%s\n' "------------------------------------------------------------"
}

info() { printf '%s==>%s %s\n' "$BLUE" "$RESET" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%swarning:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

Runs an interactive setup: installs Python dependencies, shows the resource
cost of each corpus size, and asks before downloading/ingesting paper data.

Corpus options:
  --scope SCOPE         full | category | subcategory | smoke | skip
                        (default: interactive menu)
  --category NAME       Archive for --scope category (e.g. cs)
  --subcategory NAME    arXiv id for --scope subcategory (e.g. cs.AI)
  --max-batches N       Batches for --scope smoke (default: 2)

Build options:
  --build               Embed the snapshot and build the FAISS index
  --no-build            Only ingest; skip embedding/indexing
                        (default: ask, unless --yes is used -> skip)

Other options:
  --yes, -y             Auto-confirm prompts. Requires --scope. Without
                        --build, embedding/indexing is skipped.
  --no-install          Do not run `uv sync`; use the existing .venv
  --no-test             Skip the offline pytest verification
  -h, --help            Show this help

Scopes:
  full         Whole arXiv corpus (~2.5M papers)
  category     One archive, e.g. cs (all cs.* papers)
  subcategory  One arXiv category, e.g. cs.AI
  smoke        Only the first few batches; quick ingest check (add --build
               to embed/index the result)
  skip         Do not download; use the existing snapshot, if any

Examples:
  ./setup.sh
  ./setup.sh --scope smoke
  ./setup.sh --scope category --category cs --build
  ./setup.sh --yes --scope skip --no-build
EOF
}

# ── Prompt helpers ───────────────────────────────────────────────────────────
ask_yes_no() {
    local prompt="$1" default="${2:-n}" reply=""
    if [ "$ASSUME_YES" -eq 1 ]; then
        printf '%s yes (--yes)\n' "$prompt"
        return 0
    fi
    if [ ! -t 0 ]; then
        printf '%s no (non-interactive; pass --yes to confirm)\n' "$prompt"
        return 1
    fi
    if [ "$default" = "y" ]; then
        printf '%s [Y/n]: ' "$prompt"
    else
        printf '%s [y/N]: ' "$prompt"
    fi
    read -r reply || reply=""
    reply="${reply:-$default}"
    case "$reply" in
        y|Y|yes|YES|Yes) return 0 ;;
        *) return 1 ;;
    esac
}

ask_value() {
    local prompt="$1" reply=""
    [ -t 0 ] || return 1
    printf '%s' "$prompt" >&2
    read -r reply || reply=""
    printf '%s' "$reply"
}

format_duration() {
    local seconds="$1"
    if [ "$seconds" -lt 60 ]; then
        printf '%ss' "$seconds"
    elif [ "$seconds" -lt 3600 ]; then
        printf '%s min' "$((seconds / 60))"
    else
        printf '%sh %sm' "$((seconds / 3600))" "$(((seconds % 3600) / 60))"
    fi
}

# ── Environment helpers ──────────────────────────────────────────────────────
available_disk_gb() {
    local kb
    kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
    printf '%s' "$((kb / 1024 / 1024))"
}

total_ram_mb() {
    if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.memsize >/dev/null 2>&1; then
        printf '%s' "$(( $(sysctl -n hw.memsize) / 1048576 ))"
    elif [ -r /proc/meminfo ]; then
        awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo
    else
        printf '0'
    fi
}

require_disk() {
    local need_gb="$1" avail_gb
    avail_gb="$(available_disk_gb)"
    if [ "$avail_gb" -lt "$need_gb" ]; then
        die "not enough free disk: need ~${need_gb} GB, have ${avail_gb} GB in $ROOT"
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    warn "uv is required but was not found on PATH."
    if ask_yes_no "Install uv from astral.sh now?" "y"; then
        command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        command -v uv >/dev/null 2>&1 || die "uv installed but not found; add ~/.local/bin to PATH"
        ok "uv installed: $(uv --version)"
    else
        die "install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    fi
}

count_snapshot_rows() {
    uv run python - <<'PY'
import polars as pl
from src.config import Paths

print(pl.scan_parquet(Paths.SNAPSHOT).select(pl.len()).collect().item())
PY
}

model_ready() {
    [ -f "$MODEL_FILE" ] \
        && [ -f "$MODEL_DIR/tokenizer.json" ] \
        && [ -f "$MODEL_DIR/tokenizer_config.json" ]
}

# ── Ingest progress monitor ──────────────────────────────────────────────────
monitor_staging() {
    local n last=-1
    while sleep 30; do
        n="$(find "$ROOT/data/raw/staging" -maxdepth 1 -type f -name 'batch_*.parquet' 2>/dev/null \
            | wc -l | tr -d ' ')"
        if [ "$n" != "$last" ] && [ "$n" -gt 0 ]; then
            printf '  [ingest] %s batches staged...\n' "$n"
            last="$n"
        fi
    done
}

stop_monitor() {
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

trap stop_monitor EXIT
trap 'stop_monitor; warn "interrupted; rerun ./setup.sh to resume (completed embed chunks are kept)"; exit 130' INT TERM

# ── Argument parsing ─────────────────────────────────────────────────────────
set_scope() {
    local wanted="$1"
    if [ -n "$SCOPE" ] && [ "$SCOPE" != "$wanted" ]; then
        die "--scope $SCOPE conflicts with the requested scope ($wanted)"
    fi
    SCOPE="$wanted"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --scope)
            [ "$#" -ge 2 ] || die "--scope requires a value"
            SCOPE="$2"
            shift 2
            ;;
        --scope=*)
            SCOPE="${1#*=}"
            shift
            ;;
        --category)
            [ "$#" -ge 2 ] || die "--category requires a value"
            CATEGORY="$2"
            set_scope category
            shift 2
            ;;
        --category=*)
            CATEGORY="${1#*=}"
            set_scope category
            shift
            ;;
        --subcategory)
            [ "$#" -ge 2 ] || die "--subcategory requires a value"
            SUBCATEGORY="$2"
            set_scope subcategory
            shift 2
            ;;
        --subcategory=*)
            SUBCATEGORY="${1#*=}"
            set_scope subcategory
            shift
            ;;
        --max-batches)
            [ "$#" -ge 2 ] || die "--max-batches requires a value"
            MAX_BATCHES="$2"
            shift 2
            ;;
        --max-batches=*)
            MAX_BATCHES="${1#*=}"
            shift
            ;;
        --build)
            BUILD=yes
            shift
            ;;
        --no-build)
            BUILD=no
            shift
            ;;
        --yes|-y)
            ASSUME_YES=1
            shift
            ;;
        --no-install)
            DO_INSTALL=0
            shift
            ;;
        --no-test)
            DO_TEST=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

case "$MAX_BATCHES" in
    ''|*[!0-9]*) die "--max-batches must be a positive integer" ;;
esac
[ "$MAX_BATCHES" -gt 0 ] || die "--max-batches must be a positive integer"

if [ -n "$CATEGORY" ] && [ -n "$SUBCATEGORY" ]; then
    die "use either --category or --subcategory, not both"
fi

case "$SCOPE" in
    ""|full|category|subcategory|smoke|skip) ;;
    *) die "invalid --scope '$SCOPE' (full|category|subcategory|smoke|skip)" ;;
esac

if [ "$ASSUME_YES" -eq 1 ] && [ -z "$SCOPE" ]; then
    die "--yes requires an explicit --scope (e.g. --scope skip)"
fi

# ── Intro ────────────────────────────────────────────────────────────────────
cat <<EOF
${BOLD}arXiv Paper Recommendation System — setup${RESET}

This script will:
  1. install Python dependencies into .venv (downloads packages from PyPI)
  2. show the bandwidth, disk, RAM, and time cost of each corpus size
  3. ask permission before downloading any paper data
  4. optionally embed the snapshot and build the FAISS index

No paper data is downloaded until you explicitly confirm.
Paper data source: https://huggingface.co/datasets/$DATASET
EOF

# ── 1/3 Install ──────────────────────────────────────────────────────────────
section "1/3  Python environment"

if [ "$DO_INSTALL" -eq 1 ]; then
    ensure_uv
    info "Syncing dependencies (uv sync; first run may download hundreds of MB)..."
    uv sync
    ok "dependencies installed ($(uv run python -V 2>&1))"
else
    [ -x "$ROOT/.venv/bin/python" ] || die "--no-install requires an existing .venv; run without --no-install"
    info "Skipping install (--no-install); using $ROOT/.venv"
fi

if [ "$DO_TEST" -eq 1 ]; then
    info "Verifying the install with the offline test suite..."
    if uv run pytest -q; then
        ok "offline tests passed"
    else
        warn "offline tests failed; the pipeline may not work. Re-run: uv run pytest -q"
    fi
else
    info "Skipping offline tests (--no-test)"
fi

# ── 2/3 Corpus ───────────────────────────────────────────────────────────────
section "2/3  Paper corpus"

cat <<'EOF'
Resource estimates (8GB Intel-Mac reference machine from docs/architecture.md):

  Corpus scope                          Network    Disk*    Ingest    Embed**
  Full arXiv (~2.5M papers)             ~1.4 GB    ~8 GB    ~1.5 h    ~35-40 h
  One archive, e.g. cs (~0.4-0.8M)      ~1.4 GB    ~3-4 GB  ~1.5 h    ~6-12 h
  One subcategory, e.g. cs.AI           ~1.4 GB    ~2-3 GB  ~1.5 h    ~1-4 h
  Quick smoke test (2 batches, <=16K)   ~10-40 MB  ~0.5 GB  ~1 min    ~15 min
  Skip download                         none       none     -         -

  * Disk = free space recommended for staging, snapshot, embeddings, index.
  ** Measured at ~17-20 papers/sec on the reference laptop; index adds minutes.
     Faster hardware finishes sooner; longer abstracts take longer.

  RAM: ~1-2 GB during ingest, ~2 GB peak during embed/index, ~1.5-2 GB serving.
  Filtered ingests still stream the entire mirror, so download and ingest time
  stay close to a full run even when the final snapshot is smaller. Python
  package and HuggingFace caches use additional disk outside data/.
EOF

if [ -z "$SCOPE" ]; then
    if [ ! -t 0 ]; then
        die "no corpus scope selected and stdin is not a terminal; pass --scope (e.g. --scope skip)"
    fi
    printf '\nWhat should this setup do about the paper corpus?\n'
    printf '  1) Full arXiv corpus (slowest; ~35-40 h to embed)\n'
    printf '  2) One archive, e.g. cs\n'
    printf '  3) One subcategory, e.g. cs.AI\n'
    printf '  4) Quick smoke test (2 batches; verifies the whole pipeline)\n'
    printf '  5) Skip download (use the existing snapshot, if any)\n'
    while :; do
        printf 'Choose 1-5 [4]: '
        reply=""
        read -r reply || reply=""
        reply="${reply:-4}"
        case "$reply" in
            1) SCOPE=full; break ;;
            2) SCOPE=category; break ;;
            3) SCOPE=subcategory; break ;;
            4) SCOPE=smoke; break ;;
            5) SCOPE=skip; break ;;
            *) warn "invalid choice: $reply" ;;
        esac
    done
fi

if [ "$SCOPE" = "category" ] && [ -z "$CATEGORY" ]; then
    CATEGORY="$(ask_value 'Archive name (e.g. cs): ' || true)"
    [ -n "$CATEGORY" ] || die "an archive name is required for --scope category"
fi

if [ "$SCOPE" = "category" ]; then
    case "$CATEGORY" in
        *.*) die "archive names have no dot, e.g. cs; use --subcategory for cs.AI" ;;
    esac
fi

if [ "$SCOPE" = "subcategory" ]; then
    if [ -z "$SUBCATEGORY" ]; then
        SUBCATEGORY="$(ask_value 'Subcategory id (e.g. cs.AI): ' || true)"
        [ -n "$SUBCATEGORY" ] || die "a subcategory id is required for --scope subcategory"
    fi
    case "$SUBCATEGORY" in
        *.*) ;;
        *) die "subcategory must look like archive.sub, e.g. cs.AI" ;;
    esac
fi

# Existing artifacts, shown after dependencies are installed so polars is usable.
section "Existing artifacts"
if [ -f "$ROOT/data/processed/snapshot.parquet" ]; then
    rows="$(count_snapshot_rows 2>/dev/null || true)"
    case "$rows" in
        ''|*[!0-9]*) rows="unknown" ;;
    esac
    printf '  snapshot:   data/processed/snapshot.parquet (%s rows)\n' "$rows"
else
    printf '  snapshot:   none\n'
fi
chunks=0
if [ -d "$ROOT/data/embeddings" ]; then
    chunks="$(find "$ROOT/data/embeddings" -maxdepth 1 -type f -name 'chunk_*.npy' | wc -l | tr -d ' ')"
fi
printf '  embeddings: %s chunk(s)\n' "$chunks"
if [ -f "$ROOT/data/index/arxiv.index" ]; then
    printf '  index:      data/index/arxiv.index (%s)\n' "$(du -h "$ROOT/data/index/arxiv.index" | awk '{print $1}')"
else
    printf '  index:      none\n'
fi
if model_ready; then
    printf '  model:      models/all-MiniLM-L6-v2 (ready)\n'
else
    printf '  model:      not downloaded\n'
fi
printf '  free disk:  %s GB\n' "$(available_disk_gb)"
ram_mb="$(total_ram_mb)"
if [ "$ram_mb" -gt 0 ]; then
    printf '  total RAM:  %s GB\n' "$((ram_mb / 1024))"
fi

scope_title=""
scope_network=""
scope_disk=1
scope_time=""
case "$SCOPE" in
    full)
        scope_title="full arXiv corpus (~2.5M papers)"
        scope_network="~1.4 GB"
        scope_disk=8
        scope_time="~1.5 h (streams the full mirror)"
        ;;
    category)
        scope_title="archive '$CATEGORY' (all $CATEGORY.* papers)"
        scope_network="~1.4 GB"
        scope_disk=4
        scope_time="~1.5 h (streams the full mirror, then filters)"
        ;;
    subcategory)
        scope_title="subcategory '$SUBCATEGORY'"
        scope_network="~1.4 GB"
        scope_disk=3
        scope_time="~1.5 h (streams the full mirror, then filters)"
        ;;
    smoke)
        scope_title="quick smoke test ($MAX_BATCHES batch(es), at most $((MAX_BATCHES * 8192)) rows)"
        scope_network="~10-40 MB"
        scope_disk=1
        scope_time="~1 min"
        ;;
    skip)
        scope_title=""
        ;;
esac

if [ "$SCOPE" != "skip" ]; then
    cat <<EOF

About to download and ingest:
  scope:   $scope_title
  source:  $DATASET on HuggingFace (streamed)
  network: $scope_network
  disk:    ~${scope_disk} GB recommended; $(available_disk_gb) GB available
  time:    $scope_time
  RAM:     ~1-2 GB peak

This will:
  - clear data/raw/staging/batch_*.parquet
  - replace data/processed/snapshot.parquet and ids.npy
  - delete existing embeddings/index afterwards (they would be misaligned with
    the new snapshot)

If HuggingFace rate-limits you, set HF_TOKEN before running this script.
EOF
    require_disk "$scope_disk"
    if ask_yes_no "Download and ingest this corpus now?" "n"; then
        info "Checking HuggingFace connectivity..."
        if command -v curl >/dev/null 2>&1; then
            if ! curl -fsS --max-time 15 -o /dev/null https://huggingface.co; then
                warn "could not reach huggingface.co; ingest may fail."
            fi
        fi

        info "Clearing ingest staging area..."
        rm -f "$ROOT"/data/raw/staging/batch_*.parquet
        mkdir -p "$ROOT/data/raw/staging"

        ingest_args=()
        case "$SCOPE" in
            full) ingest_args=(--full) ;;
            category) ingest_args=(--category "$CATEGORY") ;;
            subcategory) ingest_args=(--subcategory "$SUBCATEGORY") ;;
            smoke) ingest_args=(--full --max-batches "$MAX_BATCHES") ;;
        esac

        info "Streaming $DATASET; progress is shown as batches are staged."
        monitor_staging &
        MONITOR_PID=$!
        if ! uv run python -m src.ingest "${ingest_args[@]}"; then
            stop_monitor
            die "ingest failed. Partial staging data remains in data/raw/staging; rerun ./setup.sh to retry."
        fi
        stop_monitor

        rows="$(count_snapshot_rows 2>/dev/null || true)"
        case "$rows" in
            ''|*[!0-9]*) rows="unknown" ;;
        esac
        ok "snapshot written: $rows rows"

        info "Cleaning ingest staging area..."
        rm -f "$ROOT"/data/raw/staging/batch_*.parquet

        info "Removing embeddings/index built against the previous snapshot..."
        rm -rf "$ROOT/data/embeddings" "$ROOT/data/index"

        mkdir -p "$ROOT/data/processed" "$ROOT/data/embeddings" "$ROOT/data/index"
    else
        warn "Corpus download declined; continuing without downloading papers."
        SCOPE=skip
    fi
else
    info "Skipping corpus download."
fi

# ── 3/3 Embed + index ────────────────────────────────────────────────────────
section "3/3  Embeddings + FAISS index"

if [ "$SCOPE" = "skip" ] && [ ! -f "$ROOT/data/processed/snapshot.parquet" ]; then
    warn "No snapshot found, so there is nothing to embed. Run ./setup.sh again and choose a corpus scope."
    BUILD=no
fi

if [ -z "$BUILD" ]; then
    if [ "$ASSUME_YES" -eq 1 ]; then
        BUILD=no
        info "--yes without --build: skipping embedding/indexing (pass --build to include it)"
    else
        rows="$(count_snapshot_rows 2>/dev/null || true)"
        case "$rows" in
            ''|*[!0-9]*) rows=0 ;;
        esac
        build_seconds=$((rows / 18 + 30))
        build_disk_gb=$((rows * 3 / 1048576 + 1))
        index_mb=$((rows * 4 / 10000))
        [ "$index_mb" -ge 1 ] || index_mb=1
        if [ "$rows" -le 50000 ]; then
            build_default="y"
        else
            build_default="n"
        fi
        cat <<EOF

Embedding + index build:
  rows:        $rows
  embed time:  ~$(format_duration "$build_seconds") at ~18 papers/sec (resumable per chunk)
  index:       IVF8192,SQ8, roughly ${index_mb} MB, minutes to build
  disk:        ~${build_disk_gb} GB free recommended; $(available_disk_gb) GB available
  peak RAM:    ~2 GB (close other heavy apps on an 8GB machine)

EOF
        if ask_yes_no "Run embedding and index build now?" "$build_default"; then
            BUILD=yes
        else
            BUILD=no
        fi
    fi
fi

if [ "$BUILD" = "yes" ]; then
    [ -f "$ROOT/data/processed/snapshot.parquet" ] || die "no snapshot to embed; run ingest first"

    rows="$(count_snapshot_rows 2>/dev/null || true)"
    case "$rows" in
        ''|*[!0-9]*) rows=0 ;;
    esac
    build_disk_gb=$((rows * 3 / 1048576 + 1))
    require_disk "$build_disk_gb"

    ram_mb="$(total_ram_mb)"
    if [ "$ram_mb" -gt 0 ] && [ "$ram_mb" -lt 3500 ]; then
        warn "only $((ram_mb / 1024)) GB RAM detected; embedding may be very slow or run out of memory."
        if ! ask_yes_no "Continue anyway?" "n"; then
            die "aborted because of low RAM; close applications or use a smaller corpus"
        fi
    fi

    if ! model_ready; then
        warn "embedding model missing from models/all-MiniLM-L6-v2"
        if ask_yes_no "Download the ~24 MB ONNX model + tokenizer from HuggingFace?" "y"; then
            info "Downloading $MODEL_REPO ..."
            if ! uv run python - <<'PY'
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

repo = "sentence-transformers/all-MiniLM-L6-v2"
target = Path("models/all-MiniLM-L6-v2")
target.mkdir(parents=True, exist_ok=True)
files = {
    "onnx/model_quint8_avx2.onnx": "model_quint8_avx2.onnx",
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "vocab.txt": "vocab.txt",
    "special_tokens_map.json": "special_tokens_map.json",
}
for remote, local in files.items():
    cached = Path(hf_hub_download(repo_id=repo, filename=remote))
    shutil.copy2(cached, target / local)
    print(f"  downloaded {local}")
PY
            then
                BUILD=no
                warn "model download failed; skipping embed/index. Check network/HF_TOKEN and rerun: ./setup.sh --scope skip --build"
            fi
        else
            BUILD=no
            warn "model download declined; skipping embed/index."
        fi
    fi

    if [ "$BUILD" = "yes" ]; then
        model_ready || die "model is still incomplete in $MODEL_DIR"

        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        info "Embedding $rows rows with OMP_NUM_THREADS=$OMP_NUM_THREADS (Ctrl-C keeps completed chunks)..."
        if ! uv run python - <<'PY'
from src.config import Paths
from src.embed import OnnxEncoder, run

encoder = OnnxEncoder()
encoder.progress = True
summary = run(
    encoder,
    snapshot_path=Paths.SNAPSHOT,
    embeddings_dir=Paths.DATA_EMBEDDINGS,
    manifest_path=Paths.MANIFEST,
)
print(f"embed complete: {summary}")
PY
        then
            die "embedding failed. Completed chunks are kept; rerun ./setup.sh --scope skip --build to resume."
        fi
        ok "embeddings written to data/embeddings"

        info "Building the FAISS IVF8192,SQ8 index..."
        if ! uv run python -c 'from src.index import build_index; print(build_index())'; then
            die "index build failed; rerun ./setup.sh --scope skip --build"
        fi
        ok "index written to data/index/arxiv.index"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
section "Setup complete"

if [ -f "$ROOT/data/processed/snapshot.parquet" ]; then
    rows="$(count_snapshot_rows 2>/dev/null || true)"
    case "$rows" in
        ''|*[!0-9]*) rows="unknown" ;;
    esac
    printf '  snapshot: %s rows\n' "$rows"
else
    printf '  snapshot: none (no paper data downloaded)\n'
fi

if [ -f "$ROOT/data/index/arxiv.index" ]; then
    printf '  index:    data/index/arxiv.index (%s)\n' "$(du -h "$ROOT/data/index/arxiv.index" | awk '{print $1}')"
else
    printf '  index:    none yet\n'
fi

cat <<EOF

Start the API:
  uv run uvicorn src.api.app:app --reload --port 8000

Open:
  UI:        http://localhost:8000/
  API docs:  http://localhost:8000/docs

Verify end-to-end after an index exists:
  uv run python smoke_api.py
EOF

if [ -f "$ROOT/data/processed/snapshot.parquet" ]; then
    cat <<'EOF'

To embed/index the existing snapshot later:
  ./setup.sh --scope skip --build
EOF
else
    cat <<'EOF'

To download a corpus later:
  ./setup.sh                              # interactive menu
  ./setup.sh --scope category --category cs --build
EOF
fi
