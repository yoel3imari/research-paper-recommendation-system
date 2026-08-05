"""Centralized configuration for the arXiv recommendation pipeline.

Defaults encode the decisions in ``docs/architecture.md``. Environment
variables override deploy-time knobs (see ``Settings``).
"""

from pathlib import Path

import faiss
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Paths ───────────────────────────────────────────────────────────────────
class Paths:
    """Project-relative paths. All derive from the project root."""

    ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_RAW: Path = ROOT / "data" / "raw"
    DATA_PROCESSED: Path = ROOT / "data" / "processed"
    DATA_EMBEDDINGS: Path = ROOT / "data" / "embeddings"
    DATA_INDEX: Path = ROOT / "data" / "index"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"

    SNAPSHOT: Path = DATA_PROCESSED / "snapshot.parquet"
    IDS: Path = DATA_PROCESSED / "ids.npy"
    MANIFEST: Path = DATA_EMBEDDINGS / "manifest.json"
    INDEX: Path = DATA_INDEX / "arxiv.index"
    INDEX_META: Path = DATA_INDEX / "index_meta.json"
    INDEX_BENCHMARK: Path = REPORTS / "index_benchmark.md"
    EVALUATION: Path = REPORTS / "evaluation.md"

    @classmethod
    def ensure(cls) -> None:
        for directory in (
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.DATA_EMBEDDINGS,
            cls.DATA_INDEX,
            cls.MODELS,
            cls.REPORTS,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ── Index parameters ────────────────────────────────────────────────────────
class IndexParams:
    """FAISS build/search parameters (docs/architecture.md §4.3)."""

    DIM: int = 384

    PROFILE_DEFAULT: str = "ivf-sq8"
    PROFILE_HNSW: str = "hnsw"

    METRIC: int = faiss.METRIC_INNER_PRODUCT

    NLIST: int = 8192
    NPROBE: int = 64
    EF_CONSTRUCTION: int = 200
    EF_SEARCH: int = 64
    TRAIN_FRACTION: float = 0.10
    CHECKPOINT_EVERY: int = 500_000


# ── Model configuration ─────────────────────────────────────────────────────
class ModelConfig:
    """Embedding model settings (docs/architecture.md §2)."""

    NAME: str = "all-MiniLM-L6-v2"
    ONNX_FILE: str = "model_quint8_avx2.onnx"
    DIM: int = 384
    MAX_SEQ_LENGTH: int = 256
    BATCH_SIZE: int = 64
    CHUNK_SIZE: int = 100_000


# ── Environment overrides ───────────────────────────────────────────────────
class Settings(BaseSettings):
    """Deploy-time knobs. Override them with environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path | None = None
    api_base_url: str = "http://localhost:8000/api/v1"
    omp_num_threads: int | None = None
    corpus_filter: str | None = None
    dataset_name: str = "open-index/open-arxiv"

    @property
    def resolved_index_path(self) -> Path:
        return self.index_path or Paths.INDEX
