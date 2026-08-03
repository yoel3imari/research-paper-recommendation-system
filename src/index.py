"""FAISS vector-index build/search (docs/architecture.md §4.3).

Builds a quantized index from the chunked embeddings produced by
``src/embed.py`` (one ``.npy`` per chunk + ``manifest.json``):

1. Train on a representative sample (FAISS rule: ``n_train >= nlist``,
   ideally ``>= 30 * nlist``; default 10% via ``IndexParams.TRAIN_FRACTION``).
2. Stream each chunk into ``index.add()`` — the full ``(N, d)`` matrix is
   **never materialized in RAM** (that would cost ~3.66GB at full corpus).
3. Checkpoint via ``faiss.write_index()`` every ``CHECKPOINT_EVERY`` rows
   (index inserts are deterministic, so resumes are exact).
4. Save the index plus ``index_meta.json`` (profile, d, n, search params).

**Memory approach**: the IVF training sample is bounded to
``train_fraction`` rows *per chunk* (loaded through ``np.load(..., mmap_mode)``
and sliced), so peak memory is one chunk buffer plus the train fraction of the
corpus — never the full matrix. ``nlist`` is clamped to ``n_train // 4``
(via ``clamp_nlist``) so tiny synthetic corpora train without hard failure.

**Profiles**: ``ivf-sq8`` (default) — ``IndexIVFScalarQuantizer`` (QT_8bit),
``METRIC_INNER_PRODUCT``, ``nprobe=64``; ``hnsw`` — ``IndexHNSWFlat``
(``efConstruction=200``, ``efSearch=64``) for the CS-subset benchmark.

Rows are added in chunk order, so FAISS row label ``i`` ↔ snapshot row ``i``
(positional id alignment; see architecture §4.1). ``FaissVectorIndex`` can map
those labels through an id array to paper ids at search time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

from .config import IndexParams, Paths

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read the manifest JSON; ``{}`` when absent (mirrors ``src.embed``)."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as fh:
        return json.load(fh)


def iter_chunk_paths(
    embeddings_dir: Path, manifest: dict[str, dict]
) -> list[tuple[int, Path]]:
    """Return ``(chunk_index, path)`` for each complete chunk, in row order.

    Chunks are ordered by ascending index so the global add order matches the
    canonical snapshot row order (row ``i`` of the index ↔ snapshot row ``i``).
    A manifest entry whose ``.npy`` file is missing is skipped.
    """
    paths: list[tuple[int, Path]] = []
    for chunk_index in sorted(manifest, key=int):
        path = embeddings_dir / f"chunk_{int(chunk_index):05d}.npy"
        if path.exists():
            paths.append((int(chunk_index), path))
    return paths


def clamp_nlist(nlist: int, n_train: int) -> int:
    """Clamp ``nlist`` to a value FAISS can train for ``n_train`` vectors.

    FAISS requires ``n_train >= nlist`` (ideally ``>= 30 * nlist``). Tiny
    synthetic corpora would otherwise hard-fail with the configured
    ``nlist=8192``; clamping to ``n_train // 4`` (minimum 1) keeps the build
    runnable at any scale.
    """
    if n_train <= 0:
        return 1
    return max(1, min(nlist, n_train // 4))


def make_index(profile: str, d: int, nlist: int | None = None) -> faiss.Index:
    """Construct an empty, untrained FAISS index for ``profile``.

    ``ivf-sq8`` → ``IndexIVFScalarQuantizer`` (QT_8bit, inner product);
    ``hnsw`` → ``IndexHNSWFlat``. Returns the raw index; the caller must
    ``train()`` IVF indexes before ``add()``.
    """
    nlist = nlist or IndexParams.NLIST
    if profile == IndexParams.PROFILE_HNSW:
        index = faiss.IndexHNSWFlat(d, 32, IndexParams.METRIC)
        index.hnsw.efConstruction = IndexParams.EF_CONSTRUCTION
        index.hnsw.efSearch = IndexParams.EF_SEARCH
        return index
    if profile != IndexParams.PROFILE_DEFAULT:
        raise ValueError(f"unknown index profile: {profile!r}")
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        d,
        nlist,
        faiss.ScalarQuantizer.QT_8bit,
        IndexParams.METRIC,
    )
    index.nprobe = IndexParams.NPROBE
    return index


def _train_sample(
    chunk_paths: list[tuple[int, Path]], train_fraction: float, d: int
) -> np.ndarray:
    """Collect a bounded training sample: ``train_fraction`` rows per chunk.

    Each chunk is loaded as a read-only memmap and sliced, so only the sample
    rows are copied into RAM (one chunk buffer at a time).
    """
    parts: list[np.ndarray] = []
    for _, path in chunk_paths:
        vectors = np.load(path, mmap_mode="r")
        take = max(1, int(vectors.shape[0] * train_fraction))
        parts.append(np.asarray(vectors[:take], dtype=np.float32))
    if not parts:
        return np.empty((0, d), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _checkpoint(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("checkpointed index (%d rows) to %s", index.ntotal, index_path)


def _search_params(profile: str) -> dict:
    if profile == IndexParams.PROFILE_HNSW:
        return {
            "efConstruction": IndexParams.EF_CONSTRUCTION,
            "efSearch": IndexParams.EF_SEARCH,
        }
    return {"nprobe": IndexParams.NPROBE}


def _write_meta(meta: dict, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    tmp.replace(meta_path)


def build_index(
    embeddings_dir: Path | None = None,
    manifest_path: Path | None = None,
    index_path: Path | None = None,
    meta_path: Path | None = None,
    profile: str | None = None,
    train_fraction: float | None = None,
    checkpoint_every: int | None = None,
    nlist: int | None = None,
) -> dict:
    """Build a FAISS index from chunked embeddings (architecture §4.3).

    Streams ``chunk_{n:05d}.npy`` files (see ``src.embed``) into
    ``index.add()`` without concatenating the full matrix; trains IVF on a
    bounded sample; checkpoints every ``checkpoint_every`` rows; writes the
    index and ``index_meta.json``. Returns the meta dict.

    Raises ``FileNotFoundError`` when the embeddings dir/manifest contains no
    complete chunks (the real embeddings may not exist yet on a fresh machine).
    """
    embeddings_dir = embeddings_dir or Paths.DATA_EMBEDDINGS
    manifest_path = manifest_path or Paths.MANIFEST
    index_path = index_path or Paths.INDEX
    meta_path = meta_path or Paths.INDEX_META
    profile = profile or IndexParams.PROFILE_DEFAULT
    train_fraction = train_fraction or IndexParams.TRAIN_FRACTION
    checkpoint_every = checkpoint_every or IndexParams.CHECKPOINT_EVERY
    nlist = nlist or IndexParams.NLIST

    chunk_paths = iter_chunk_paths(embeddings_dir, load_manifest(manifest_path))
    if not chunk_paths:
        raise FileNotFoundError(
            f"no complete embedding chunks in {embeddings_dir} "
            f"(manifest: {manifest_path}); run src.embed first"
        )

    _, first_path = chunk_paths[0]
    d = int(np.load(first_path, mmap_mode="r").shape[1])

    if profile == IndexParams.PROFILE_DEFAULT:
        sample = _train_sample(chunk_paths, train_fraction, d)
        if sample.shape[0] == 0:
            raise RuntimeError("training sample is empty; cannot build IVF index")
        index = make_index(profile, d, nlist=clamp_nlist(nlist, sample.shape[0]))
        index.train(sample)
    else:
        index = make_index(profile, d)

    n = 0
    for _, path in chunk_paths:
        vectors = np.load(path)
        index.add(vectors)
        n += int(vectors.shape[0])
        if n % checkpoint_every == 0:
            _checkpoint(index, index_path)

    _checkpoint(index, index_path)
    meta = {
        "profile": profile,
        "d": d,
        "n": n,
        "params": _search_params(profile),
    }
    _write_meta(meta, meta_path)
    logger.info("index built: profile=%s d=%d n=%d", profile, d, n)
    return meta


def load_index(index_path: Path | None = None) -> faiss.Index:
    """Load a FAISS index from disk (default ``Paths.INDEX``)."""
    return faiss.read_index(str(index_path or Paths.INDEX))


@runtime_checkable
class VectorIndex(Protocol):
    """Contract for a searchable vector index (architecture §5).

    Keeps the swap-in story (FAISS today, a vector DB later) without building
    it; ``FaissVectorIndex`` is the concrete implementation.
    """

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class FaissVectorIndex:
    """FAISS-backed ``VectorIndex`` with optional id alignment.

    Wraps a built index and an optional ``ids`` array mapping each FAISS row
    label to a paper id (row ``i`` of the index ↔ ``ids[i]``; architecture
    §4.1). ``search`` returns ``(scores, labels)`` where ``labels`` are FAISS
    row labels, mapped through ``ids`` when provided.
    """

    def __init__(self, index: faiss.Index, ids: np.ndarray | None = None) -> None:
        self.index = index
        self.ids = ids

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, labels = self.index.search(query, k)
        if self.ids is not None:
            labels = self.ids[labels]
        return scores, labels
