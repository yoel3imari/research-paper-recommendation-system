"""FAISS vector-index build/search (docs/architecture.md §4.3).

Builds a quantized index from the chunked embeddings that ``src/embed.py``
produces (one ``.npy`` per chunk and one ``manifest.json``):

1. Train on a representative sample. FAISS requires ``n_train >= nlist``,
   ideally ``>= 30 * nlist``. The default is 10% via
   ``IndexParams.TRAIN_FRACTION``.
2. Stream each chunk into ``index.add()``. Never materialize the full
   ``(N, d)`` matrix in RAM; at full corpus it costs ~3.66GB.
3. Checkpoint with ``faiss.write_index()`` every ``CHECKPOINT_EVERY`` rows.
   Index inserts are deterministic, so resumes are exact.
4. Save the index plus ``index_meta.json`` with profile, d, n, and search
   params.

**Memory approach**: the IVF training sample has at most ``train_fraction``
rows per chunk. Each chunk loads as a memmap through ``np.load(..., mmap_mode)``
and slices to the sample rows. Peak memory is one chunk buffer plus the train
fraction of the corpus, never the full matrix. ``clamp_nlist`` clamps
``nlist`` to ``n_train // 4``, so tiny synthetic corpora train without hard
failure.

**Profiles**: ``ivf-sq8`` (default) uses ``IndexIVFScalarQuantizer`` with
``QT_8bit``, ``METRIC_INNER_PRODUCT``, and ``nprobe=64``. ``hnsw`` uses
``IndexHNSWFlat`` with ``efConstruction=200`` and ``efSearch=64`` for the
CS-subset benchmark.

Rows are added in chunk order. FAISS row label ``i`` then corresponds to
snapshot row ``i`` (positional id alignment; see architecture §4.1). At
search time, ``FaissVectorIndex`` maps those labels through an id array to
paper ids.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

from .config import IndexParams, Paths
from .embed import load_manifest

logger = logging.getLogger(__name__)


def iter_chunk_paths(
    embeddings_dir: Path, manifest: dict[str, dict]
) -> list[tuple[int, Path]]:
    """Returns ``(chunk_index, path)`` for each complete chunk, in row order.

    Chunks appear in ascending index order. The global add order then matches
    the canonical snapshot row order; row ``i`` of the index corresponds to
    snapshot row ``i``. The function skips a manifest entry whose ``.npy``
    file is missing.
    """
    paths: list[tuple[int, Path]] = []
    for chunk_index in sorted(manifest, key=int):
        path = embeddings_dir / f"chunk_{int(chunk_index):05d}.npy"
        if path.exists():
            paths.append((int(chunk_index), path))
    return paths


def clamp_nlist(nlist: int, n_train: int) -> int:
    """Clamps ``nlist`` to a value that satisfies ``n_train >= nlist``.

    FAISS requires ``n_train >= nlist``, ideally ``>= 30 * nlist``. Tiny
    synthetic corpora hard-fail with the configured ``nlist=8192``. The clamp
    to ``n_train // 4``, with a minimum of 1, keeps the build runnable at any
    scale.
    """
    if n_train <= 0:
        return 1
    return max(1, min(nlist, n_train // 4))


def make_index(profile: str, d: int, nlist: int | None = None) -> faiss.Index:
    """Constructs an empty, untrained FAISS index for ``profile``.

    ``ivf-sq8`` produces an ``IndexIVFScalarQuantizer`` (QT_8bit, inner
    product). ``hnsw`` produces an ``IndexHNSWFlat``. Returns the raw index.
    Train IVF indexes before ``add()``.
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
    """Collects a bounded training sample: ``train_fraction`` rows per chunk.

    Loads each chunk as a read-only memmap and slices it. Copies only the
    sample rows into RAM, one chunk buffer at a time.
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
    """Builds a FAISS index from chunked embeddings (architecture §4.3).

    Streams ``chunk_{n:05d}.npy`` files (see ``src.embed``) into ``index.add()``.
    It does not concatenate the full matrix. Trains IVF on a bounded sample.
    Checkpoints every ``checkpoint_every`` rows. Writes the index and
    ``index_meta.json``. Returns the meta dict.

    When the embeddings dir/manifest contains no complete chunks, raises
    ``FileNotFoundError``. On a fresh machine, the real embeddings do not
    exist yet.
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
    """Loads a FAISS index from disk. Defaults to ``Paths.INDEX``."""
    return faiss.read_index(str(index_path or Paths.INDEX))


@runtime_checkable
class VectorIndex(Protocol):
    """Contract for a searchable vector index (architecture §5).

    Enables a future swap-in, FAISS today and a vector DB later. The protocol
    does not build the swap-in. ``FaissVectorIndex`` is the concrete
    implementation.
    """

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class FaissVectorIndex:
    """FAISS-backed ``VectorIndex`` with optional id alignment.

    Wraps a built index and an optional ``ids`` array. The ``ids`` array maps
    each FAISS row label to a paper id; row ``i`` of the index corresponds to
    ``ids[i]`` (architecture §4.1). ``search`` returns ``(scores, labels)``,
    where ``labels`` are FAISS row labels. When ``ids`` is provided, the
    function maps ``labels`` through ``ids``.
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
