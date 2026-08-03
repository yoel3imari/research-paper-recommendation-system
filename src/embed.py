"""Chunked embedding pipeline (docs/architecture.md §4.2).

Encodes the snapshot into L2-normalized 384-dim vectors: each row contributes
``title + " " + abstract``. Output is one ``.npy`` per fixed chunk plus a
``manifest.json`` recording ``{n: {start, end, rows, shape}}`` per chunk,
enabling crash-safe resume — a chunk present in the manifest with a matching
shape is complete and is skipped on rerun (atomic chunks; no per-id
bookkeeping).

The encoder sits behind an ``Encoder`` protocol so unit tests substitute a
stub. The real ``OnnxEncoder`` (dynamic UINT8 ONNX via a bare
``onnxruntime.InferenceSession``) loads its heavy deps only on instantiation,
never at import time, so importing this module performs no network access and
loads no model.

The encoder runs ONNX directly through ``onnxruntime`` rather than via
``optimum``. ``optimum[onnxruntime]``'s lazy-import structure is incompatible
with the pinned wheel set on this Intel Mac (it imports ``optimum.exporter.
onnx.main_export`` at top level, which the installed ``optimum``/``optimum-onnx``
versions do not provide), so depending on it breaks the whole runtime path.
The mean-pooled ``last_hidden_state`` we need is exactly what the BERT ONNX
graph outputs, so no exporter machinery is required.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

from .config import ModelConfig, Paths

logger = logging.getLogger(__name__)

EPSILON = 1e-12


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows (cosine similarity via inner product)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, EPSILON)


def build_texts(df: pl.DataFrame) -> list[str]:
    """Compose ``title + " " + abstract`` per row (architecture §4.2)."""
    return (df["title"].fill_null("") + " " + df["abstract"].fill_null("")).to_list()


def iter_chunks(df: pl.DataFrame, chunk_size: int):
    """Yield ``(start_row, df_slice)`` for each fixed-size chunk of the snapshot.

    ``start`` is the absolute row index in the canonical snapshot (row ``i`` of
    every artifact corresponds to snapshot row ``i``), so the ``chunk_{n}.npy``
    outputs stay positionally aligned with the snapshot.
    """
    for start in range(0, df.height, chunk_size):
        yield start, df.slice(start, chunk_size)


@runtime_checkable
class Encoder(Protocol):
    """Any object that turns texts into an L2-normalized ``(N, dim)`` matrix."""

    dim: int

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray: ...


class OnnxEncoder:
    """Real encoder: tokenizer + dynamic UINT8 ONNX (all-MiniLM-L6-v2).

    Heavy imports live in ``__init__`` only (instantiation), so tests that
    never construct this object stay free of the model stack and of network.
    """

    def __init__(
        self,
        model_dir: Path = Paths.MODELS / ModelConfig.NAME,
        onnx_file: str = ModelConfig.ONNX_FILE,
        max_seq_length: int = ModelConfig.MAX_SEQ_LENGTH,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.dim = ModelConfig.DIM
        self._max_seq_length = max_seq_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._session = ort.InferenceSession(
            model_dir / onnx_file,
            providers=["CPUExecutionProvider"],
        )

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                truncation=True,
                max_length=self._max_seq_length,
                padding=True,
                return_tensors="np",
            )
            feed = {name: np.asarray(value) for name, value in encoded.items()}
            last_hidden = np.asarray(self._session.run(None, feed)[0])
            outputs.append(l2_normalize(last_hidden.mean(axis=1)))
        return np.concatenate(outputs, axis=0)


def encode_chunk(
    texts: list[str],
    encoder: Encoder,
    batch_size: int = ModelConfig.BATCH_SIZE,
) -> np.ndarray:
    """Encode one chunk. Result is L2-normalized ``(len(texts), dim)``."""
    return encoder.encode_texts(texts, batch_size=batch_size)


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read the manifest JSON; ``{}`` when absent."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as fh:
        return json.load(fh)


def save_manifest(manifest: dict[str, dict], manifest_path: Path) -> None:
    """Atomically write the manifest (tmp + rename avoids torn writes)."""
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    tmp.replace(manifest_path)


def chunk_is_complete(
    manifest: dict[str, dict],
    chunk_index: int,
    rows: int,
    dim: int,
) -> bool:
    """A chunk is complete iff recorded with matching ``rows`` and ``shape``.

    ``rows`` must equal the recorded row count and ``shape`` must equal
    ``(rows, dim)``; any mismatch means the prior ``.npy`` write was torn or
    truncated and the chunk must be re-encoded.
    """
    entry = manifest.get(str(chunk_index))
    if entry is None:
        return False
    return entry.get("rows") == rows and entry.get("shape") == [rows, dim]


def write_chunk(
    vectors: np.ndarray,
    embeddings_dir: Path,
    chunk_index: int,
    start_row: int,
    manifest: dict[str, dict],
    manifest_path: Path,
) -> None:
    """Write ``chunk_{n:05d}.npy`` and record its extent in the manifest."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    artifact = embeddings_dir / f"chunk_{chunk_index:05d}.npy"
    np.save(artifact, vectors)
    manifest[str(chunk_index)] = {
        "start": start_row,
        "end": start_row + vectors.shape[0],
        "rows": vectors.shape[0],
        "shape": list(vectors.shape),
    }
    save_manifest(manifest, manifest_path)
    logger.info("wrote %s (%d rows)", artifact.name, vectors.shape[0])


def run(
    encoder: Encoder,
    snapshot_path: Path | None = None,
    embeddings_dir: Path | None = None,
    manifest_path: Path | None = None,
    chunk_size: int | None = None,
    batch_size: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Encode the snapshot in chunks with crash-safe resume.

    Returns a summary dict (``chunks_written``, ``chunks_skipped``). When
    ``resume`` is true, chunks already complete in the manifest are skipped.
    """
    snapshot_path = snapshot_path or Paths.SNAPSHOT
    embeddings_dir = embeddings_dir or Paths.DATA_EMBEDDINGS
    manifest_path = manifest_path or Paths.MANIFEST
    chunk_size = chunk_size or ModelConfig.CHUNK_SIZE
    batch_size = batch_size or ModelConfig.BATCH_SIZE

    df = pl.read_parquet(snapshot_path)
    manifest = load_manifest(manifest_path) if resume else {}
    written = 0
    skipped = 0

    for start, chunk in iter_chunks(df, chunk_size):
        chunk_index = start // chunk_size
        rows = chunk.height
        if chunk_is_complete(manifest, chunk_index, rows, encoder.dim):
            skipped += 1
            continue
        texts = build_texts(chunk)
        vectors = encode_chunk(texts, encoder, batch_size)
        write_chunk(
            vectors,
            embeddings_dir,
            chunk_index,
            start,
            manifest,
            manifest_path,
        )
        written += 1

    logger.info("embed complete: %d written, %d skipped", written, skipped)
    return {"chunks_written": written, "chunks_skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed the snapshot (docs/architecture.md §4.2)."
    )
    parser.add_argument("--snapshot", default=str(Paths.SNAPSHOT), help="snapshot parquet")
    parser.add_argument("--embeddings-dir", default=str(Paths.DATA_EMBEDDINGS))
    parser.add_argument("--no-resume", action="store_true", help="ignore the manifest on rerun")
    parser.add_argument("--chunk-size", type=int, help="rows per chunk (default: config)")
    parser.add_argument("--batch-size", type=int, help="encode batch size (default: config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    encoder = OnnxEncoder()
    summary = run(
        encoder,
        snapshot_path=Path(args.snapshot),
        embeddings_dir=Path(args.embeddings_dir),
        resume=not args.no_resume,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
    print(f"embed complete: {summary}")


if __name__ == "__main__":
    main()