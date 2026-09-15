"""Disk-backed, sharded latent cache used by QuRO's offline/online boundary.

The cache stores one fixed-size tensor ``(m, h)`` per document.  It deliberately
does not know how the tensor was produced: PISCO, COCOM, or a local prototype can
all feed the same writer.  The online readout therefore depends only on this
stable interface rather than on a compressor implementation.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheMetadata:
    compressor: str
    latent_size: int
    hidden_size: int
    dtype: str
    format_version: int = CACHE_FORMAT_VERSION
    # Offline-compression provenance.  ``compr_rate`` and ``doc_max_length`` are
    # what make xi_off computable after the fact: PISCO truncates every document
    # at doc_max_length, so the ratio must be taken against the tokens the
    # compressor actually consumed, never against the untruncated document.
    compr_rate: Optional[int] = None
    doc_max_length: Optional[int] = None
    checkpoint: Optional[str] = None


def _safe_torch_load(path: str, map_location="cpu"):
    """Use the safer loader when supported, while retaining torch 2.0 support."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.1
        return torch.load(path, map_location=map_location)


class LatentCacheWriter:
    """Incrementally write document latents into bounded-size ``.pt`` shards."""

    def __init__(
        self,
        root: str,
        metadata: CacheMetadata,
        shard_size: int = 1024,
        overwrite: bool = False,
        resume: bool = False,
    ):
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.root = os.path.abspath(root)
        self.metadata = metadata
        self.shard_size = shard_size
        manifest_path = os.path.join(self.root, "manifest.json")
        restored: Dict[str, object] = {}
        if os.path.exists(manifest_path):
            if resume:
                # Compressing a large corpus takes hours; losing it to one crash
                # is not acceptable, so an interrupted run can be continued.
                with open(manifest_path, encoding="utf-8") as f:
                    restored = json.load(f)
                for field in ("compressor", "latent_size", "hidden_size", "dtype"):
                    if restored.get(field) != getattr(metadata, field):
                        raise ValueError(
                            f"cannot resume {manifest_path}: {field} is "
                            f"{restored.get(field)!r}, expected {getattr(metadata, field)!r}")
            elif not overwrite:
                raise FileExistsError(f"cache already exists: {manifest_path}")
        os.makedirs(self.root, exist_ok=True)
        self._ids: List[str] = []
        self._latents: List[torch.Tensor] = []
        self._token_counts: List[int] = []
        self._original_counts: List[int] = []
        self._documents: Dict[str, Dict[str, int]] = dict(restored.get("documents", {}))
        self._shards: List[Dict[str, object]] = list(restored.get("shards", []))

    @property
    def cached_ids(self) -> "set[str]":
        """IDs already on disk; a resumed run skips re-compressing them."""
        return set(self._documents)

    def add(self, doc_id: str, latents: torch.Tensor, source_token_count: int,
            original_token_count: Optional[int] = None) -> None:
        """Store one document latent.

        ``source_token_count`` is what the compressor consumed (post-truncation)
        and is the denominator for xi_off; ``original_token_count`` records how
        long the document really was, so truncation loss stays visible.
        """
        doc_id = str(doc_id)
        if not doc_id:
            raise ValueError("doc_id must be non-empty")
        if doc_id in self._documents or doc_id in self._ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        expected = (self.metadata.latent_size, self.metadata.hidden_size)
        if tuple(latents.shape) != expected:
            raise ValueError(f"{doc_id}: expected latent shape {expected}, got {tuple(latents.shape)}")
        if not torch.isfinite(latents).all():
            raise ValueError(f"{doc_id}: latent contains NaN or Inf")
        self._ids.append(doc_id)
        self._latents.append(latents.detach().to(device="cpu", dtype=getattr(torch, self.metadata.dtype)))
        self._token_counts.append(int(source_token_count))
        self._original_counts.append(int(source_token_count if original_token_count is None
                                         else original_token_count))
        if len(self._ids) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._ids:
            return
        shard_idx = len(self._shards)
        filename = f"shard-{shard_idx:05d}.pt"
        payload = {
            "doc_ids": list(self._ids),
            "latents": torch.stack(self._latents, dim=0),
            "source_token_counts": torch.tensor(self._token_counts, dtype=torch.long),
            "original_token_counts": torch.tensor(self._original_counts, dtype=torch.long),
        }
        tmp = os.path.join(self.root, filename + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(self.root, filename))
        for row, doc_id in enumerate(self._ids):
            self._documents[doc_id] = {
                "shard": shard_idx,
                "row": row,
                "source_token_count": self._token_counts[row],
                "original_token_count": self._original_counts[row],
            }
        self._shards.append({"file": filename, "count": len(self._ids)})
        self._ids, self._latents = [], []
        self._token_counts, self._original_counts = [], []

    def close(self) -> str:
        self.flush()
        manifest = {
            **asdict(self.metadata),
            "shard_size": self.shard_size,
            "num_documents": len(self._documents),
            "shards": self._shards,
            "documents": self._documents,
        }
        path = os.path.join(self.root, "manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()


class LatentCache:
    """Read document latents lazily and batch them as ``(B, K, m, h)``."""

    def __init__(self, root: str, max_open_shards: int = 2):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        version = int(self.manifest.get("format_version", -1))
        if version != CACHE_FORMAT_VERSION:
            raise ValueError(f"unsupported cache format {version}; expected {CACHE_FORMAT_VERSION}")
        self.metadata = CacheMetadata(
            compressor=self.manifest["compressor"],
            latent_size=int(self.manifest["latent_size"]),
            hidden_size=int(self.manifest["hidden_size"]),
            dtype=self.manifest["dtype"],
            format_version=version,
            compr_rate=self.manifest.get("compr_rate"),
            doc_max_length=self.manifest.get("doc_max_length"),
            checkpoint=self.manifest.get("checkpoint"),
        )
        self.documents: Dict[str, Dict[str, int]] = self.manifest["documents"]
        self.shards: List[Dict[str, object]] = self.manifest["shards"]
        self.max_open_shards = max(1, int(max_open_shards))
        self._loaded: "OrderedDict[int, Dict[str, object]]" = OrderedDict()

    def __contains__(self, doc_id: str) -> bool:
        return str(doc_id) in self.documents

    def __len__(self) -> int:
        return len(self.documents)

    def _load_shard(self, index: int) -> Dict[str, object]:
        if index in self._loaded:
            value = self._loaded.pop(index)
            self._loaded[index] = value
            return value
        filename = str(self.shards[index]["file"])
        value = _safe_torch_load(os.path.join(self.root, filename), map_location="cpu")
        self._loaded[index] = value
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return value

    def get(self, doc_id: str) -> Tuple[torch.Tensor, int]:
        key = str(doc_id)
        if key not in self.documents:
            raise KeyError(f"document is absent from latent cache: {key}")
        loc = self.documents[key]
        shard = self._load_shard(int(loc["shard"]))
        row = int(loc["row"])
        return shard["latents"][row], int(loc.get("source_token_count", 0))

    def get_many(
        self,
        batch_doc_ids: Sequence[Sequence[str]],
        max_docs: Optional[int] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``latents, document_mask, source_token_counts``.

        ``latents`` has shape ``(B, K, m, h)``.  Missing padding positions are
        zeros and marked false in ``document_mask``; unknown non-padding IDs are
        errors because silently dropping retrieved evidence invalidates metrics.
        """
        rows = [list(x[:max_docs] if max_docs is not None else x) for x in batch_doc_ids]
        batch_size = len(rows)
        k = max(1, max((len(x) for x in rows), default=0))
        m, h = self.metadata.latent_size, self.metadata.hidden_size
        out_dtype = dtype or getattr(torch, self.metadata.dtype)
        latents = torch.zeros(batch_size, k, m, h, dtype=out_dtype)
        doc_mask = torch.zeros(batch_size, k, dtype=torch.bool)
        token_counts = torch.zeros(batch_size, k, dtype=torch.long)
        for i, doc_ids in enumerate(rows):
            for j, doc_id in enumerate(doc_ids):
                value, n_tokens = self.get(doc_id)
                latents[i, j] = value.to(dtype=out_dtype)
                doc_mask[i, j] = True
                token_counts[i, j] = n_tokens
        if device is not None:
            latents = latents.to(device)
            doc_mask = doc_mask.to(device)
            token_counts = token_counts.to(device)
        return latents, doc_mask, token_counts


def import_tensor_records(
    records: Iterable[Tuple[str, torch.Tensor, int]],
    root: str,
    compressor: str,
    dtype: str = "float16",
    shard_size: int = 1024,
    overwrite: bool = False,
) -> str:
    """Import latents exported by PISCO/COCOM or another external compressor."""
    iterator = iter(records)
    try:
        first = next(iterator)
    except StopIteration as e:
        raise ValueError("cannot build an empty latent cache") from e
    doc_id, tensor, n_tokens = first
    if tensor.ndim != 2:
        raise ValueError(f"expected each latent tensor to be rank 2, got {tensor.shape}")
    metadata = CacheMetadata(compressor, tensor.size(0), tensor.size(1), dtype)
    with LatentCacheWriter(root, metadata, shard_size=shard_size, overwrite=overwrite) as writer:
        writer.add(doc_id, tensor, n_tokens)
        for doc_id, tensor, n_tokens in iterator:
            writer.add(doc_id, tensor, n_tokens)
    return os.path.join(os.path.abspath(root), "manifest.json")
