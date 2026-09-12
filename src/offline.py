"""Adapters for frozen offline compressors.

QuRO intentionally treats PISCO/COCOM as replaceable producers of document
latents.  Their research repositories do not expose one stable Python API, so
v0.0 uses a small callable contract instead of importing private internals here.
An adapter factory receives ``checkpoint`` and ``device`` and returns either a
callable or an object with ``encode_texts``.  The result must be ``(B, m, h)``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


class FrozenCompressorAdapter:
    """Normalize an external compressor to ``encode_texts -> (B,m,h)``."""

    def __init__(self, backend: Any, name: str):
        self.backend = backend
        self.name = name
        if isinstance(backend, torch.nn.Module):
            backend.eval()
            for parameter in backend.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        fn = getattr(self.backend, "encode_texts", None)
        if fn is None and callable(self.backend):
            fn = self.backend
        if fn is None:
            raise TypeError("compressor adapter must be callable or define encode_texts(texts)")
        value = fn(list(texts))
        if isinstance(value, dict):
            value = value.get("latents", value.get("embeddings"))
        if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
            value = value[0]
        if not torch.is_tensor(value) or value.ndim != 3:
            raise TypeError("compressor must return a tensor shaped (batch, latent_size, hidden_size)")
        if value.size(0) != len(texts):
            raise ValueError(f"compressor returned batch {value.size(0)} for {len(texts)} texts")
        if not torch.isfinite(value).all():
            raise ValueError("compressor returned NaN or Inf")
        return value.detach()


def load_adapter(
    factory_spec: str,
    checkpoint: Optional[str],
    device: str,
    factory_kwargs: Optional[Dict[str, Any]] = None,
) -> FrozenCompressorAdapter:
    """Load ``module.path:factory`` without coupling QuRO to a research repo.

    Example factory in a local PISCO integration module::

        def build(checkpoint, device, **kwargs):
            model = PISCO.from_pretrained(checkpoint).to(device).eval()
            return MyPiscoLatentExporter(model)
    """
    if ":" not in factory_spec:
        raise ValueError("factory must have form 'module.path:callable_name'")
    module_name, attr = factory_spec.split(":", 1)
    factory: Callable[..., Any] = getattr(importlib.import_module(module_name), attr)
    backend = factory(checkpoint=checkpoint, device=device, **(factory_kwargs or {}))
    return FrozenCompressorAdapter(backend, name=factory_spec)
