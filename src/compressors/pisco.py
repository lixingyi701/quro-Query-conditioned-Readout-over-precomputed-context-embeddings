"""Frozen PISCO / COCOM compressors, exposed through QuRO's ``encode_texts`` contract.

Both checkpoints share one architecture (``third_party/modelling_pisco.py``): the
Mistral decoder compresses a document into ``m = doc_max_length // compr_rate``
memory embeddings that live in the decoder's own hidden space.  Two consequences
drive the rest of QuRO:

* latents are ``(m, 4096)`` in Mistral space, so the readout can bridge straight
  back into the generator without a cross-model projection;
* the very same object is also the generator (see :func:`build_generator`), so a
  PISCO baseline and QuRO differ only in how the latents reach the decoder.

The published checkpoints store only adapters plus the resized first/last layers
and pull the 14 GB base model from the Hub.  :func:`ensure_local_base` rewrites
``decoder_model_name`` in the downloaded ``config.json`` to the local Mistral
copy so nothing is re-downloaded and the loader works offline.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src import paths

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def ensure_local_base(checkpoint_dir: str, base_model: Optional[str] = None) -> str:
    """Point a downloaded PISCO/COCOM config at the local Mistral checkpoint.

    Idempotent, and a no-op when ``decoder_model_name`` already resolves on disk.
    The original value is preserved under ``decoder_model_name_original`` so the
    provenance of a cache stays auditable.
    """
    base_model = base_model or paths.MISTRAL_PATH
    config_path = os.path.join(checkpoint_dir, "config.json")
    paths.require(config_path, "compressor config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    current = config.get("decoder_model_name")
    if current and os.path.isdir(current):
        return current
    paths.require(base_model, "local Mistral base model")
    config.setdefault("decoder_model_name_original", current)
    config["decoder_model_name"] = base_model
    if config.get("compr_base_model_name") and not os.path.isdir(config["compr_base_model_name"]):
        config.setdefault("compr_base_model_name_original", config["compr_base_model_name"])
        config["compr_base_model_name"] = base_model
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    os.replace(tmp, config_path)
    return base_model


def load_cocom(checkpoint: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load a frozen PISCO/COCOM checkpoint from the vendored implementation."""
    paths.require(checkpoint, "compressor checkpoint")
    ensure_local_base(checkpoint)

    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    # PISCO ships modelling_pisco.py; the older COCOM v1 release ships modeling_cocom.py.
    if os.path.exists(os.path.join(checkpoint, "modelling_pisco.py")):
        from third_party.modelling_pisco import COCOM
    elif os.path.exists(os.path.join(checkpoint, "modeling_cocom.py")):
        from third_party.modeling_cocom import COCOM
    else:
        raise FileNotFoundError(
            f"{checkpoint} contains neither modelling_pisco.py nor modeling_cocom.py")

    model = COCOM.from_pretrained(checkpoint)
    model = model.to(device=device, dtype=DTYPES[dtype])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.compressor_id = f"{os.path.basename(checkpoint.rstrip('/'))}:rate{raw.get('compr_rate')}"
    return model


class FrozenDocumentCompressor:
    """Adapter exposing one frozen compressor as ``encode_texts -> (B, m, h)``."""

    def __init__(self, model, checkpoint: str):
        self.model = model
        self.checkpoint = checkpoint
        self.name = getattr(model, "compressor_id", os.path.basename(checkpoint))
        self.doc_max_length = int(getattr(model, "doc_max_length", model.config.doc_max_length))
        self.compr_rate = int(model.config.compr_rate)
        self.latent_size = self.doc_max_length // self.compr_rate
        self.hidden_size = int(model.decoder.config.hidden_size)
        self.tokenizer = model.decoder_tokenizer

    @property
    def device(self):
        return self.model.decoder.device

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        latents = self.model.compress_documents(list(texts))
        if latents.shape[1:] != (self.latent_size, self.hidden_size):
            raise ValueError(
                f"{self.name} returned {tuple(latents.shape)}, "
                f"expected (*, {self.latent_size}, {self.hidden_size})")
        return latents.detach()

    def token_stats(self, texts: Sequence[str]) -> List[Dict[str, int]]:
        """Tokens the compressor actually consumed, and the untruncated length.

        PISCO hard-truncates every document at ``doc_max_length`` (128), so the
        offline compression ratio must be computed against ``fed_tokens``.
        Reporting ``original_tokens`` keeps the truncation loss visible instead of
        silently inflating the claimed ratio.
        """
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [{"fed_tokens": min(len(ids), self.doc_max_length),
                 "original_tokens": len(ids)} for ids in encoded]

    def metadata(self) -> Dict[str, object]:
        return {
            "compressor": self.name,
            "checkpoint": self.checkpoint,
            "compr_rate": self.compr_rate,
            "doc_max_length": self.doc_max_length,
            "latent_size": self.latent_size,
            "hidden_size": self.hidden_size,
        }


def build(checkpoint: Optional[str] = None, device: str = "cuda",
          dtype: str = "bfloat16", **_) -> FrozenDocumentCompressor:
    """Factory for ``scripts/build_latent_cache.py --adapter src.compressors.pisco:build``."""
    checkpoint = checkpoint or paths.PISCO_MISTRAL
    return FrozenDocumentCompressor(load_cocom(checkpoint, device, dtype), checkpoint)


def build_generator(checkpoint: Optional[str] = None, device: str = "cuda",
                    dtype: str = "bfloat16"):
    """Load the same checkpoint to be used as QuRO's generator.

    Returns the full ``COCOM`` object: ``.decoder`` is Mistral plus the trained
    ``decoder_adapter``, with the embedding table already resized for the
    ``<MEM*>/<AE>/<ENC>/<SEP>`` tokens.  QuRO reuses it so that the PISCO baseline
    and QuRO share backbone, prompt and LoRA initialisation exactly.
    """
    checkpoint = checkpoint or paths.PISCO_MISTRAL
    return load_cocom(checkpoint, device, dtype)
