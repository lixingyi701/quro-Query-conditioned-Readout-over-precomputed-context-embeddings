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


def load_cocom(checkpoint: str, device: str = "cuda", dtype: str = "bfloat16",
               attn_implementation: Optional[str] = None):
    """Load a frozen PISCO/COCOM checkpoint from the vendored implementation.

    ``attn_implementation`` is normally left alone, so the decoder runs on
    whatever backend transformers selects.  The attention diagnostics
    (``src/infeasibility.py``) need ``"eager"``, because FlashAttention and the
    fused SDPA kernels never materialise the attention weights -- let alone the
    pre-softmax QK logits -- so there is nothing to read out of them.
    """
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

    kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = COCOM.from_pretrained(checkpoint, **kwargs)
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
                    dtype: str = "bfloat16",
                    attn_implementation: Optional[str] = None):
    """Load the same checkpoint to be used as QuRO's generator.

    Returns the full ``COCOM`` object: ``.decoder`` is Mistral plus the trained
    ``decoder_adapter``, with the embedding table already resized for the
    ``<MEM*>/<AE>/<ENC>/<SEP>`` tokens.  QuRO reuses it so that the PISCO baseline
    and QuRO share backbone, prompt and LoRA initialisation exactly.
    """
    checkpoint = checkpoint or paths.PISCO_MISTRAL
    return load_cocom(checkpoint, device, dtype, attn_implementation)


class CocomV1Compressor(FrozenDocumentCompressor):
    """COCOM v1 behind the same ``encode_texts -> (B, m, h)`` contract as PISCO.

    Two differences from PISCO have to be handled explicitly, and both would
    silently corrupt the cache if ignored.

    *The memory-token count is not fixed.*  ``modeling_cocom.py`` derives it from
    the padded batch length, so ``m`` would vary with whichever document in a
    batch happens to be longest -- fatal for a fixed-shape cache, and it would
    also make the offline compression ratio meaningless.  Documents are therefore
    padded and truncated to exactly ``doc_max_length`` tokens, giving a constant
    ``m = doc_max_length // compr_rate``.  Holding the fed length at PISCO's 128
    is also what makes the two comparable: same tokens in, different ``m`` out, so
    a sweep over ``m`` is a sweep over the offline compression ratio alone.

    *The memory tokens sit on the other side.*  PISCO appends them
    (``modelling_pisco.py:393``), COCOM v1 prepends them
    (``modeling_cocom.py:283``).  For a causal decoder that is not a cosmetic
    difference: prepended memory tokens precede the document and cannot attend to
    it, so their hidden states would be identical for every input.  ``mem_side``
    is therefore configurable and :func:`probe_mem_side` decides it by measurement
    rather than by reading the code.
    """

    def __init__(self, model, checkpoint: str, doc_max_length: int = 128,
                 mem_side: str = "append"):
        self.model = model
        self.checkpoint = checkpoint
        self.name = getattr(model, "compressor_id", os.path.basename(checkpoint))
        self.doc_max_length = int(doc_max_length)
        self.compr_rate = int(model.config.compr_rate)
        self.latent_size = self.doc_max_length // self.compr_rate
        self.hidden_size = int(model.decoder.config.hidden_size)
        self.tokenizer = model.decoder_tokenizer
        self.mem_side = mem_side
        self.mem_token_id = int(self.tokenizer.convert_tokens_to_ids("<MEM>"))

    def _encoder_inputs(self, texts: Sequence[str]):
        tok = self.tokenizer
        wrapped = [f"{tok.enc_token}{tok.bos_token}{text}{tok.eos_token}" for text in texts]
        encoded = tok(wrapped, return_tensors="pt", padding="max_length",
                      max_length=self.doc_max_length + 3, truncation=True,
                      add_special_tokens=False)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        mem = torch.full((ids.size(0), self.latent_size), self.mem_token_id, dtype=torch.long)
        ones = torch.ones_like(mem)
        if self.mem_side == "append":
            ids, mask = torch.cat([ids, mem], 1), torch.cat([mask, ones], 1)
        else:
            ids, mask = torch.cat([mem, ids], 1), torch.cat([ones, mask], 1)
        device = self.model.decoder.device
        return ids.to(device), mask.to(device)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        ids, mask = self._encoder_inputs(list(texts))
        latents = self.model.compr_decoder(ids, mask)
        if latents.shape[1:] != (self.latent_size, self.hidden_size):
            raise ValueError(
                f"{self.name} returned {tuple(latents.shape)}, "
                f"expected (*, {self.latent_size}, {self.hidden_size})")
        return latents.detach()

    def metadata(self) -> Dict[str, object]:
        out = super().metadata()
        out["mem_side"] = self.mem_side
        return out


@torch.no_grad()
def probe_mem_side(compressor: "CocomV1Compressor", texts: Sequence[str]) -> Dict[str, object]:
    """Decide where the memory tokens belong by compressing real documents.

    A prepended memory token precedes the document under a causal mask, so its
    hidden state cannot depend on the document: the tell-tale is that two
    different documents compress to the same latents.  ``between_doc_distance``
    near zero means that layout is inert and must not be used to build a cache.
    """
    report = {}
    for side in ("append", "prepend"):
        compressor.mem_side = side
        latents = compressor.encode_texts(texts).float()
        flat = latents.reshape(latents.size(0), -1)
        pairwise = torch.cdist(flat, flat)
        off = ~torch.eye(flat.size(0), dtype=torch.bool, device=flat.device)
        report[side] = {
            "between_doc_distance": round(pairwise[off].mean().item(), 4),
            "latent_std": round(latents.std().item(), 4),
            "latent_absmax": round(latents.abs().max().item(), 3),
        }
    usable = [s for s in ("append", "prepend") if report[s]["between_doc_distance"] > 1e-3]
    report["verdict"] = usable[0] if len(usable) == 1 else (usable or ["none"])[0]
    report["note"] = ("both layouts vary with the document; pick by downstream quality"
                      if len(usable) == 2 else "only one layout carries document information")
    return report


def build_cocom(checkpoint: Optional[str] = None, device: str = "cuda",
                dtype: str = "bfloat16", doc_max_length: int = 128,
                mem_side: str = "append", **_) -> CocomV1Compressor:
    """Factory for ``--adapter src.compressors.pisco:build_cocom``."""
    checkpoint = checkpoint or os.path.join(paths.MODELS_DIR, "cocom-v1-4-mistral-7b")
    model = load_cocom(checkpoint, device, dtype)
    return CocomV1Compressor(model, checkpoint, doc_max_length, mem_side)


class ChunkedPiscoCompressor(FrozenDocumentCompressor):
    """PISCO at a finer granularity: ``n_chunks`` independent compressions per document.

    PISCO's ``m`` is not a knob.  ``n_mem_tokens = doc_max_length // compr_rate``
    is pinned at 8 by the eight *trained* ``<MEM0..7>`` embeddings, and
    ``add_memory_tokens_to_inputs`` asserts the count matches, so raising
    ``compr_rate`` only trips the assertion -- ``<MEM8>`` does not exist and was
    never trained.  Feeding longer documents does move the offline ratio, but in
    the wrong direction: ``m`` stays 8, so ``K * m`` is unchanged and the online
    selection problem is exactly as hard.

    Splitting the document instead gives the missing axis.  Four 32-token chunks
    at PISCO's own rate yield ``4 * 8 = 32`` latents over the same 128 fed tokens,
    i.e. the same offline ratio as COCOM rate 4 while holding the compressor
    fixed.  The two routes to ``m = 32`` confound different things -- COCOM
    changes the checkpoint, chunking removes cross-chunk context -- so agreement
    between them is what makes a conclusion about ``m`` trustworthy.
    """

    def __init__(self, model, checkpoint: str, n_chunks: int = 4, doc_max_length: int = 128):
        super().__init__(model, checkpoint)
        self.n_chunks = int(n_chunks)
        self.doc_max_length = int(doc_max_length)
        self.chunk_length = self.doc_max_length // self.n_chunks
        self.per_chunk_latents = int(len(self.tokenizer.mem_tokens))
        self.latent_size = self.per_chunk_latents * self.n_chunks
        self.name = f"{self.name}:chunk{self.n_chunks}"

    def _chunk_texts(self, texts: Sequence[str]) -> List[str]:
        """Split at token level, then decode: chunk borders must be token borders."""
        encoded = self.tokenizer(list(texts), add_special_tokens=False,
                                 truncation=True, max_length=self.doc_max_length)["input_ids"]
        chunks = []
        for ids in encoded:
            for c in range(self.n_chunks):
                piece = ids[c * self.chunk_length : (c + 1) * self.chunk_length]
                chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True)
                              if piece else "")
        return chunks

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        chunks = self._chunk_texts(texts)
        # max_length is the *fed* length, independent of the memory-token count,
        # so a 32-token chunk pads to 35 instead of PISCO's usual 131.
        inputs = self.model.prepare_encoder_inputs(chunks, max_length=self.chunk_length)
        device = self.model.decoder.device
        latents = self.model.compress(
            enc_input_ids=inputs["input_ids"].to(device),
            enc_attention_mask=inputs["attention_mask"].to(device))
        latents = latents.reshape(len(texts), self.latent_size, self.hidden_size)
        if latents.shape[1:] != (self.latent_size, self.hidden_size):
            raise ValueError(f"{self.name} returned {tuple(latents.shape)}")
        return latents.detach()

    def metadata(self) -> Dict[str, object]:
        out = super().metadata()
        out.update({"latent_size": self.latent_size, "n_chunks": self.n_chunks,
                    "chunk_length": self.chunk_length,
                    "effective_compr_rate": self.doc_max_length / self.latent_size})
        return out


def build_chunked(checkpoint: Optional[str] = None, device: str = "cuda",
                  dtype: str = "bfloat16", n_chunks: int = 4, **_) -> ChunkedPiscoCompressor:
    """Factory for ``--adapter src.compressors.pisco:build_chunked``."""
    checkpoint = checkpoint or paths.PISCO_MISTRAL
    return ChunkedPiscoCompressor(load_cocom(checkpoint, device, dtype), checkpoint, n_chunks)
