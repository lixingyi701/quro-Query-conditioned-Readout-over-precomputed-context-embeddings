"""Build the generator stack: the causal LM that consumes QuRO's soft tokens.

For the main experiments the generator is *PISCO's own decoder* -- Mistral-7B-Instruct-v0.2
with the published ``decoder_adapter`` and an embedding table already resized for
the ``<MEM*>/<SEP>`` tokens.  Reusing it rather than loading a fresh Mistral buys
three things at once:

* a warm start -- the decoder already reads soft tokens in this exact space, so
  QuRO begins near a working system instead of at random;
* an airtight baseline -- PISCO and QuRO then share backbone, prompt and LoRA
  initialisation, and differ only in how latents reach the decoder;
* the memory-slot vocabulary the prompt builder needs.

``toy`` keeps the whole pipeline runnable on CPU with no downloads, exercising the
same prompt and readout code paths as the real stack.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from . import paths

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

GENERATOR_LORA_INITS = ("pisco", "random", "frozen")


@dataclass
class GeneratorStack:
    lm: nn.Module
    tokenizer: object
    query_tokenizer: object
    n_mem_tokens: int
    cocom: Optional[nn.Module] = None     # full PISCO object, for cache building / baselines


def _reset_lora_parameters(lm, adapter_name: str) -> int:
    """Re-initialise one adapter to PEFT's default, discarding the trained weights.

    This is the control arm for "did the warm start do the work?", so it must
    really discard the published weights rather than perturb them.
    """
    reset = 0
    for name, parameter in lm.named_parameters():
        if adapter_name not in name:
            continue
        with torch.no_grad():
            if "lora_A" in name:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                reset += 1
            elif "lora_B" in name:
                parameter.zero_()
                reset += 1
    return reset


def _configure_generator_training(lm, lora_init: str, adapter_name: str = "decoder_adapter"):
    """Freeze the backbone and expose only the chosen adapter to the optimiser."""
    if lora_init not in GENERATOR_LORA_INITS:
        raise ValueError(f"generator_lora_init must be one of {GENERATOR_LORA_INITS}")
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    if lora_init == "frozen":
        lm.eval()
        return 0
    if lora_init == "random":
        _reset_lora_parameters(lm, adapter_name)
    trainable = 0
    for name, parameter in lm.named_parameters():
        if adapter_name in name and ("lora_A" in name or "lora_B" in name):
            parameter.requires_grad_(True)
            trainable += parameter.numel()
    if trainable == 0:
        raise RuntimeError(
            f"no LoRA parameters found for adapter {adapter_name!r}; "
            "the checkpoint may not carry adapters")
    return trainable


def build_pisco_stack(cfg) -> GeneratorStack:
    from .compressors.pisco import build_generator

    checkpoint = cfg.generator.name_or_path or paths.PISCO_MISTRAL
    device = cfg.generator.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cocom = build_generator(checkpoint, device=device, dtype=cfg.generator.dtype)

    lm = cocom.decoder
    # PISCO ships two adapters; only the decoder one belongs on the generation
    # path (the encoder one is the compressor, which QuRO never runs online).
    if "decoder_adapter" in getattr(cocom, "adapter_keys", []):
        lm.set_adapter("decoder_adapter")
    # PISCO trains its <MEM*> embeddings; QuRO overwrites those positions with
    # readout outputs, so they are inert here and must not collect gradients.
    lm.get_input_embeddings().weight.requires_grad_(False)

    trainable = _configure_generator_training(lm, cfg.generator.lora_init)
    print(f"[generator] pisco decoder: {trainable/1e6:.2f}M trainable LoRA params "
          f"(init={cfg.generator.lora_init})")

    query_tokenizer = cocom.decoder_tokenizer
    if cfg.query_encoder.kind == "hf":
        from .hf_encoder import build_encoder_tokenizer
        query_tokenizer = build_encoder_tokenizer(cfg.query_encoder)
    return GeneratorStack(lm=lm, tokenizer=cocom.decoder_tokenizer,
                          query_tokenizer=query_tokenizer,
                          n_mem_tokens=int(cocom.n_mem_tokens), cocom=cocom)


def build_toy_stack(cfg) -> GeneratorStack:
    from .data import build_toy_tokenizer
    from .toy import build_toy

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    tokenizer = build_toy_tokenizer(
        cfg.data.vocab_files, cfg.data,
        save_to=os.path.join(cfg.train.out_dir, "toy_tokenizer.json"))
    lm = build_toy(len(tokenizer), cfg.generator)
    for parameter in lm.parameters():
        parameter.requires_grad_(cfg.generator.lora_init != "frozen")
    return GeneratorStack(lm=lm, tokenizer=tokenizer, query_tokenizer=tokenizer,
                          n_mem_tokens=len(tokenizer.mem_tokens))


def build_generator_stack(cfg) -> GeneratorStack:
    if cfg.generator.kind == "pisco":
        return build_pisco_stack(cfg)
    if cfg.generator.kind == "toy":
        return build_toy_stack(cfg)
    raise ValueError(f"unknown generator kind: {cfg.generator.kind}")
