"""
构建 tokenizer + generator。tiny 用内置 ToyCausalLM，其余走 HuggingFace。
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_tokenizer_and_generator(cfg) -> Tuple[object, torch.nn.Module]:
    g = cfg.generator

    if g.name_or_path == "toy":
        from .data import build_toy_tokenizer
        from .toy import build_toy

        # 词表覆盖全部 split：留出文档里的答案词若不在词表就会变成 UNK，泛化实验直接失效。
        # 这不是标签泄漏——真实场景里 tokenizer 本来就是固定的、开放词表的。
        paths = cfg.data.vocab_files
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        tok = build_toy_tokenizer(paths, cfg.data,
                                  save_to=os.path.join(cfg.train.out_dir, "toy_tokenizer.json"))
        gen = build_toy(len(tok), g)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(g.name_or_path, trust_remote_code=g.trust_remote_code)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        kwargs = dict(torch_dtype=DTYPES[g.dtype], trust_remote_code=g.trust_remote_code)
        if g.attn_implementation:
            kwargs["attn_implementation"] = g.attn_implementation
        gen = AutoModelForCausalLM.from_pretrained(g.name_or_path, **kwargs)

    # ``freeze`` refers to the base model.  LoRA parameters remain trainable and
    # are saved separately by QuRO; this preserves the cache-compatible frozen
    # generator backbone while still teaching it to consume readout embeddings.
    if g.freeze or g.lora:
        for p in gen.parameters():
            p.requires_grad_(False)
    if g.lora:
        if g.name_or_path == "toy":
            raise ValueError("generator LoRA is only supported for Hugging Face models")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("generator LoRA requires peft: pip install peft") from e
        gen = get_peft_model(gen, LoraConfig(
            r=g.lora_r,
            lora_alpha=g.lora_alpha,
            lora_dropout=g.lora_dropout,
            target_modules=list(g.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        ))
        gen.print_trainable_parameters()
    elif g.freeze:
        gen.eval()
    return tok, gen


def generator_hidden_size(gen) -> int:
    return int(gen.config.hidden_size)
