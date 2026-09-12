"""Token-level Hugging Face encoder used for online queries.

Cache-backed QuRO freezes this module and obtains document latents from the
offline PISCO/COCOM cache. Prototype mode may also reuse it for documents.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class HFDocEncoder(nn.Module):
    def __init__(self, dc):
        super().__init__()
        from transformers import AutoModel

        self.dc = dc
        self._ng_cache = None
        self.backbone = AutoModel.from_pretrained(
            dc.name_or_path,
            torch_dtype=DTYPES[dc.dtype],
            trust_remote_code=dc.trust_remote_code,
        )
        self.out_dim = int(self.backbone.config.hidden_size)
        self.max_len = dc.max_doc_len
        self.layer_index = dc.layer_index

        if dc.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        if dc.lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:                     # 明确报错，别让人以为 LoRA 静默生效了
                raise ImportError("--lora_doc_encoder 需要 peft：pip install peft") from e
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=dc.lora_r, lora_alpha=dc.lora_alpha, lora_dropout=dc.lora_dropout,
                target_modules=list(dc.lora_targets), bias="none",
                task_type="FEATURE_EXTRACTION",
            ))
            self.backbone.print_trainable_parameters()

        # HF 模型自带位置编码（RoPE），默认不再叠一层可学习位置嵌入
        self.pos_emb = nn.Embedding(dc.max_doc_len, self.out_dim) if dc.add_learned_pos else None
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(self.out_dim)

    @property
    def _needs_grad(self) -> bool:
        # 缓存：0.6B 有几百个参数张量，每步前向遍历两遍纯属浪费
        if getattr(self, "_ng_cache", None) is None:
            self._ng_cache = any(p.requires_grad for p in self.backbone.parameters())
        return self._ng_cache

    def enable_grad_ckpt(self):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                # 底座冻结时输入不带梯度，checkpoint 会整段跳过重算，LoRA 也就收不到梯度
                self.backbone.enable_input_require_grads()

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ids.size(1) > self.max_len:
            ids = ids[:, : self.max_len]
            mask = None if mask is None else mask[:, : self.max_len]
        attn = torch.ones_like(ids) if mask is None else mask.long()

        want_hidden = self.layer_index != -1
        ctx = torch.enable_grad() if self._needs_grad else torch.no_grad()
        with ctx:
            out = self.backbone(input_ids=ids, attention_mask=attn,
                                output_hidden_states=want_hidden)
            h = out.hidden_states[self.layer_index] if want_hidden else out.last_hidden_state

        # 压缩器跑 fp32：它是唯一在更新的部分，别让 bf16 的舍入噪声混进梯度
        x = h.float()
        if self.pos_emb is not None:
            t = min(x.size(1), self.max_len)
            pos = torch.arange(t, device=x.device).clamp(max=self.max_len - 1)
            x = x[:, :t] + self.pos_emb(pos)[None]
        return self.norm(x)


def build_encoder_tokenizer(dc):
    """编码器和生成器的词表不同（Qwen3 vs Qwen2.5），文档/query 必须用编码器自己的 tokenizer。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(dc.name_or_path, trust_remote_code=dc.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
