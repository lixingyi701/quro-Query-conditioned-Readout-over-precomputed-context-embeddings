"""
tiny 预设专用的零依赖 generator：ToyTokenizer + ToyCausalLM。

目的：在没有网络、没有 HF 权重的机器上也能立刻跑通整条链路，
验证 Encode/Process/Decode 的张量流、mask、以及 loss 能否正确反传到压缩器。
接口刻意与 HF 对齐（get_input_embeddings / forward(inputs_embeds, attention_mask, labels) / generate），
这样 model.py 里不需要为两种 generator 写两套代码。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


class ToyTokenizer:
    """词级别 tokenizer，词表从语料现场构建并存成 json。"""

    def __init__(self, vocab: Optional[List[str]] = None):
        specials = [PAD, BOS, EOS, UNK]
        vocab = vocab or []
        self.itos: List[str] = specials + [t for t in vocab if t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    # ---- HF 风格属性 ----
    @property
    def pad_token_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_token_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return len(self.itos)

    # ---- 核心 ----
    @classmethod
    def build_from_texts(cls, texts: Sequence[str], min_freq: int = 1) -> "ToyTokenizer":
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(_TOKEN_RE.findall(t))
        vocab = [w for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_freq]
        return cls(vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in _TOKEN_RE.findall(text)]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.itos):
                continue
            t = self.itos[i]
            if skip_special_tokens and t in (PAD, BOS, EOS):
                continue
            toks.append(t)
        return " ".join(toks)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ToyTokenizer":
        with open(path, encoding="utf-8") as f:
            itos = json.load(f)
        tok = cls()
        tok.itos = itos
        tok.stoi = {t: i for i, t in enumerate(itos)}
        return tok


# --------------------------------------------------------------------------------------
@dataclass
class ToyOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


@dataclass
class ToyLMConfig:
    hidden_size: int
    vocab_size: int
    num_layers: int
    num_heads: int
    max_position_embeddings: int


class _Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class ToyCausalLM(nn.Module):
    """随机初始化的小型 causal LM，接口对齐 HF AutoModelForCausalLM 的子集。"""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layer: int = 4,
                 n_head: int = 4, max_pos: int = 2048):
        super().__init__()
        self.config = ToyLMConfig(hidden_size=d_model, vocab_size=vocab_size,
                                  num_layers=n_layer, num_heads=n_head,
                                  max_position_embeddings=max_pos)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,                    # (B, T, d)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) 1=有效
        labels: Optional[torch.Tensor] = None,          # (B, T)，-100 忽略
        **kwargs,
    ) -> ToyOutput:
        b, t, _ = inputs_embeds.shape
        assert t <= self.config.max_position_embeddings, (
            f"序列长度 {t} 超过 toy LM 的 max_pos={self.config.max_position_embeddings}；"
            f"调小 readout budget / answer length，或调大 toy_max_pos"
        )
        pos = torch.arange(t, device=inputs_embeds.device)
        x = inputs_embeds + self.pos(pos)[None]

        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, causal, kpm)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            # 与 HF 一致：内部做 shift，预测 t+1
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return ToyOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 16, eos_token_id: Optional[int] = None, **kwargs):
        """贪心解码。与 HF 用 inputs_embeds 时一致：只返回新生成的 token id。"""
        b = inputs_embeds.size(0)
        dev = inputs_embeds.device
        emb = inputs_embeds
        am = attention_mask if attention_mask is not None else torch.ones(emb.shape[:2], device=dev, dtype=torch.long)
        out = torch.zeros(b, 0, dtype=torch.long, device=dev)
        done = torch.zeros(b, dtype=torch.bool, device=dev)
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=emb, attention_mask=am).logits[:, -1]
            nxt = logits.argmax(-1)                                  # (B,)
            if eos_token_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_token_id), nxt)
                done = done | (nxt == eos_token_id)
            out = torch.cat([out, nxt[:, None]], dim=1)
            emb = torch.cat([emb, self.embed(nxt)[:, None]], dim=1)
            am = torch.cat([am, torch.ones(b, 1, device=dev, dtype=am.dtype)], dim=1)
            if eos_token_id is not None and bool(done.all()):
                break
        return out


def build_toy(vocab_size: int, cfg) -> ToyCausalLM:
    return ToyCausalLM(vocab_size=vocab_size, d_model=cfg.toy_d_model,
                       n_layer=cfg.toy_n_layer, n_head=cfg.toy_n_head, max_pos=cfg.toy_max_pos)
