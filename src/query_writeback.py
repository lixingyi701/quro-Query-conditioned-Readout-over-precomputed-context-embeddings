"""RQ: query-as-Q reads PISCO memory, then writes back through the same attention.

Distinct from R (latent-as-Q -> query KV -> latent self-attention). Per head:
    A = softmax(Q_query K_latent^T / sqrt(d_head))
    R = A V_latent; C = A^T R; E = Z + zero_init_out(GELU(C))
No column renormalisation, pooling of the identity path, or latent self-attention.
"""
from __future__ import annotations

import torch
from torch import nn

from .baselines import _flatten


class PiscoQueryWritebackReadout(nn.Module):
    needs_query = True

    def __init__(self, cache_hidden, gen_hidden, query_dim, d_readout=256,
                 num_heads=8, num_blocks=1, dropout=0.0):
        super().__init__()
        if cache_hidden != gen_hidden:
            raise ValueError("query writeback requires cache_hidden == gen_hidden")
        if num_blocks != 1:
            raise ValueError("query writeback implements one read/write stage; num_blocks must be 1")
        if num_heads < 1 or d_readout < 1 or d_readout % num_heads:
            raise ValueError("d_readout must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_readout // num_heads
        self.query_proj = nn.Sequential(nn.LayerNorm(query_dim),
                                        nn.Linear(query_dim, d_readout, bias=False))
        self.memory_norm = nn.LayerNorm(cache_hidden)
        self.key_proj = nn.Linear(cache_hidden, d_readout, bias=False)
        self.value_proj = nn.Linear(cache_hidden, d_readout, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_readout, gen_hidden)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False):
        if doc_latents.ndim != 4 or document_mask.shape != doc_latents.shape[:2]:
            raise ValueError("expected doc_latents (B,K,m,h) and document_mask (B,K)")
        raw, mask = _flatten(doc_latents, document_mask.bool())
        raw = raw.float()
        if query_emb is None or query_emb.ndim != 3 or query_emb.size(0) != raw.size(0):
            raise ValueError("query writeback requires query embeddings (B,Tq,hq)")
        if query_mask is None:
            query_mask = torch.ones(query_emb.shape[:2], device=query_emb.device, dtype=torch.bool)
        if query_mask.shape != query_emb.shape[:2]:
            raise ValueError("query_mask must have shape (B,Tq)")
        query_mask = query_mask.bool()
        if not bool(mask.any(-1).all()) or not bool(query_mask.any(-1).all()):
            raise ValueError("each row needs at least one valid latent and query token")

        # Clear padding before learned operations, including NaN cache padding.
        memory = self.memory_norm(raw.masked_fill(~mask[..., None], 0))
        query = self.query_proj(query_emb.float().masked_fill(~query_mask[..., None], 0))
        q, k, v = self._split(query), self._split(self.key_proj(memory)), self._split(self.value_proj(memory))
        # FP32 scores/read/write avoid overflow in AMP; cast at the output bridge.
        # Explicitly disable autocast because .float() alone does not disable matmul autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim ** -0.5
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            attention = scores.softmax(-1)
            # Padded queries must neither read nor write. Do not softmax all -inf rows.
            attention = attention.masked_fill(~query_mask[:, None, :, None], 0)
            read = attention @ v.float()
            written = attention.transpose(-1, -2) @ read
        b, _, m, _ = written.shape
        written = written.transpose(1, 2).reshape(b, m, -1)
        delta = self.out_proj(self.dropout(self.activation(written.to(self.out_proj.weight.dtype))))
        delta = delta.masked_fill(~mask[..., None], 0)
        # budget is a label only: retain every original latent, order and token mask.
        return raw + delta, {
            "token_mask": mask, "latent_mask": mask,
            "attention": attention if return_attn else None,
            "attention_axes": "batch,head,query_token,latent",
            "refinement_rms": delta.detach()[mask].square().mean().sqrt(),
        }
