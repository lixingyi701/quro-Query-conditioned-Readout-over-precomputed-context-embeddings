"""Question-conditioned, within-document readout on top of an S0 projector."""
from __future__ import annotations

import torch
from torch import nn

from .projector import SharedDocumentProjector


class QueryGuidedEvidenceProjector(nn.Module):
    def __init__(self, hidden_size, query_dim, memories_per_document, base_hidden=512,
                 attention_dim=256, num_heads=8, query_mode="conditioned",
                 base_trainable=False, stage="full"):
        super().__init__()
        if stage not in {"full", "first_only", "slotwise"}:
            raise ValueError("evidence stage must be full, first_only or slotwise")
        if query_mode not in {"conditioned", "agnostic_matched"}:
            raise ValueError("evidence readout needs real or fixed query tokens")
        if min(hidden_size, query_dim, memories_per_document, attention_dim, num_heads) < 1:
            raise ValueError("evidence dimensions must be positive")
        if attention_dim % num_heads:
            raise ValueError("attention width must be divisible by heads")
        self.hidden_size, self.query_dim = hidden_size, query_dim
        self.memories_per_document = memories_per_document
        self.base_hidden, self.attention_dim, self.num_heads = base_hidden, attention_dim, num_heads
        self.query_mode, self.base_trainable, self.stage = query_mode, base_trainable, stage
        self.needs_query = query_mode == "conditioned"
        self.base = SharedDocumentProjector(hidden_size, query_dim, memories_per_document,
                                            base_hidden, attention_dim, num_heads, query_mode="none")
        self.base.requires_grad_(base_trainable)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.memory_proj = nn.Linear(hidden_size, attention_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, attention_dim)
        self.condition_attention = nn.MultiheadAttention(attention_dim, num_heads, batch_first=True)
        self.condition_norm = nn.LayerNorm(attention_dim)
        self.condition_norm.requires_grad_(stage != "first_only")
        # Always construct this block, even for first_only: common arms consume
        # identical RNG and have identical initial values in the shared layers.
        self.content_attention = nn.MultiheadAttention(attention_dim, num_heads, batch_first=True)
        self.content_attention.requires_grad_(stage != "first_only")
        self.output_norm = nn.LayerNorm(attention_dim)
        self.ffn = nn.Sequential(nn.Linear(attention_dim, 2*attention_dim), nn.GELU(),
                                 nn.Linear(2*attention_dim, attention_dim))
        self.out_proj = nn.Linear(attention_dim, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        fixed = torch.randn(1, 4, query_dim, generator=torch.Generator().manual_seed(0))
        # Same buffer/initialisation in all arms. Real-query arms never read it.
        self.register_buffer("fixed_query", fixed)

    def layout(self):
        names = ("hidden_size", "query_dim", "memories_per_document", "base_hidden",
                 "attention_dim", "num_heads", "query_mode", "base_trainable", "stage")
        return {name: getattr(self, name) for name in names}

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False):
        if budget is not None:
            raise ValueError("evidence_projector uses all cached memories without an explicit budget")
        # S0 masks and detaches source memories, but does NOT detach its own
        # output: the joint pilot must receive gradients through the base.
        e0, aux = self.base(doc_latents, document_mask)
        b, k, m, h = doc_latents.shape
        dm = document_mask.bool()
        e0_docs = e0.reshape(b, k, m, h)
        # Gather valid documents only. Padding is excluded from BOTH attention
        # query and key/value sides; an all-masked softmax is never evaluated.
        owners = dm.nonzero(as_tuple=True)[0]
        u = self.memory_proj(self.memory_norm(e0_docs[dm]))
        if self.needs_query:
            if query_emb is None or query_emb.ndim != 3:
                raise ValueError("evidence readout requires contextual question tokens")
            if query_emb.size(0) != b or query_emb.size(2) != self.query_dim:
                raise ValueError("question batch/width mismatch")
            q = query_emb.detach().float()
            qm = (torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
                  if query_mask is None else query_mask.bool())
            if tuple(qm.shape) != tuple(q.shape[:2]) or not bool(qm.any(1).all()):
                raise ValueError("question mask must match and have a valid token per row")
        else:
            q = self.fixed_query.expand(b, -1, -1)
            qm = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        q = self.query_proj(self.query_norm(torch.where(qm[..., None], q, 0.0)))
        q = torch.where(qm[..., None], q, 0.0)
        c, query_weights = self.condition_attention(
            u, q[owners], q[owners], key_padding_mask=~qm[owners],
            need_weights=return_attn, average_attn_weights=False)
        s = u + c
        content_weights = None
        if self.stage == "first_only":
            r = s
        else:
            diagonal = (torch.eye(m, device=u.device, dtype=torch.bool)
                        if self.stage == "slotwise" else None)
            content, content_weights = self.content_attention(
                self.condition_norm(s), u, u,
                attn_mask=(~diagonal if diagonal is not None else None),
                need_weights=return_attn, average_attn_weights=False)
            r = s + content
        valid_delta = self.out_proj(self.ffn(self.output_norm(r)))
        delta = torch.zeros_like(e0_docs).index_put(tuple(dm.nonzero(as_tuple=True)), valid_delta)
        out = e0_docs + delta
        token_mask = aux["token_mask"]
        result = {"token_mask": token_mask, "latent_mask": token_mask, "attention": None,
                  "output_mode": "full", "delta_ms": delta.square().sum()/token_mask.sum().clamp_min(1)/h,
                  "pooled_ms": e0.square().sum()/token_mask.sum().clamp_min(1)/h}
        if return_attn:
            result.update(query_attention=query_weights, content_attention=content_weights,
                          valid_document_indices=dm.nonzero())
        return out.flatten(1, 2), result
