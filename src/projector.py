"""Joint query-conditioned MLP over reusable document memories.

The first layer sees all K*m memories and the ordered query tokens together.
Splitting that layer into memory/query matrices is exactly concatenation followed
by a Linear, with independent fan-in initialisation for the two input sources.
Only the residual branch is normalised; the native decoder receives Z + delta.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class JointQueryProjector(nn.Module):
    def __init__(self, hidden_size, query_dim, max_documents, memories_per_document,
                 max_query_tokens, hidden_dim=128, query_mode="conditioned"):
        super().__init__()
        if min(hidden_size, query_dim, max_documents, memories_per_document,
               max_query_tokens, hidden_dim) < 1:
            raise ValueError("projector dimensions must be positive")
        if query_mode not in {"conditioned", "agnostic_matched"}:
            raise ValueError(f"unknown projector query mode: {query_mode}")
        self.hidden_size = hidden_size
        self.query_dim = query_dim
        self.max_documents = max_documents
        self.memories_per_document = memories_per_document
        self.max_query_tokens = max_query_tokens
        self.max_budget = max_documents * memories_per_document
        self.query_mode = query_mode
        self.needs_query = query_mode == "conditioned"
        self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.query_norm = nn.LayerNorm(query_dim, elementwise_affine=False)
        self.memory_proj = nn.Linear(self.max_budget * hidden_size, hidden_dim)
        self.query_proj = nn.Linear(max_query_tokens * query_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, self.max_budget * hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # The matched control keeps identical trainable matrices, but receives a
        # fixed query-independent input (including fixed length/padding).
        fixed = torch.randn(1, max_query_tokens, query_dim,
                            generator=torch.Generator().manual_seed(0))
        self.register_buffer("fixed_query", fixed if not self.needs_query else None)

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False):
        if budget is not None and int(budget) != self.max_budget:
            raise ValueError(f"joint_projector preserves the full {self.max_budget}-slot budget")
        if doc_latents.ndim != 4:
            raise ValueError("document latents must have shape (B, K, m, h)")
        b, k, m, h = doc_latents.shape
        if k > self.max_documents or (m, h) != (self.memories_per_document, self.hidden_size):
            raise ValueError(
                f"expected K <= {self.max_documents}, m={self.memories_per_document}, "
                f"h={self.hidden_size}; got {tuple(doc_latents.shape)}")
        if tuple(document_mask.shape) != (b, k):
            raise ValueError("document mask does not match the cached latents")
        doc_mask = document_mask.bool()
        if not bool(doc_mask.any(1).all()):
            raise ValueError("each query needs at least one valid document")
        memory = torch.where(doc_mask[:, :, None, None], doc_latents.detach().float(), 0.0)
        memory = F.pad(memory, (0, 0, 0, 0, 0, self.max_documents - k))
        doc_mask = F.pad(doc_mask, (0, self.max_documents - k), value=False)
        token_mask = doc_mask[:, :, None].expand(-1, -1, m).reshape(b, self.max_budget)
        memory = memory.reshape(b, self.max_budget, h)

        if self.needs_query:
            if query_emb is None or query_emb.ndim != 3:
                raise ValueError("conditioned projector requires ordered query token embeddings")
            if query_emb.size(0) != b or query_emb.size(2) != self.query_dim:
                raise ValueError("query embeddings have incompatible batch/hidden dimensions")
            if query_emb.size(1) > self.max_query_tokens:
                raise ValueError("query exceeds projector's configured maximum length")
            q = query_emb.detach().float()
            qm = (torch.ones(q.shape[:2], device=q.device, dtype=torch.bool)
                  if query_mask is None else query_mask.bool())
            if tuple(qm.shape) != tuple(q.shape[:2]) or not bool(qm.any(1).all()):
                raise ValueError("query mask must match embeddings and contain valid tokens")
            # Mask before and after normalisation: pad embeddings must have no
            # influence even if their values are large or nonfinite.
            q = self.query_norm(torch.where(qm[:, :, None], q, 0.0))
            q = torch.where(qm[:, :, None], q, 0.0)
            q = F.pad(q, (0, 0, 0, self.max_query_tokens - q.size(1)))
        else:
            q = self.query_norm(self.fixed_query).expand(b, -1, -1)

        u = F.gelu(self.memory_proj(self.memory_norm(memory).flatten(1))
                   + self.query_proj(q.flatten(1)))
        delta = self.out_proj(u).reshape(b, self.max_budget, h)
        delta = torch.where(token_mask[:, :, None], delta, 0.0)
        out = memory + delta
        denom = token_mask.sum().clamp_min(1) * h
        return out, {"attention": None, "token_mask": token_mask,
                     "latent_mask": token_mask, "output_mode": "full",
                     "delta_ms": delta.square().sum() / denom,
                     "pooled_ms": memory.square().sum() / denom}
