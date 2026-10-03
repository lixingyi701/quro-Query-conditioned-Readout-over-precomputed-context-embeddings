"""Query-conditioned residual projectors over reusable document memories.

The shared document projector reads variable-length questions with attention;
the earlier joint projector flattens all documents/questions as a control.
Only residual-branch features are normalised; the decoder receives native Z+delta.
"""

from __future__ import annotations

import math

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


class SharedDocumentProjector(nn.Module):
    """Shared m->m residual MLP, preceded by memory-to-query attention.

    K and question length are runtime axes, never axes of a learned weight.
    Optional document attention mixes query-conditioned document hidden vectors
    without document/rank positional embeddings, hence is permutation equivariant.
    """

    def __init__(self, hidden_size, query_dim, memories_per_document, hidden_dim=512,
                 attention_dim=256, num_heads=8, conditioning="cross_attention",
                 query_mode="conditioned", cross_document=False, query_position=False,
                 support_head=False):
        super().__init__()
        if min(hidden_size, query_dim, memories_per_document, hidden_dim,
               attention_dim, num_heads) < 1:
            raise ValueError("projector dimensions must be positive")
        if attention_dim % num_heads or (cross_document and hidden_dim % num_heads):
            raise ValueError("attention widths must be divisible by num_heads")
        if conditioning not in {"cross_attention", "last"}:
            raise ValueError(f"unknown query conditioning: {conditioning}")
        if query_mode not in {"conditioned", "agnostic_matched"}:
            raise ValueError(f"unknown projector query mode: {query_mode}")
        self.hidden_size, self.query_dim = hidden_size, query_dim
        self.memories_per_document, self.hidden_dim = memories_per_document, hidden_dim
        self.attention_dim, self.num_heads = attention_dim, num_heads
        self.conditioning, self.query_mode = conditioning, query_mode
        self.cross_document, self.query_position = cross_document, query_position
        self.needs_query = query_mode == "conditioned"
        self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.query_norm = nn.LayerNorm(query_dim, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(attention_dim, elementwise_affine=False)
        self.to_value = nn.Linear(query_dim, attention_dim, bias=False)
        self.to_query = (nn.Linear(hidden_size, attention_dim, bias=False)
                         if conditioning == "cross_attention" else None)
        self.to_key = (nn.Linear(query_dim, attention_dim, bias=False)
                       if conditioning == "cross_attention" else None)
        m = memories_per_document
        self.memory_proj = nn.Linear(m * hidden_size, hidden_dim)
        self.context_proj = nn.Linear(m * attention_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, m * hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.document_norm = (nn.LayerNorm(hidden_dim, elementwise_affine=False)
                              if cross_document else None)
        self.document_attention = (nn.MultiheadAttention(hidden_dim, num_heads,
                                                         dropout=0.0, batch_first=True)
                                   if cross_document else None)
        fixed = torch.randn(1, 4, query_dim, generator=torch.Generator().manual_seed(0))
        self.register_buffer("fixed_query", fixed if not self.needs_query else None)
        # Created after all existing weights: enabling this head preserves their init.
        self.support_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.support_classifier = nn.Linear(hidden_dim, 1) if support_head else None

    @staticmethod
    def positional_features(mask, width, dtype):
        # Positions count valid question tokens, so left/right padding agrees.
        positions = (mask.long().cumsum(1) - 1).clamp_min(0).to(dtype)
        frequency = torch.exp(torch.arange(0, width, 2, device=mask.device, dtype=dtype)
                              * (-math.log(10000.0) / width))
        angle = positions[:, :, None] * frequency
        pe = torch.zeros(*mask.shape, width, device=mask.device, dtype=dtype)
        pe[:, :, 0::2] = angle.sin()
        pe[:, :, 1::2] = angle[:, :, :width // 2].cos()
        return pe

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False, return_support=False):
        if budget is not None:
            raise ValueError("shared_projector always preserves all memories; leave budget unset")
        if doc_latents.ndim != 4:
            raise ValueError("document latents must have shape (B, K, m, h)")
        b, k, m, h = doc_latents.shape
        if k < 1 or (m, h) != (self.memories_per_document, self.hidden_size):
            raise ValueError("cache memory count/hidden width does not match shared projector")
        if tuple(document_mask.shape) != (b, k):
            raise ValueError("document mask does not match cached latents")
        dm = document_mask.bool()
        if not bool(dm.any(1).all()):
            raise ValueError("each query needs at least one valid document")
        memory = torch.where(dm[:, :, None, None], doc_latents.detach().float(), 0.0)
        z = self.memory_norm(memory)
        if self.needs_query:
            if (query_emb is None or query_emb.ndim != 3
                    or query_emb.size(0) != b or query_emb.size(2) != self.query_dim):
                raise ValueError("shared projector requires compatible query token states")
            q = query_emb.detach().float()
            qm = (torch.ones(q.shape[:2], device=q.device, dtype=torch.bool)
                  if query_mask is None else query_mask.bool())
            if tuple(qm.shape) != tuple(q.shape[:2]) or not bool(qm.any(1).all()):
                raise ValueError("query mask must match embeddings and contain valid tokens")
        else:
            q = self.fixed_query.expand(b, -1, -1)
            qm = torch.ones(q.shape[:2], device=q.device, dtype=torch.bool)
        q = self.query_norm(torch.where(qm[:, :, None], q, 0.0))
        if self.query_position:
            q = self.query_norm(q + self.positional_features(qm, self.query_dim, q.dtype))
        q = torch.where(qm[:, :, None], q, 0.0)

        attention = None
        if self.conditioning == "last":
            positions = torch.arange(q.size(1), device=q.device)[None, :].expand(b, -1)
            last = positions.masked_fill(~qm, -1).max(1).values
            context = self.to_value(q[torch.arange(b, device=q.device), last])
            context = context[:, None, None, :].expand(b, k, m, -1)
        else:
            heads, dh = self.num_heads, self.attention_dim // self.num_heads
            queries = self.to_query(z).reshape(b, k*m, heads, dh).transpose(1, 2)
            keys = self.to_key(q).reshape(b, q.size(1), heads, dh).transpose(1, 2)
            values = self.to_value(q).reshape(b, q.size(1), heads, dh).transpose(1, 2)
            valid_keys = qm[:, None, None, :]
            if return_attn:
                scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(dh)
                weights = scores.masked_fill(~valid_keys, float("-inf")).softmax(-1)
                context = weights @ values
                attention = weights * dm[:, None, :, None, None].expand(
                    b, heads, k, m, q.size(1)).reshape(b, heads, k*m, q.size(1))
            else:
                context = F.scaled_dot_product_attention(queries, keys, values,
                                                          attn_mask=valid_keys, dropout_p=0.0)
            context = context.transpose(1, 2).reshape(b, k, m, self.attention_dim)
        # Normalise the readout, not the native Z+delta output. This bounds the
        # MLP input RMS approximately independently of question length, with eps
        # handling near-zero vectors; it is not an exact constant-norm promise.
        context = self.context_norm(context)
        context = torch.where(dm[:, :, None, None], context, 0.0)
        u = F.gelu(self.memory_proj(z.flatten(2)) + self.context_proj(context.flatten(2)))
        if self.cross_document:
            un = self.document_norm(u)
            mixed, _ = self.document_attention(un, un, un, key_padding_mask=~dm,
                                               need_weights=False)
            u = u + mixed
        delta = self.out_proj(u).reshape(b, k, m, h)
        support_logits = None
        if return_support and self.support_classifier is not None:
            support_logits = self.support_classifier(self.support_norm(u)).squeeze(-1)
            support_logits = torch.where(dm, support_logits, 0.0)
        delta = torch.where(dm[:, :, None, None], delta, 0.0)
        token_mask = dm[:, :, None].expand(b, k, m).reshape(b, k*m)
        denom = token_mask.sum().clamp_min(1) * h
        return (memory + delta).reshape(b, k*m, h), {
            "attention": None, "query_attention": attention, "support_logits": support_logits,
            "token_mask": token_mask, "latent_mask": token_mask,
            "output_mode": "full", "delta_ms": delta.square().sum() / denom,
            "pooled_ms": memory.square().sum() / denom}
