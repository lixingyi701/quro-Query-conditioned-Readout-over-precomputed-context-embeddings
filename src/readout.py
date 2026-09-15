"""QuRO's online readout: cached latents + query -> B generator-ready soft tokens.

This is the module the paper is about.  Everything upstream (a frozen PISCO or
COCOM compressor) and downstream (a frozen Mistral decoder plus LoRA) is
off-the-shelf; the readout is what turns a reusable, query-independent document
memory into a query-specific, aggressively short prefix.

Three properties are load-bearing and are implemented deliberately:

**Bottleneck width.**  PISCO latents are 4096-dimensional.  Running attention at
that width costs 211M parameters, which would undercut the efficiency argument
the method rests on.  The readout therefore works in ``d_readout`` (default 1024)
and bridges back out, for roughly 24M parameters.

**Residual readout.**  The output is ``E = s * AttnPool(alpha, Z) + Delta`` where
``alpha`` comes from the first cross-attention and ``Delta`` is a zero-initialised
branch.  Two things follow.  The output inherits the scale of the cached latents
(measured std ~1.9, abs-max ~23 for PISCO -- a LayerNorm'd output at scale 1
would be badly mismatched against what the decoder LoRA was trained on).  And at
step 0 QuRO degenerates to attention-pooled PISCO rather than noise, so training
starts from a working system and every point gained is attributable to
query-conditioned selection.

**Slot self-attention.**  Perceiver IO's decoder is a single cross-attention, but
its output queries carry structured positional codes that keep them distinct.
QuRO's slots only have a learned prior, so without letting them see each other
they can all read the same evidence.  Each readout block is therefore
``cross-attention -> slot self-attention -> feed-forward``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .perceiver import AttentionBlock, OutputQueryBuilder


class ReadoutBlock(nn.Module):
    """One ``cross-attention -> slot self-attention`` stage over the latent memory."""

    def __init__(self, d_readout: int, num_heads: int = 8, head_dim: Optional[int] = None,
                 cross_widening: int = 1, self_widening: int = 2, dropout: float = 0.0):
        super().__init__()
        self.cross = AttentionBlock(
            q_dim=d_readout, kv_dim=d_readout, num_heads=num_heads, head_dim=head_dim,
            widening=cross_widening, dropout=dropout)
        self.slot_self = AttentionBlock(
            q_dim=d_readout, num_heads=num_heads, head_dim=head_dim,
            widening=self_widening, dropout=dropout)

    def forward(self, slots, memory, latent_mask=None, return_attn=False):
        slots, attn = self.cross(slots, x_kv=memory, mask=latent_mask, return_attn=return_attn)
        slots, _ = self.slot_self(slots)
        return slots, attn


class QuroReadout(nn.Module):
    """``(B,K,m,h_cache)`` cached latents + query tokens -> ``(B,budget,h_gen)``.

    Attention cost is ``O(budget * K * m)`` and is independent of the original
    document lengths, which is the entire basis of the efficiency claim.
    """

    def __init__(
        self,
        cache_hidden: int,
        gen_hidden: int,
        query_dim: int,
        d_readout: int = 1024,
        max_budget: int = 8,
        num_blocks: int = 1,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        output_query_mode: str = "xattn",
        cross_widening: int = 1,
        self_widening: int = 2,
        dropout: float = 0.0,
        max_document_sources: int = 32,
        max_latents_per_document: int = 64,
        add_document_source: bool = True,
        add_slot_index: bool = True,
        residual_readout: bool = True,
    ):
        super().__init__()
        self.cache_hidden = cache_hidden
        self.gen_hidden = gen_hidden
        self.d_readout = d_readout
        self.max_budget = max_budget
        self.add_document_source = add_document_source
        self.add_slot_index = add_slot_index
        # The residual path pools raw cached latents, so it only exists when the
        # cache and the generator share a representation space (they do for
        # PISCO/COCOM, whose latents are Mistral hidden states).
        self.residual_readout = bool(residual_readout and cache_hidden == gen_hidden)

        self.in_norm = nn.LayerNorm(cache_hidden)
        self.in_proj = nn.Linear(cache_hidden, d_readout)

        self.document_source = (nn.Embedding(max_document_sources, d_readout)
                                if add_document_source else None)
        self.slot_index = (nn.Embedding(max_latents_per_document, d_readout)
                           if add_slot_index else None)
        for embedding in (self.document_source, self.slot_index):
            if embedding is not None:
                nn.init.normal_(embedding.weight, std=0.02)

        self.output_query = OutputQueryBuilder(
            d_readout, max_budget, query_dim, output_query_mode, num_heads, dropout)
        self.blocks = nn.ModuleList([
            ReadoutBlock(d_readout, num_heads, head_dim, cross_widening, self_widening, dropout)
            for _ in range(max(1, num_blocks))
        ])
        self.out_norm = nn.LayerNorm(d_readout)
        self.out_proj = nn.Linear(d_readout, gen_hidden)

        if self.residual_readout:
            # Delta starts at exactly zero: step 0 reproduces attention-pooled
            # cached latents, i.e. a working PISCO-like system.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self.base_scale = nn.Parameter(torch.ones(()))

    @property
    def needs_query(self) -> bool:
        return self.output_query.mode != "agnostic"

    def prepare_memory(self, doc_latents: torch.Tensor, document_mask: torch.Tensor):
        """``(B,K,m,h) -> (B,K*m,d_readout)`` plus the flattened key-padding mask."""
        if doc_latents.ndim != 4:
            raise ValueError(f"doc_latents must be (B,K,m,h), got {tuple(doc_latents.shape)}")
        b, k, m, h = doc_latents.shape
        if h != self.cache_hidden:
            raise ValueError(f"cache hidden size is {h}, readout expects {self.cache_hidden}")
        if document_mask.shape != (b, k):
            raise ValueError(f"document_mask must be {(b, k)}, got {tuple(document_mask.shape)}")
        if self.document_source is not None and k > self.document_source.num_embeddings:
            raise ValueError(f"retrieved K={k} exceeds max_document_sources")
        if self.slot_index is not None and m > self.slot_index.num_embeddings:
            raise ValueError(f"cached m={m} exceeds max_latents_per_document")

        raw = doc_latents.reshape(b, k * m, h).float()
        x = self.in_proj(self.in_norm(raw))
        if self.document_source is not None:
            ranks = torch.arange(k, device=x.device)
            x = x + self.document_source(ranks)[None, :, None, :].expand(b, k, m, -1).reshape(b, k * m, -1)
        if self.slot_index is not None:
            slots = torch.arange(m, device=x.device)
            x = x + self.slot_index(slots)[None, None, :, :].expand(b, k, m, -1).reshape(b, k * m, -1)
        latent_mask = document_mask[:, :, None].expand(b, k, m).reshape(b, k * m)
        return x, raw, latent_mask

    def forward(
        self,
        doc_latents: torch.Tensor,
        document_mask: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size = doc_latents.size(0)
        num_outputs = int(budget or self.max_budget)
        memory, raw, latent_mask = self.prepare_memory(doc_latents, document_mask)

        slots = self.output_query(query_emb, query_mask,
                                  batch_size=batch_size, num_outputs=num_outputs)
        # The first block's weights are both the residual pooling coefficients and
        # the attribution map used for the interpretability analysis, so they are
        # always materialised.
        slots, attention = self.blocks[0](slots, memory, latent_mask, return_attn=True)
        for block in self.blocks[1:]:
            slots, _ = block(slots, memory, latent_mask)

        delta = self.out_proj(self.out_norm(slots))
        # Mean *square*, not RMS: ``out_proj`` is zero-initialised, so delta is
        # exactly 0 at step 0 and sqrt() would have an infinite derivative there,
        # poisoning the whole backward pass with NaNs.
        aux = {"attention": attention if return_attn else None,
               "latent_mask": latent_mask,
               "delta_ms": delta.pow(2).mean()}
        if not self.residual_readout:
            aux["pooled_ms"] = delta.detach().pow(2).mean()
            return delta, aux

        alpha = attention.mean(dim=1)                       # (B, budget, K*m)
        pooled = torch.bmm(alpha.to(raw.dtype), raw)        # (B, budget, h_cache)
        aux["pooled_ms"] = pooled.detach().pow(2).mean()
        return self.base_scale * pooled + delta, aux

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
