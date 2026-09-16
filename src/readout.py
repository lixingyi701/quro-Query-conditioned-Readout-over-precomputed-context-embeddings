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
starts from a working system.

The two terms are separately switchable via ``output_mode``:

===============  ==========================================
``full``         ``s * AttnPool(alpha, Z) + Delta``
``pool_only``    ``s * AttnPool(alpha, Z)``
``delta_only``   ``Delta``
===============  ==========================================

``out_proj_init`` is a *separate* knob on purpose.  The legacy
``residual_readout=False`` flag changed the output branch and the initialisation
at the same time, so a drop under it could not be attributed to either -- see
``docs/warning_and_target.md`` W2.  ``forward(output_mode=...)`` also overrides
the composition at inference time, so one trained checkpoint can be probed under
all three branches without retraining.  Attributing a gain to "selection" needs
``pool_only`` to carry it; a ``delta_only`` model that matches ``full`` means the
output is being synthesised rather than selected.

**Slot self-attention.**  Perceiver IO's decoder is a single cross-attention, but
its output queries carry structured positional codes that keep them distinct.
QuRO's slots only have a learned prior, so without letting them see each other
they can all read the same evidence.  Each readout block is therefore
``cross-attention -> slot self-attention -> feed-forward``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perceiver import AttentionBlock, OutputQueryBuilder

#: Which terms of ``s * AttnPool(alpha, Z) + Delta`` reach the decoder.
OUTPUT_MODES = ("full", "pool_only", "delta_only")
#: How ``out_proj`` starts.  ``zeros`` makes Delta exactly 0 at step 0.
OUT_PROJ_INITS = ("zeros", "default")


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

    def forward(self, slots, memory, latent_mask=None, return_attn=False, score_bias=None):
        slots, attn = self.cross(slots, x_kv=memory, mask=latent_mask,
                                 return_attn=return_attn, score_bias=score_bias)
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
        cosine_prior: bool = True,
        prior_mode: str = "rank",
        tau_init: float = 20.0,
        add_slot_index: bool = True,
        residual_readout: bool = True,
        output_mode: Optional[str] = None,
        out_proj_init: Optional[str] = None,
    ):
        super().__init__()
        self.cache_hidden = cache_hidden
        self.gen_hidden = gen_hidden
        self.d_readout = d_readout
        self.max_budget = max_budget
        self.add_document_source = add_document_source
        self.add_slot_index = add_slot_index
        # The pooling branch returns raw cached latents, so it only exists when the
        # cache and the generator share a representation space (they do for
        # PISCO/COCOM, whose latents are Mistral hidden states).
        self.can_pool = bool(cache_hidden == gen_hidden)

        # Legacy ``residual_readout=False`` meant "return Delta alone", so it maps
        # to delta_only.  An explicit output_mode always wins.
        if output_mode is None:
            output_mode = "full" if residual_readout else "delta_only"
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {output_mode!r}")
        self.requested_output_mode = output_mode
        if output_mode in ("full", "pool_only") and not self.can_pool:
            print(f"[readout] cache is {cache_hidden}-d but the generator is {gen_hidden}-d; "
                  f"the pooling branch needs a shared space, so output_mode "
                  f"{output_mode!r} is demoted to 'delta_only'")
            output_mode = "delta_only"
        self.output_mode = output_mode
        # ``residual_readout`` is kept as a read-only alias so old checkpoints and
        # the residual penalty keep working; it is true iff both terms are live.
        self.residual_readout = (output_mode == "full")

        if out_proj_init is None:
            # Zeros only make sense when something else already carries the output.
            out_proj_init = "zeros" if output_mode in ("full", "pool_only") else "default"
        if out_proj_init not in OUT_PROJ_INITS:
            raise ValueError(f"out_proj_init must be one of {OUT_PROJ_INITS}, got {out_proj_init!r}")
        self.out_proj_init = out_proj_init

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

        # Query-latent cosine, injected as an additive bias on the first block's
        # attention logits.  ``tau`` starts large because cosine lives in [-1, 1]
        # and a softmax over a range that small is still essentially uniform; at
        # tau ~ 20 the initial attention reproduces the non-parametric top-B rule.
        # Parameterised in log space so it stays positive under any optimiser.
        self.cosine_prior = bool(cosine_prior)
        self.prior_mode = prior_mode
        self.log_tau = nn.Parameter(torch.tensor(float(math.log(tau_init))))
        # Per-slot targeting.  The cosine prior alone carries no slot index, so
        # every output slot attends to the single best-scoring latent -- measured
        # inter-slot cosine 1.000, i.e. B-1 of the budget wasted.  Slot b is
        # therefore centred on the *b-th ranked* candidate of its own row rather
        # than on a fixed score: ``topk`` values are differentiable, so this is a
        # differentiable form of top-B's rank assignment and lands each slot on a
        # distinct latent at step 0.
        #
        # Fixing the centres in score space instead was tried and fails: the
        # standardised scores are roughly normal, so a band near the mean covers
        # many candidates at once and the attention flattens (effective support
        # 11.7 of 40, gold attention back down to 41%).  Anchoring on actual order
        # statistics sidesteps the density problem entirely.
        self.slot_offset = nn.Parameter(torch.zeros(max_budget))

        if self.out_proj_init == "zeros":
            # Delta starts at exactly zero: step 0 reproduces attention-pooled
            # cached latents, i.e. a working PISCO-like system.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        # base_scale is created whenever pooling is *possible*, not only when it is
        # currently switched on, so that one checkpoint can be replayed under every
        # output_mode without a state_dict mismatch.
        if self.can_pool:
            self.base_scale = nn.Parameter(torch.ones(()))

    @property
    def needs_query(self) -> bool:
        return not self.output_query.is_query_agnostic

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

    def cosine_bias(self, raw: torch.Tensor, query_vector: Optional[torch.Tensor],
                    latent_mask: Optional[torch.Tensor] = None, budget: int = 0):
        """Row-standardised ``tau * z(cos(query, z_j))``, or None.

        The raw cosine is standardised across the candidates of each row before
        ``tau`` is applied.  Measured on this cache, a query vector and the cached
        latents are close to orthogonal -- cosine mean 0.033, within-row std 0.037
        -- so an unstandardised ``tau = 20`` yields a logit spread of 0.74 against
        0.33 for the randomly initialised learned term: the prior barely wins, and
        the attention stays near-uniform.  After standardisation ``tau`` reads
        directly in standard deviations, so the initial peakedness is the same
        whatever the geometry of a particular cache or compressor.
        """
        if not self.cosine_prior or query_vector is None:
            return None
        query = query_vector.float()
        if query.size(-1) != raw.size(-1):
            raise ValueError(
                f"query_vector is {query.size(-1)}-d but the cache is {raw.size(-1)}-d; "
                "the cosine prior needs the query encoded in the cached latents' space")
        cos = F.cosine_similarity(raw, query[:, None, :].expand_as(raw), dim=-1)

        weight = (torch.ones_like(cos) if latent_mask is None
                  else latent_mask.to(cos.dtype))
        count = weight.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (cos * weight).sum(-1, keepdim=True) / count
        var = (((cos - mean) ** 2) * weight).sum(-1, keepdim=True) / count
        score = (cos - mean) / var.clamp_min(1e-8).sqrt()            # (B, K*m)

        # Centre slot b on the b-th largest score in this row, plus a learnable
        # drift.  Invalid latents must not win a rank, hence the mask.
        if latent_mask is not None:
            score = score.masked_fill(~latent_mask, float("-inf"))
        k = min(max(1, budget), score.size(-1))
        centres = score.topk(k, dim=-1).values                        # (B, k)
        if k < budget:                                                # fewer latents than slots
            centres = torch.cat([centres, centres[:, -1:].expand(-1, budget - k)], dim=-1)
        if self.prior_mode == "shared":
            centres = centres[:, :1].expand(-1, budget)      # every slot -> the best one
        centres = centres + self.slot_offset[:budget][None, :]
        score = score.masked_fill(torch.isinf(score), 0.0)
        return -self.log_tau.exp() * (score[:, None, :] - centres[:, :, None]).abs()

    def forward(
        self,
        doc_latents: torch.Tensor,
        document_mask: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        return_attn: bool = False,
        query_vector: Optional[torch.Tensor] = None,
        output_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        mode = self.output_mode if output_mode is None else output_mode
        if mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {mode!r}")
        if mode in ("full", "pool_only") and not self.can_pool:
            raise ValueError(
                f"output_mode={mode!r} needs the pooling branch, but the cache is "
                f"{self.cache_hidden}-d and the generator is {self.gen_hidden}-d")
        batch_size = doc_latents.size(0)
        num_outputs = int(budget or self.max_budget)
        memory, raw, latent_mask = self.prepare_memory(doc_latents, document_mask)

        slots = self.output_query(query_emb, query_mask,
                                  batch_size=batch_size, num_outputs=num_outputs)
        # The first block's weights are both the residual pooling coefficients and
        # the attribution map used for the interpretability analysis, so they are
        # always materialised.
        score_bias = self.cosine_bias(raw, query_vector, latent_mask, num_outputs)
        slots, attention = self.blocks[0](slots, memory, latent_mask, return_attn=True,
                                          score_bias=score_bias)
        for block in self.blocks[1:]:
            slots, _ = block(slots, memory, latent_mask)

        # Mean *square*, not RMS: ``out_proj`` may be zero-initialised, so delta is
        # exactly 0 at step 0 and sqrt() would have an infinite derivative there,
        # poisoning the whole backward pass with NaNs.
        aux = {"attention": attention if return_attn else None,
               "latent_mask": latent_mask,
               "tau": self.log_tau.detach().exp(),
               "slot_offset": self.slot_offset.detach(),
               "output_mode": mode}

        delta = None
        if mode in ("full", "delta_only"):
            delta = self.out_proj(self.out_norm(slots))
        aux["delta_ms"] = (delta.pow(2).mean() if delta is not None
                           else slots.new_zeros(()))

        pooled = None
        if mode in ("full", "pool_only"):
            alpha = attention.mean(dim=1)                   # (B, budget, K*m)
            pooled = torch.bmm(alpha.to(raw.dtype), raw)    # (B, budget, h_cache)
        # pooled_ms is the denominator of the residual penalty, so it must describe
        # the pooling branch or nothing -- the old code reported delta here under
        # residual_readout=False, which made the ratio identically 1.
        aux["pooled_ms"] = (pooled.detach().pow(2).mean() if pooled is not None
                            else slots.new_zeros(()))

        if mode == "delta_only":
            return delta, aux
        if mode == "pool_only":
            return self.base_scale * pooled, aux
        return self.base_scale * pooled + delta, aux

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
