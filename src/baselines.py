"""Readout baselines that share QuRO's interface, so only the readout differs.

Every arm in the main table consumes the same cached latents, the same prompt and
the same decoder; swapping one of these modules in for :class:`~src.readout.QuroReadout`
changes exactly one thing -- how ``K * m`` cached latents become ``B`` soft tokens.

``PiscoDirectReadout``
    The published system: no selection at all, every cached latent is handed to
    the decoder.  Its budget is ``K * m``, so it is a reference row rather than a
    budget-matched competitor.

``SimilarityTopBReadout``
    Query-conditioned but non-parametric: keep the ``B`` latents most similar to
    the query.  This is the arm that matters most for interpreting a QuRO win --
    if the trained cross-attention cannot beat cosine similarity, it has not
    learned selection, it is just doing retrieval with extra steps
    (``QURO_EXPERIMENTAL_DESIGN.md`` §4).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten(doc_latents: torch.Tensor, document_mask: torch.Tensor):
    b, k, m, h = doc_latents.shape
    return doc_latents.reshape(b, k * m, h), document_mask[:, :, None].expand(b, k, m).reshape(b, k * m)


class PiscoDirectReadout(nn.Module):
    """Pass cached latents through untouched; budget is whatever the cache holds."""

    needs_query = False

    def __init__(self, cache_hidden: int, gen_hidden: int):
        super().__init__()
        if cache_hidden != gen_hidden:
            raise ValueError("pisco_direct requires the cache and generator to share a space")
        self.cache_hidden = cache_hidden
        self.gen_hidden = gen_hidden

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False):
        memory, latent_mask = _flatten(doc_latents, document_mask)
        return memory.float(), {"attention": None, "latent_mask": latent_mask,
                                "token_mask": latent_mask}


class SimilarityTopBReadout(nn.Module):
    """Keep the ``B`` cached latents closest to the query; no trained parameters.

    The query vector must live in the cache's space.  Since PISCO latents are
    Mistral hidden states, it is obtained by running the frozen generator over the
    query and mean-pooling -- which also means this baseline is *more* expensive
    online than QuRO, and that should be reported rather than hidden.
    """

    # The query reaches this arm as a pooled generator-space vector, not through
    # the readout's own query encoder, so that encoder is never run.
    needs_query = False

    def __init__(self, cache_hidden: int, gen_hidden: int, query_pooler=None):
        super().__init__()
        if cache_hidden != gen_hidden:
            raise ValueError("similarity_topb requires the cache and generator to share a space")
        self.cache_hidden = cache_hidden
        self.gen_hidden = gen_hidden
        self.query_pooler = query_pooler

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False, query_vector=None):
        memory, latent_mask = _flatten(doc_latents, document_mask)
        memory = memory.float()
        if query_vector is None:
            raise ValueError("similarity_topb needs query_vector in the cached-latent space")
        scores = F.cosine_similarity(
            memory, query_vector.float()[:, None, :].expand_as(memory), dim=-1)
        scores = scores.masked_fill(~latent_mask, float("-inf"))

        k = min(int(budget or memory.size(1)), memory.size(1))
        top = scores.topk(k, dim=1)
        # Cache order is meaningful (document rank, then position); restore it so
        # the decoder sees evidence in reading order rather than score order.
        index = top.indices.sort(dim=1).values
        selected = memory.gather(1, index[:, :, None].expand(-1, -1, memory.size(-1)))
        token_mask = latent_mask.gather(1, index)
        return selected, {"attention": None, "latent_mask": latent_mask, "token_mask": token_mask}


@torch.no_grad()
def pool_query_in_generator_space(lm, input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pooled last hidden state of the frozen generator over the query."""
    output = lm(input_ids=input_ids, attention_mask=attention_mask.long(),
                output_hidden_states=True)
    hidden = output.hidden_states[-1].float()
    weight = attention_mask[:, :, None].to(hidden.dtype)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)
