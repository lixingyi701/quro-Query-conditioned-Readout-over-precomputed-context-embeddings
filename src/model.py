"""QuRO v0.1: query-conditioned readout over frozen, precomputed document latents.

    offline (once per document)   d  --frozen PISCO/COCOM-->  Z_d in R^(m x h)   -> disk
    online  (once per query)      q + Z in R^(K*m x h)  --readout-->  E in R^(B x h) -> decoder

Only the readout is trained (plus optional generator LoRA).  The compressor is
never loaded during online training, which is the point: if training re-encoded
documents every epoch the cacheability claim would be untested.

The decoder is PISCO's own Mistral plus its ``decoder_adapter``, and the prompt is
PISCO's own template with ``B`` memory slots instead of ``K * m``.  A PISCO
baseline is therefore the same object with :class:`~src.baselines.PiscoDirectReadout`
swapped in -- same backbone, same prompt, same LoRA initialisation, differing only
in how latents reach the decoder.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .baselines import PiscoDirectReadout, SimilarityTopBReadout, pool_query_in_generator_space
from .prompt import PiscoPromptBuilder, assemble_inputs
from .readout import QuroReadout


class TokenEmbeddingQueryEncoder(nn.Module):
    """Trainable token encoder used by the dependency-free CPU contract tests."""

    def __init__(self, vocab_size, d_model, max_len, pad_id=0):
        super().__init__()
        self.out_dim = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[pad_id].zero_()

    def forward(self, ids, mask=None):
        ids = ids[:, : self.max_len]
        x = self.tok_emb(ids)
        pos = torch.arange(x.size(1), device=ids.device)
        return self.norm(x + self.pos_emb(pos)[None])


class QueryBudgetSelector(nn.Module):
    """Predict one discrete output-token budget from the query.

    Discrete buckets rather than a continuous ratio: the density-aware line of
    work found that fully dynamic continuous rates underperform static ones
    (``QURO_EXPERIMENTAL_DESIGN.md`` §7.5), and buckets also keep batching sane.
    """

    def __init__(self, query_dim: int, buckets: Sequence[int]):
        super().__init__()
        self.register_buffer("buckets", torch.tensor(sorted(set(buckets)), dtype=torch.long))
        self.classifier = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, query_dim), nn.GELU(),
            nn.Linear(query_dim, len(self.buckets)),
        )

    @staticmethod
    def masked_mean(x, mask):
        if mask is None:
            return x.mean(1)
        weight = mask.unsqueeze(-1).to(x.dtype)
        return (x * weight).sum(1) / weight.sum(1).clamp_min(1.0)

    def forward(self, query_emb, query_mask):
        logits = self.classifier(self.masked_mean(query_emb, query_mask))
        return self.buckets[logits.argmax(-1)], logits


class QuROModel(nn.Module):
    """Cache-first QuRO: frozen query encoder + trained readout + frozen LM (+LoRA)."""

    def __init__(self, cfg, lm, tokenizer, query_encoder, n_mem_tokens: int,
                 cache_hidden: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self._lm = [lm]                     # hidden from state_dict; it is frozen or LoRA-only
        self.tok = tokenizer
        self.query_encoder = query_encoder
        self.d_gen = int(lm.config.hidden_size)
        self.cache_hidden = int(cache_hidden or cfg.readout.cache_hidden or self.d_gen)
        self.n_mem_tokens = int(n_mem_tokens)
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        r = cfg.readout
        if r.kind == "quro":
            self.readout = QuroReadout(
                cache_hidden=self.cache_hidden, gen_hidden=self.d_gen,
                query_dim=query_encoder.out_dim, d_readout=r.d_readout,
                max_budget=r.max_budget, num_blocks=r.num_blocks, num_heads=r.num_heads,
                head_dim=r.head_dim, output_query_mode=r.output_query_mode,
                cross_widening=r.cross_widening, self_widening=r.self_widening,
                dropout=r.dropout, max_document_sources=r.max_document_sources,
                max_latents_per_document=r.max_latents_per_document,
                add_document_source=r.add_document_source, add_slot_index=r.add_slot_index,
                residual_readout=r.residual_readout)
        elif r.kind == "pisco_direct":
            self.readout = PiscoDirectReadout(self.cache_hidden, self.d_gen)
        elif r.kind == "similarity_topb":
            self.readout = SimilarityTopBReadout(self.cache_hidden, self.d_gen)
        else:
            raise ValueError(f"unknown readout kind: {r.kind}")

        self.budget_selector = QueryBudgetSelector(query_encoder.out_dim, r.budget_buckets)
        if not r.adaptive_budget:
            for parameter in self.budget_selector.parameters():
                parameter.requires_grad_(False)

        self.prompt_builders = {
            mode: PiscoPromptBuilder(tokenizer, self.n_mem_tokens, mode)
            for mode in ("D0", "D1", "D2", "D3")
        }
        self.decoder_input_mode = cfg.decoder.input_mode
        self.query_text_dropout = float(cfg.decoder.query_text_dropout)

    # -- plumbing -------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        # The LM is held in a list to keep it out of state_dict, which also keeps
        # it out of nn.Module's mode switching -- so drive it explicitly.  PISCO's
        # adapters use lora_dropout=0.1; leaving that on during evaluation makes
        # greedy decoding non-deterministic.
        self.lm.train(mode and self.cfg.generator.lora_init != "frozen")
        # A frozen encoder must stay in eval mode or its dropout keeps firing,
        # making the readout's Q side noisy and the run irreproducible.
        if self.cfg.query_encoder.freeze and not self.cfg.query_encoder.lora:
            self.query_encoder.eval()
        return self

    @property
    def lm(self):
        return self._lm[0]

    @property
    def gen_dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    def trainable_parameters(self):
        own = [p for p in self.parameters() if p.requires_grad]
        return own + [p for p in self.lm.parameters() if p.requires_grad]

    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def parameter_report(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "readout": count(self.readout),
            "budget_selector": count(self.budget_selector),
            "query_encoder": count(self.query_encoder),
            "generator_lora": sum(p.numel() for p in self.lm.parameters() if p.requires_grad),
            "total": self.num_trainable(),
        }

    # -- query ----------------------------------------------------------
    def encode_query(self, query_ids, query_mask):
        max_len = self.cfg.data.max_query_len
        return self.query_encoder(query_ids[:, :max_len],
                                  None if query_mask is None else query_mask[:, :max_len])

    def _resolve_budgets(self, query_emb, query_mask, budget, batch_size, device):
        logits = None
        if self.cfg.readout.adaptive_budget and budget is None:
            if query_emb is None:
                raise ValueError("adaptive budget requires a query representation")
            budgets, logits = self.budget_selector(query_emb, query_mask)
            return budgets, logits
        if self.cfg.readout.adaptive_budget:
            _, logits = self.budget_selector(query_emb, query_mask)

        if budget is None:
            budget = self.cfg.readout.max_budget
        if isinstance(budget, int):
            budgets = torch.full((batch_size,), budget, dtype=torch.long, device=device)
        else:
            budgets = torch.as_tensor(budget, dtype=torch.long, device=device)
            if budgets.ndim == 0:
                budgets = budgets.expand(batch_size)
        if budgets.shape != (batch_size,):
            raise ValueError(f"budget must resolve to shape ({batch_size},)")
        allowed = set(self.cfg.readout.budget_buckets)
        bad = [int(x) for x in budgets.detach().cpu() if int(x) not in allowed]
        if bad:
            raise ValueError(f"budgets {sorted(set(bad))} are outside buckets {sorted(allowed)}")
        return budgets, logits

    # -- readout --------------------------------------------------------
    def readout_cached(self, batch, budget=None, return_attn=False):
        latents, document_mask = batch["cached_latents"], batch["document_mask"]
        device = latents.device
        needs_query = getattr(self.readout, "needs_query", True) or self.cfg.readout.adaptive_budget
        query_emb = (self.encode_query(batch["query_ids"], batch["query_mask"])
                     if needs_query else None)
        budgets, budget_logits = self._resolve_budgets(
            query_emb, batch.get("query_mask"), budget, latents.size(0), device)

        kwargs = {}
        if isinstance(self.readout, SimilarityTopBReadout):
            kwargs["query_vector"] = pool_query_in_generator_space(
                self.lm, batch["query_gen_ids"], batch["query_gen_mask"])
        soft_tokens, aux = self.readout(
            latents, document_mask, query_emb, batch.get("query_mask"),
            budget=int(budgets.max().item()), return_attn=return_attn, **kwargs)

        token_mask = aux.get("token_mask")
        if token_mask is None:
            token_mask = (torch.arange(soft_tokens.size(1), device=device)[None, :]
                          < budgets[:, None])
        return {"soft_tokens": soft_tokens, "soft_token_mask": token_mask,
                "budgets": token_mask.sum(1), "budget_logits": budget_logits, "aux": aux}

    # -- decoder --------------------------------------------------------
    def build_prompts(self, batch, token_mask, training: bool):
        """One prompt per row, with the memory-slot count equal to that row's budget.

        ``query_text_dropout`` randomly renders a row in D1 (question text removed).
        With no plain-text query the only route for query information is the soft
        tokens, so the loss cannot fall unless the readout really conditions on the
        query -- this is a training constraint, not merely an eval ablation.
        """
        prompts = []
        for i, query in enumerate(batch["queries"]):
            budget = int(token_mask[i].sum().item())
            if budget < 1:
                raise ValueError(f"row {i} has an empty readout budget")
            mode = self.decoder_input_mode
            if training and self.query_text_dropout > 0 and mode == "D0":
                if random.random() < self.query_text_dropout:
                    mode = "D1"
            prompts.append(self.prompt_builders[mode].build(query, budget))
        return prompts

    def qa_loss(self, batch, budget=None, return_attn=False):
        result = self.readout_cached(batch, budget=budget, return_attn=return_attn)
        prompts = self.build_prompts(batch, result["soft_token_mask"], training=self.training)
        packed = assemble_inputs(
            self.lm.get_input_embeddings(), prompts,
            result["soft_tokens"], result["soft_token_mask"],
            target_ids=batch["target_ids"], pad_token_id=self.pad_id, pad_side="right")
        return self.lm(**packed).loss, result

    def residual_penalty(self, result) -> torch.Tensor:
        """Keep the trained residual small relative to the pooled cached latents.

        With a zero-initialised residual the model starts as attention-pooled
        PISCO.  Penalising ``||Delta|| / ||pooled||`` for the first few hundred
        steps stops the readout from being dragged away from that working solution
        before it has learned anything, which is the practical form of the slow
        cross-attention convergence noted in ``QURO_EXPERIMENTAL_DESIGN.md`` §8.1.
        """
        aux = result["aux"]
        if "delta_ms" not in aux or not getattr(self.readout, "residual_readout", False):
            return torch.zeros((), device=result["soft_tokens"].device)
        # Already a mean square, so no sqrt is taken anywhere on the grad path.
        return aux["delta_ms"] / aux["pooled_ms"].detach().clamp_min(1e-6)

    def forward(self, batch, budget=None, residual_weight: float = 0.0):
        qa, result = self.qa_loss(batch, budget=budget)
        total = self.cfg.train.beta_qa * qa
        output = {"qa_loss": qa.detach(),
                  "mean_budget": result["budgets"].float().mean().detach()}
        if residual_weight > 0:
            penalty = self.residual_penalty(result)
            total = total + residual_weight * penalty
            output["residual_penalty"] = penalty.detach()
        if result["budget_logits"] is not None and "budget" in batch:
            buckets = self.budget_selector.buckets
            targets = (batch["budget"][:, None] == buckets[None, :]).long().argmax(-1)
            budget_loss = nn.functional.cross_entropy(result["budget_logits"], targets)
            total = total + self.cfg.train.budget_loss_weight * budget_loss
            output["budget_loss"] = budget_loss.detach()
        output["loss"] = total
        return output

    @torch.no_grad()
    def generate_answer(self, batch, max_new_tokens=32, budget=None):
        was_training = self.training
        self.eval()
        try:
            result = self.readout_cached(batch, budget=budget)
            prompts = self.build_prompts(batch, result["soft_token_mask"], training=False)
            packed = assemble_inputs(
                self.lm.get_input_embeddings(), prompts,
                result["soft_tokens"], result["soft_token_mask"],
                target_ids=None, pad_token_id=self.pad_id, pad_side="left")
            ids = self.lm.generate(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=getattr(self.tok, "eos_token_id", None),
                pad_token_id=getattr(self.tok, "pad_token_id", None))
            return [self.tok.decode(row.tolist(), skip_special_tokens=True).strip()
                    for row in ids]
        finally:
            self.train(was_training)

    # -- checkpointing --------------------------------------------------
    def trainable_state_dict(self):
        keep = {name for name, value in self.named_parameters() if value.requires_grad}
        keep |= {name for name, _ in self.named_buffers()
                 if not name.startswith("query_encoder.backbone.")}
        return {name: value for name, value in self.state_dict().items() if name in keep}

    def save(self, path, optimizer=None, scheduler=None, step=None):
        trainable = {name for name, p in self.lm.named_parameters() if p.requires_grad}
        payload = {
            "format_version": 2,
            "state_dict": self.trainable_state_dict(),
            "generator_trainable": {name: value.detach().cpu()
                                    for name, value in self.lm.state_dict().items()
                                    if name in trainable},
            "config": asdict(self.cfg),
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, path)

    def load(self, path, strict=False, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=strict)
        if ckpt.get("generator_trainable"):
            self.lm.load_state_dict(ckpt["generator_trainable"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        missing = [x for x in missing if not x.startswith("query_encoder.backbone.")]
        return missing, unexpected, ckpt.get("step")


def build_model(cfg, cache_hidden: Optional[int] = None):
    """Assemble tokenizer, frozen LM and QuRO for the configured backend."""
    from .generator import build_generator_stack
    stack = build_generator_stack(cfg)
    query_encoder = build_query_encoder(cfg, stack)
    model = QuROModel(cfg, stack.lm, stack.tokenizer, query_encoder,
                      stack.n_mem_tokens, cache_hidden)
    return stack, model


def build_query_encoder(cfg, stack):
    """Frozen token-level query encoder; its hidden states are the readout's Q side."""
    if cfg.query_encoder.kind == "hf":
        from .hf_encoder import HFTokenEncoder
        return HFTokenEncoder(cfg.query_encoder)
    return TokenEmbeddingQueryEncoder(
        vocab_size=max(len(stack.query_tokenizer),
                       int(getattr(stack.query_tokenizer, "vocab_size", 0) or 0)),
        d_model=cfg.query_encoder.d_model,
        max_len=cfg.data.max_query_len,
        pad_id=getattr(stack.query_tokenizer, "pad_token_id", 0) or 0)
