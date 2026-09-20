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

from .baselines import (PiscoDirectReadout, SimilarityTopBReadout,
                        encode_query_in_generator_space, pool_query_in_generator_space)
from .distill import distillation_loss
from .prompt import (DECODER_INPUT_MODES, QUERY_SLOT_MODES, SLOTLESS_MODES,
                     PiscoPromptBuilder, assemble_inputs)
from .readout import QuroReadout
from .refinement import PiscoResidualReadout
from .query_writeback import PiscoQueryWritebackReadout


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


class GeneratorQueryEncoder(nn.Module):
    """Encode the query with the frozen generator, so Q and the latents share a space.

    PISCO/COCOM latents *are* generator hidden states.  Running the query through
    a separate encoder leaves the readout to learn the bridge between two
    unrelated coordinate systems from scratch, supervised only by a distant answer
    CE -- measured cost: the learned attention lands on the gold document 39.3% of
    the time where cosine scoring in a shared space reaches 67.6%.

    The forward pass is short (a query is ~15 tokens against the decoder's ~70) and
    the non-parametric baseline already pays it, so the comparison stays fair; it
    does belong in the efficiency accounting.

    ``representation`` decides whether that encoding is a *fixed function*:

    ``shared_current``
        The query is encoded through whatever the decoder adapter currently is.
        Since the decoder's LoRA is being trained on the answer loss, the query
        representation drifts across training.  ``no_grad`` stops gradient, not
        drift.  This is the historical behaviour and every published run used it.
    ``fixed_adapter``
        A frozen copy of the adapter is taken at construction and activated for
        the query forward only.  The query representation is then constant, so a
        gain can be attributed to the readout rather than to the query space and
        the decoder co-adapting.
    """

    def __init__(self, lm, pooling: str = "last",
                 representation: str = "shared_current",
                 adapter_name: str = "decoder_adapter"):
        super().__init__()
        self._lm = [lm]                      # hidden from state_dict; frozen
        self.pooling = pooling
        self.representation = representation
        self.decoder_adapter = adapter_name
        self.query_adapter = None
        self.out_dim = int(lm.config.hidden_size)
        self.last_pooled = None              # set by forward(), read by the readout
        self.query_adapter_hash = None
        if representation == "fixed_adapter":
            active = self._active_adapters(lm)
            if len(active) > 1:
                # peft.PeftModel.set_adapter takes a single name while
                # transformers' takes a list, so restoring a multi-adapter set
                # after the query forward cannot be done portably.  Refuse here
                # instead of restoring only the first one and losing the rest.
                raise ValueError(
                    f"fixed_adapter needs exactly one active adapter to restore, "
                    f"found {active}; combining adapters is not supported yet")
            self.query_adapter = self._freeze_adapter_copy(lm, adapter_name)
            # The copy is of whatever the decoder adapter holds at this moment.
            # That is the published adapter only when the encoder is built before
            # any weights are loaded *and* generator_lora_init is not "random"
            # (which resets the adapter first).  So the copy is not reliably
            # reconstructible from the model path, and save() stores it; the hash
            # is what makes a mismatch an error instead of a silent substitution.
            self.query_adapter_hash = self.adapter_hash(lm, self.query_adapter)

    @staticmethod
    def adapter_hash(lm, adapter_name: str) -> str:
        import hashlib

        digest = hashlib.sha1()
        for name, parameter in sorted(lm.named_parameters()):
            if f".{adapter_name}." in name:
                digest.update(name.encode("utf-8"))
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    @staticmethod
    def _active_adapters(lm) -> list:
        """Name(s) of the currently active adapter, across three spellings.

        ``peft.PeftModel`` exposes ``active_adapters`` as a *property* returning a
        list; transformers' ``PeftAdapterMixin`` exposes it as a *method*; older
        versions only have the scalar ``active_adapter``.  Taking the attribute
        without calling it yields a bound method, which then gets handed to
        ``set_adapter`` as if it were a name.
        """
        for attribute in ("active_adapters", "active_adapter"):
            value = getattr(lm, attribute, None)
            if value is None:
                continue
            if callable(value):
                value = value()
            if isinstance(value, (list, tuple)):
                return [str(x) for x in value]
            return [str(value)]
        return []

    @staticmethod
    def _grad_snapshot(lm) -> dict:
        """Which parameters are trainable right now.

        ``set_adapter`` documents that it sets the target adapter to
        ``requires_grad=True``.  Switching adapters for a query forward would
        therefore quietly hand the optimiser a different trainable set -- and
        since the optimiser holds the Parameter objects, the damage shows up as
        the decoder silently not training rather than as an error.  Snapshot and
        restore instead of trusting the side effect.
        """
        return {name: p.requires_grad for name, p in lm.named_parameters()}

    @staticmethod
    def _restore_grads(lm, snapshot: dict) -> None:
        for name, parameter in lm.named_parameters():
            want = snapshot.get(name)
            if want is not None and parameter.requires_grad != want:
                parameter.requires_grad_(want)

    @staticmethod
    def _freeze_adapter_copy(lm, adapter_name: str) -> str:
        """Duplicate ``adapter_name`` under a new name and freeze the copy.

        Copying the adapter rather than the whole 7B keeps the backbone shared;
        only the LoRA deltas are duplicated, which is a few hundred MB at most.
        """
        import copy as _copy

        configs = getattr(lm, "peft_config", None)
        if not configs or adapter_name not in configs:
            raise ValueError(
                f"cannot freeze a copy of adapter {adapter_name!r}: the generator "
                f"exposes {sorted(configs or [])}")
        frozen = "quro_query_adapter"
        if frozen in configs:
            raise ValueError(f"adapter {frozen!r} already exists; refusing to overwrite")
        before = GeneratorQueryEncoder._grad_snapshot(lm)
        # Two libraries expose add_adapter with the arguments in opposite orders:
        #   peft.PeftModel            (adapter_name, peft_config)
        #   transformers PeftAdapterMixin (adapter_config, adapter_name)
        # PISCO's decoder is a transformers model with the mixin, not a PeftModel,
        # so hardcoding either order works in exactly one of the two places.
        # Dispatch on the real signature rather than on the class.
        import inspect

        parameters = list(inspect.signature(lm.add_adapter).parameters)
        config = _copy.deepcopy(configs[adapter_name])
        if parameters and parameters[0] in ("adapter_config", "peft_config"):
            lm.add_adapter(config, frozen)
        else:
            lm.add_adapter(frozen, config)
        # add_adapter creates a *fresh* adapter, so the weights have to be copied
        # across explicitly -- otherwise the query would be encoded through a
        # randomly initialised LoRA rather than through PISCO's trained one.
        source = dict(lm.named_parameters())
        copied = 0
        with torch.no_grad():
            for name, parameter in lm.named_parameters():
                if f".{frozen}." not in name:
                    continue
                origin = name.replace(f".{frozen}.", f".{adapter_name}.")
                if origin not in source:
                    raise ValueError(f"no counterpart for {name} in {adapter_name}")
                parameter.copy_(source[origin])
                parameter.requires_grad_(False)
                copied += 1
        if copied == 0:
            raise RuntimeError(f"adapter {frozen!r} has no parameters to freeze")
        # add_adapter leaves the new adapter active; put the decoder's back, then
        # undo the requires_grad churn both calls caused.  The frozen copy is not
        # in the snapshot (it did not exist yet), so it stays False.
        lm.set_adapter(adapter_name)
        GeneratorQueryEncoder._restore_grads(lm, before)
        trainable = sum(1 for n, p in lm.named_parameters()
                        if p.requires_grad and adapter_name in n)
        print(f"[query] fixed_adapter: froze {copied} tensors copied from "
              f"{adapter_name}; {trainable} decoder tensors remain trainable")
        return frozen

    def forward(self, ids, mask=None):
        return self._encode(ids, mask)[0]

    def pooled(self, ids, mask=None):
        """Sentence vector only, through the *same* controls as ``forward``.

        The cosine prior and the non-parametric top-B arm need this vector but
        not the per-token states, and they used to obtain it by calling
        ``pool_query_in_generator_space`` on the raw LM.  That path skipped both
        the adapter switch and the eval-mode guard, so A1 and S kept encoding
        their queries through the *current* decoder adapter under LoRA dropout
        while a run tagged ``fixed_adapter`` claimed otherwise -- and those are
        exactly the control arms C1 is compared against.
        """
        return self._encode(ids, mask)[1]

    def _encode(self, ids, mask=None):
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        lm = self._lm[0]
        # The generator and this encoder are the same object, so during training
        # the LM carries PISCO's lora_dropout=0.1.  Encoding the query under
        # dropout would make the cosine prior stochastic and the run
        # irreproducible, so force eval mode here and restore it afterwards.
        was_training = lm.training
        lm.eval()
        # Whatever this forward switches must be put back exactly, or the next
        # decoder forward silently runs on the wrong adapter and the answer loss
        # trains nothing.  Read the active set rather than assuming it.
        switching = self.query_adapter is not None
        previous = self._active_adapters(lm) if switching else []
        grads = self._grad_snapshot(lm) if switching else None
        try:
            if switching:
                lm.set_adapter(self.query_adapter)
            with torch.no_grad():
                hidden, pooled = encode_query_in_generator_space(
                    lm, ids, mask, self.pooling)
        finally:
            if switching:
                # Restoring a multi-adapter set is not uniformly supported --
                # transformers' set_adapter takes a list, peft.PeftModel takes a
                # single name -- so the constructor rejects that case rather than
                # letting this line silently drop everything after the first.
                lm.set_adapter(previous[0] if previous else self.decoder_adapter)
                self._restore_grads(lm, grads)
            lm.train(was_training)
        self.last_pooled = pooled
        return hidden, pooled


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
                residual_readout=r.residual_readout,
                output_mode=r.output_mode, out_proj_init=r.out_proj_init,
                cosine_prior=r.cosine_prior and cache_hidden == self.d_gen,
                prior_mode=r.prior_mode, tau_init=r.tau_init)
        elif r.kind == "pisco_direct":
            self.readout = PiscoDirectReadout(self.cache_hidden, self.d_gen)
        elif r.kind == "pisco_residual":
            self.readout = PiscoResidualReadout(
                self.cache_hidden, self.d_gen, query_encoder.out_dim,
                d_readout=r.d_readout, num_heads=r.num_heads,
                num_blocks=r.num_blocks, dropout=r.dropout)
        elif r.kind == "pisco_query_writeback":
            self.readout = PiscoQueryWritebackReadout(
                self.cache_hidden, self.d_gen, query_encoder.out_dim,
                d_readout=r.d_readout, num_heads=r.num_heads,
                num_blocks=r.num_blocks, dropout=r.dropout)
        elif r.kind == "similarity_topb":
            self.readout = SimilarityTopBReadout(self.cache_hidden, self.d_gen)
        else:
            raise ValueError(f"unknown readout kind: {r.kind}")

        self.budget_selector = QueryBudgetSelector(query_encoder.out_dim, r.budget_buckets)
        if not r.adaptive_budget:
            for parameter in self.budget_selector.parameters():
                parameter.requires_grad_(False)

        self.prompt_builders = {
            mode: PiscoPromptBuilder(tokenizer, self.n_mem_tokens, mode,
                                     query_tokens=cfg.decoder.query_tokens)
            for mode in DECODER_INPUT_MODES
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
    def uses_cosine_prior(self) -> bool:
        return bool(getattr(self.readout, "cosine_prior", False))

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
        report = {
            "readout": count(self.readout),
            "budget_selector": count(self.budget_selector),
            "query_encoder": count(self.query_encoder),
            "generator_lora": sum(p.numel() for p in self.lm.parameters() if p.requires_grad),
            "total": self.num_trainable(),
        }
        # The query path shares the decoder's weights, so "how many parameters"
        # does not describe it; which representation it uses does.
        report["query_representation"] = getattr(
            self.query_encoder, "representation", "n/a")
        digest = getattr(self.query_encoder, "query_adapter_hash", None)
        if digest is not None:
            report["query_adapter_hash"] = digest
        return report

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

    def compress_query(self, batch, tokens: int):
        """The question as ``tokens`` embeddings in the generator's own space.

        Segment mean-pooling over the query encoder's hidden states, which are
        already generator hidden states -- the same space the cached latents and
        therefore the soft tokens live in, so they can be written straight into
        prompt positions with no learned projection.

        Parameter-free on purpose.  A learned compressor would confound "the
        question survives compression" with "we added capacity"; if mean-pooling
        to a quarter of the length costs nothing, that is the stronger result, and
        if it costs a lot we know a learned one is worth trying.

        Order is preserved by pooling contiguous segments rather than attending:
        a question is not a bag of words, and "who directed X" and "X directed
        who" would otherwise land on the same vectors.
        """
        hidden = self.encode_query(batch["query_gen_ids"], batch["query_gen_mask"])
        mask = batch["query_gen_mask"][:, :hidden.size(1)].to(hidden.dtype)
        rows, length, _ = hidden.shape
        out = hidden.new_zeros(rows, tokens, hidden.size(-1))
        lengths = mask.sum(1).clamp_min(1.0)
        for row in range(rows):
            n = int(lengths[row].item())
            # Segment boundaries over the *real* tokens only; padding must not
            # dilute a segment, and a short question must not leave empty ones.
            edges = [round(i * n / tokens) for i in range(tokens + 1)]
            for slot in range(tokens):
                start, stop = edges[slot], max(edges[slot] + 1, edges[slot + 1])
                stop = min(stop, n)
                piece = hidden[row, start:stop]
                out[row, slot] = piece.mean(0) if piece.size(0) else hidden[row, :n].mean(0)
        return out

    def _compressed_query(self, batch):
        """Only computed when a mode actually reserves slots for it."""
        if self.decoder_input_mode not in QUERY_SLOT_MODES:
            return None
        return self.compress_query(batch, self.cfg.decoder.query_tokens)

    def query_vector(self, batch):
        """Generator-space sentence vector, through the configured representation.

        Every arm that needs this vector must come through here.  A1 (agnostic
        slots + cosine prior) and S (non-parametric top-B) have
        ``needs_query=False``, so they never call the encoder's forward, and the
        old code fell back to ``pool_query_in_generator_space`` on the raw LM --
        skipping the adapter switch and the eval-mode guard.  The effect was that
        the two control arms encoded their queries through the *current* decoder
        adapter under LoRA dropout while the run record said ``fixed_adapter``.
        Since those arms are what C1's margin is measured against, the bypass
        biased the comparison rather than merely mislabelling it.
        """
        cached = getattr(self.query_encoder, "last_pooled", None)
        if cached is not None:
            return cached
        pooled = getattr(self.query_encoder, "pooled", None)
        if callable(pooled):
            return pooled(batch["query_gen_ids"][:, :self.cfg.data.max_query_len],
                          batch["query_gen_mask"][:, :self.cfg.data.max_query_len])
        # Encoders that are not the generator (toy, hf) have no adapter to switch
        # and no shared dropout, so the raw call is equivalent for them.
        return pool_query_in_generator_space(
            self.lm, batch["query_gen_ids"], batch["query_gen_mask"],
            self.cfg.query_encoder.pooling)

    # -- readout --------------------------------------------------------
    def readout_cached(self, batch, budget=None, return_attn=False, output_mode=None):
        """``output_mode`` overrides the readout's output branch for this call only.

        Probing one trained checkpoint under full / pool_only / delta_only measures
        what the trained model *currently relies on*; retraining under a mode
        measures what it could compensate for.  They answer different questions and
        neither substitutes for the other (HANDOFF.md §3 W2).
        """
        latents, document_mask = batch["cached_latents"], batch["document_mask"]
        device = latents.device
        needs_query = getattr(self.readout, "needs_query", True) or self.cfg.readout.adaptive_budget
        # Clear first: a stale pooled vector from the previous batch would be
        # silently reused when the encoder is skipped.
        if hasattr(self.query_encoder, "last_pooled"):
            self.query_encoder.last_pooled = None
        query_emb = (self.encode_query(batch["query_ids"], batch["query_mask"])
                     if needs_query else None)
        budgets, budget_logits = self._resolve_budgets(
            query_emb, batch.get("query_mask"), budget, latents.size(0), device)

        # One generator-space query vector serves two arms: the non-parametric
        # baseline scores with it directly, and the trained readout uses it as the
        # cosine prior its attention starts from.
        kwargs = {}
        if isinstance(self.readout, SimilarityTopBReadout) or self.uses_cosine_prior:
            kwargs["query_vector"] = self.query_vector(batch)
        if output_mode is not None:
            if not isinstance(self.readout, QuroReadout):
                raise ValueError(
                    f"output_mode is only defined for the quro readout, not "
                    f"{type(self.readout).__name__}")
            kwargs["output_mode"] = output_mode
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
        documents = batch.get("document_texts")
        prompts = []
        for i, query in enumerate(batch["queries"]):
            budget = int(token_mask[i].sum().item())
            mode = self.decoder_input_mode
            if training and self.query_text_dropout > 0 and mode == "D0":
                if random.random() < self.query_text_dropout:
                    mode = "D1"
            if mode not in SLOTLESS_MODES and budget < 1:
                raise ValueError(f"row {i} has an empty readout budget")
            prompts.append(self.prompt_builders[mode].build(
                query, budget, documents[i] if documents else None))
        return prompts

    @staticmethod
    def answer_positions(labels):
        """Where the answer is predicted, in answer-relative order.

        A causal LM's logit at ``t`` predicts the token at ``t+1``, so the
        supervised positions are ``labels[:, 1:] != -100`` read off ``logits[:, :-1]``.
        Teacher and student prompts have different lengths -- 80 soft tokens
        against 8 -- so absolute positions do not correspond; the answer suffix is
        identical, so the *relative* index within each row does.  Distillation
        aligns on that index, never on the raw position.

        Returns ``(rows, cols, order)``: which batch row, which logit index, and
        which answer token it is.
        """
        supervised = labels[:, 1:] != -100
        rows, cols = supervised.nonzero(as_tuple=True)
        # Rank within each row: nonzero() yields row-major order, so a cumulative
        # count over the mask gives the answer-relative index directly.
        order = (supervised.long().cumsum(dim=1)[rows, cols] - 1)
        return rows, cols, order

    def qa_loss(self, batch, budget=None, return_attn=False, return_logits=False):
        result = self.readout_cached(batch, budget=budget, return_attn=return_attn)
        prompts = self.build_prompts(batch, result["soft_token_mask"], training=self.training)
        packed = assemble_inputs(
            self.lm.get_input_embeddings(), prompts,
            result["soft_tokens"], result["soft_token_mask"],
            target_ids=batch["target_ids"], pad_token_id=self.pad_id, pad_side="right",
            query_tokens=self._compressed_query(batch))
        output = self.lm(**packed)
        if return_logits:
            rows, cols, order = self.answer_positions(packed["labels"])
            # Only the supervised rows are kept: the full (B, T, V) tensor is
            # mostly prompt, and distillation has nothing to say about it.
            result["answer_logits"] = output.logits[rows, cols]
            result["answer_rows"] = rows
            result["answer_order"] = order
            result["answer_targets"] = packed["labels"][:, 1:][rows, cols]
        return output.loss, result

    def residual_penalty(self, result) -> torch.Tensor:
        """Keep the trained residual small relative to the pooled cached latents.

        With a zero-initialised residual the model starts as attention-pooled
        PISCO.  Penalising ``||Delta|| / ||pooled||`` for the first few hundred
        steps stops the readout from being dragged away from that working solution
        before it has learned anything, which is the practical form of the slow
        cross-attention convergence noted in ``QURO_EXPERIMENTAL_DESIGN.md`` §8.1.
        """
        aux = result["aux"]
        # Only "full" has both terms, so only there is the ratio defined.  Keyed on
        # the aux value rather than the module so a branch override at call time
        # cannot silently leave a penalty applied to a branch that is switched off.
        if "delta_ms" not in aux or aux.get("output_mode", "full") != "full":
            return torch.zeros((), device=result["soft_tokens"].device)
        # Already a mean square, so no sqrt is taken anywhere on the grad path.
        return aux["delta_ms"] / aux["pooled_ms"].detach().clamp_min(1e-6)

    def distillation_loss(self, batch, result, teacher):
        """KL(P || student) over the answer tokens, aligned by answer index.

        The teacher's prompt is longer -- 80 soft tokens against 8 -- so absolute
        positions do not correspond and only the answer-relative index does.  The
        stored target ids are compared against the student's as an assertion:
        if the two ever teacher-force different tokens, the alignment is wrong and
        the loss would be pulling towards the distribution for some other word.
        """
        rows = result["answer_rows"].cpu().tolist()
        orders = result["answer_order"].cpu().tolist()
        ids = [batch["ids"][row] for row in rows]
        index, probability, tail, keep, targets = teacher.gather(
            ids, orders, result["answer_logits"].device)
        if not bool(keep.any()):
            return None
        mismatch = keep & (targets.long() != result["answer_targets"].long())
        if bool(mismatch.any()):
            raise ValueError(
                f"{int(mismatch.sum())} answer positions teacher-force a different "
                "token than the cached teacher did; the distillation alignment is "
                "wrong (check max_answer_len and the tokenizer against the cache "
                "metadata)")
        return distillation_loss(
            result["answer_logits"][keep], index[keep], probability[keep],
            tail[keep], temperature=teacher.temperature)

    def forward(self, batch, budget=None, residual_weight: float = 0.0,
                teacher=None, kd_weight: float = 0.0):
        want_logits = teacher is not None and kd_weight > 0
        qa, result = self.qa_loss(batch, budget=budget, return_logits=want_logits)
        total = self.cfg.train.beta_qa * qa
        output = {"qa_loss": qa.detach(),
                  "mean_budget": result["budgets"].float().mean().detach()}
        if want_logits:
            # The gold CE stays: the teacher is wrong on plenty of questions, and
            # replacing the labels with it would cap the student at the teacher's
            # mistakes as well as its ceiling.
            kd = self.distillation_loss(batch, result, teacher)
            if kd is not None:
                total = total + kd_weight * (teacher.temperature ** 2) * kd
                output["kd_loss"] = kd.detach()
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
                target_ids=None, pad_token_id=self.pad_id, pad_side="left",
                query_tokens=self._compressed_query(batch))
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
        # A residual run can freeze an already-trained P decoder. Those weights
        # differ from the published model and must survive save/reload too.
        trainable |= set(getattr(self, "baseline_decoder_names", []))
        payload = {
            "format_version": 2,
            "state_dict": self.trainable_state_dict(),
            "generator_trainable": {name: value.detach().cpu()
                                    for name, value in self.lm.state_dict().items()
                                    if name in trainable},
            "config": asdict(self.cfg),
            "step": step,
            "baseline_initialization": getattr(self, "baseline_initialization", None),
            "baseline_decoder_names": sorted(getattr(self, "baseline_decoder_names", [])),
        }
        # The frozen query adapter is not trainable, so the filter above drops it
        # -- and it cannot be reconstructed from the checkpoint path alone: with
        # generator_lora_init="random" the decoder adapter is reset *before* the
        # copy is taken, and a local model directory is not an immutable version
        # either.  Store the weights and the hash so a reload restores exactly
        # what was trained against instead of whatever the path happens to hold.
        adapter = getattr(self.query_encoder, "query_adapter", None)
        if adapter is not None:
            payload["query_adapter"] = {
                "name": adapter,
                "representation": self.query_encoder.representation,
                "hash": self.query_encoder.query_adapter_hash,
                "state": {name: value.detach().cpu()
                          for name, value in self.lm.state_dict().items()
                          if f".{adapter}." in name},
            }
        payload["query_representation"] = getattr(
            self.query_encoder, "representation", None)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, path)

    def _restore_query_adapter(self, ckpt) -> None:
        """Put back the frozen query adapter, and refuse a silent mismatch.

        Rebuilding the copy from the model path is not enough to call the run
        reproducible, so the weights travel with the checkpoint.  Both directions
        are errors: loading a fixed_adapter checkpoint into a shared_current model
        would evaluate a different system under the same name, and the reverse
        would leave a frozen adapter in place that the run never trained against.
        """
        saved = ckpt.get("query_adapter")
        want = ckpt.get("query_representation")
        have = getattr(self.query_encoder, "representation", None)
        if want is not None and have is not None and want != have:
            raise ValueError(
                f"checkpoint was trained with query representation {want!r} but "
                f"this model is built with {have!r}; they are different systems")
        if saved is None:
            return
        adapter = getattr(self.query_encoder, "query_adapter", None)
        if adapter is None:
            raise ValueError(
                "checkpoint carries a frozen query adapter but this model has "
                "none; rebuild it with --query_representation fixed_adapter")
        state = {name.replace(f".{saved['name']}.", f".{adapter}."): value
                 for name, value in saved["state"].items()}
        if not state:
            raise ValueError("checkpoint's query adapter payload is empty")
        self.lm.load_state_dict(state, strict=False)
        for name, parameter in self.lm.named_parameters():
            if f".{adapter}." in name:
                parameter.requires_grad_(False)
        digest = GeneratorQueryEncoder.adapter_hash(self.lm, adapter)
        if saved.get("hash") and digest != saved["hash"]:
            raise ValueError(
                f"the frozen query adapter restored to hash {digest} but the "
                f"checkpoint recorded {saved['hash']}; the query representation "
                "does not match the one that was trained against")
        self.query_encoder.query_adapter_hash = digest

    def load(self, path, strict=False, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_kind = ckpt.get("config", {}).get("readout", {}).get("kind")
        current_kind = self.cfg.readout.kind
        if "pisco_query_writeback" in {saved_kind, current_kind} and saved_kind != current_kind:
            raise ValueError(
                f"readout kind mismatch: checkpoint={saved_kind!r}, model={current_kind!r}; "
                "RQ cannot load R/QuRO weights; use --baseline_run for a fresh P initialisation")
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=strict)
        if ckpt.get("generator_trainable"):
            self.lm.load_state_dict(ckpt["generator_trainable"], strict=False)
        self._restore_query_adapter(ckpt)
        self.baseline_initialization = ckpt.get("baseline_initialization")
        self.baseline_decoder_names = ckpt.get("baseline_decoder_names", [])
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
    if cfg.query_encoder.kind == "generator":
        from .generator import detect_adapter_name
        return GeneratorQueryEncoder(
            stack.lm, cfg.query_encoder.pooling,
            representation=cfg.query_encoder.representation,
            adapter_name=detect_adapter_name(stack.lm))
    if cfg.query_encoder.kind == "hf":
        from .hf_encoder import HFTokenEncoder
        return HFTokenEncoder(cfg.query_encoder)
    return TokenEmbeddingQueryEncoder(
        vocab_size=max(len(stack.query_tokenizer),
                       int(getattr(stack.query_tokenizer, "vocab_size", 0) or 0)),
        d_model=cfg.query_encoder.d_model,
        max_len=cfg.data.max_query_len,
        pad_id=getattr(stack.query_tokenizer, "pad_token_id", 0) or 0)
