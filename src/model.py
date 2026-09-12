"""QuRO v0.0: query-conditioned readout over precomputed document latents.

Online boundary:
    doc_latents (B,K,m,h_cache) -> memory (B,K*m,d) -> B soft tokens.

The document encoder remains as a prototype/cache builder. Production
experiments should read frozen PISCO/COCOM exports from LatentCache.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .perceiver import (
    OutputQueryBuilder,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverProcessor,
    Projector,
)


class StandaloneDocEncoder(nn.Module):
    """Small trainable token encoder for the self-contained prototype."""

    def __init__(self, vocab_size, d_model, max_len, learned_pos=True, pad_id=0):
        super().__init__()
        self.out_dim = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model) if learned_pos else None
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[pad_id].zero_()
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, ids, mask=None):
        x = self.tok_emb(ids)
        if self.pos_emb is not None:
            x = x[:, : self.max_len]
            pos = torch.arange(x.size(1), device=ids.device)
            x = x + self.pos_emb(pos)[None]
        return self.norm(x)


class GeneratorEmbeddingDocEncoder(nn.Module):
    """Prototype encoder that reuses frozen generator token embeddings."""

    def __init__(self, generator, max_len, learned_pos=True):
        super().__init__()
        self._emb = [generator.get_input_embeddings()]
        self.out_dim = self._emb[0].embedding_dim
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, self.out_dim) if learned_pos else None
        self.norm = nn.LayerNorm(self.out_dim)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    @property
    def emb(self):
        return self._emb[0]

    def forward(self, ids, mask=None):
        with torch.no_grad():
            x = self.emb(ids)
        x = x.float()[:, : self.max_len]
        if self.pos_emb is not None:
            pos = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_emb(pos)[None]
        return self.norm(x)


class QueryBudgetSelector(nn.Module):
    """Predict one discrete output-token budget from the query."""

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


class PerceiverRAGCompressor(nn.Module):
    """Cache-first QuRO model with an optional prototype offline encoder."""

    def __init__(self, cfg, generator, tokenizer, enc_tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.cache_only = False
        self.tok = tokenizer
        self.enc_tok = enc_tokenizer if enc_tokenizer is not None else tokenizer
        self._generator = [generator]
        self.d_gen = int(generator.config.hidden_size)

        p, dc = cfg.perceiver, cfg.doc_encoder
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.enc_pad_id = getattr(self.enc_tok, "pad_token_id", 0) or 0
        enc_vocab = max(len(self.enc_tok), int(getattr(self.enc_tok, "vocab_size", 0) or 0))
        if dc.kind == "standalone":
            self.doc_encoder = StandaloneDocEncoder(
                enc_vocab, dc.d_model, dc.max_doc_len, dc.learned_pos, self.enc_pad_id)
        elif dc.kind == "hf_encoder":
            from .hf_encoder import HFDocEncoder
            self.doc_encoder = HFDocEncoder(dc)
        else:
            self.doc_encoder = GeneratorEmbeddingDocEncoder(
                generator, dc.max_doc_len, dc.learned_pos)
        query_dim = self.doc_encoder.out_dim

        # Prototype compressor; cached PISCO/COCOM latents bypass these modules.
        self.encoder = PerceiverEncoder(
            p.num_latents, p.d_latent, query_dim,
            num_heads=p.enc_num_heads, head_dim=p.enc_head_dim,
            num_blocks=p.enc_num_blocks, self_per_block=p.enc_self_per_block,
            share_weights=p.enc_share_weights, cross_widening=p.cross_ff_widening,
            self_widening=p.ff_widening, dropout=p.dropout)
        self.processor = PerceiverProcessor(
            p.d_latent, p.proc_num_layers, p.proc_num_heads,
            widening=p.ff_widening, dropout=p.dropout)

        cache_dim = p.cached_hidden_size or p.d_latent
        self.cache_projector = (nn.Identity() if cache_dim == p.d_latent
                                else nn.Linear(cache_dim, p.d_latent))
        self.document_source = (nn.Embedding(p.max_document_sources, p.d_latent)
                                if p.add_document_source else None)
        if self.document_source is not None:
            nn.init.normal_(self.document_source.weight, std=0.02)

        self.output_query = OutputQueryBuilder(
            p.d_latent, p.num_compressed, query_dim, p.output_query_mode,
            p.dec_num_heads, p.dropout)
        self.decoder = PerceiverDecoder(
            p.d_latent, p.dec_num_heads, p.dec_head_dim,
            p.cross_ff_widening, p.dropout)
        if cfg.projector.kind == "identity":
            if p.d_latent != self.d_gen:
                raise ValueError("identity projector requires d_latent == generator hidden_size")
            self.projector = nn.Identity()
        else:
            self.projector = Projector(
                p.d_latent, self.d_gen, cfg.projector.kind,
                cfg.projector.hidden_mult, cfg.projector.dropout)
        self.budget_selector = QueryBudgetSelector(query_dim, p.budget_buckets)
        if not p.adaptive_budget:
            for parameter in self.budget_selector.parameters():
                parameter.requires_grad_(False)
        self.bos_id = getattr(tokenizer, "bos_token_id", None)

    @property
    def generator(self):
        return self._generator[0]

    @property
    def gen_dtype(self):
        return self.generator.get_input_embeddings().weight.dtype

    def num_trainable(self):
        own = sum(x.numel() for x in self.parameters() if x.requires_grad)
        gen = sum(x.numel() for x in self.generator.parameters() if x.requires_grad)
        return own + gen

    def set_cache_only(self):
        """Freeze the offline prototype and query encoder in cache-backed training."""
        self.cache_only = True
        for module in (self.doc_encoder, self.encoder, self.processor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cache_only:
            for module in (self.doc_encoder, self.encoder, self.processor):
                module.eval()
        return self

    def _encode_query(self, ids, mask):
        max_len = self.cfg.data.max_query_len
        return self.doc_encoder(ids[:, :max_len], None if mask is None else mask[:, :max_len])

    # Offline prototype -------------------------------------------------
    def encode_document(self, doc_ids, doc_mask, return_attn=False):
        max_len = self.cfg.doc_encoder.max_doc_len
        doc_ids, doc_mask = doc_ids[:, :max_len], doc_mask[:, :max_len]
        doc_emb = self.doc_encoder(doc_ids, doc_mask)
        latents, attn = self.encoder(doc_emb, doc_mask, return_attn=return_attn)
        return self.processor(latents), attn

    def encode_documents(self, input_ids, token_mask, document_mask):
        """Encode documents independently: (B,K,L) -> (B,K,m,d)."""
        if input_ids.ndim != 3 or token_mask.shape != input_ids.shape:
            raise ValueError("input_ids/token_mask must have shape (B,K,L)")
        if document_mask.shape != input_ids.shape[:2]:
            raise ValueError("document_mask must have shape (B,K)")
        b, k, length = input_ids.shape
        valid = document_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        out = torch.zeros(b * k, self.cfg.perceiver.num_latents,
                          self.cfg.perceiver.d_latent, device=input_ids.device)
        if valid.numel():
            ids = input_ids.reshape(b * k, length).index_select(0, valid)
            mask = token_mask.reshape(b * k, length).index_select(0, valid)
            encoded, _ = self.encode_document(ids, mask)
            out.index_copy_(0, valid, encoded)
        return out.view(b, k, self.cfg.perceiver.num_latents, -1)

    @torch.no_grad()
    def cache_document(self, doc_ids, doc_mask):
        return self.encode_document(doc_ids, doc_mask)[0]

    # Online readout ----------------------------------------------------
    def prepare_latent_memory(self, doc_latents, document_mask):
        if doc_latents.ndim != 4:
            raise ValueError("doc_latents must have shape (B,K,m,h)")
        b, k, m, _ = doc_latents.shape
        if document_mask.shape != (b, k):
            raise ValueError(f"document_mask must be {(b, k)}, got {tuple(document_mask.shape)}")
        if k > self.cfg.perceiver.max_document_sources:
            raise ValueError("retrieved K exceeds max_document_sources")
        x = self.cache_projector(doc_latents.float())
        if self.document_source is not None:
            ranks = torch.arange(k, device=x.device)
            x = x + self.document_source(ranks)[None, :, None, :]
        latent_mask = document_mask[:, :, None].expand(b, k, m).reshape(b, k * m)
        return x.reshape(b, k * m, -1), latent_mask

    def _resolve_budgets(self, query_emb, query_mask, budget, batch_size):
        logits = None
        device = query_mask.device
        predicted = None
        if self.cfg.perceiver.adaptive_budget:
            if query_emb is None:
                raise ValueError("adaptive budget requires a query representation")
            predicted, logits = self.budget_selector(query_emb, query_mask)
        if budget is None and predicted is not None:
            budgets = predicted
        elif budget is None:
            budgets = torch.full((batch_size,), self.cfg.perceiver.num_compressed,
                                 dtype=torch.long, device=device)
        elif isinstance(budget, int):
            budgets = torch.full((batch_size,), budget, dtype=torch.long, device=device)
        else:
            budgets = torch.as_tensor(budget, dtype=torch.long, device=device)
            if budgets.ndim == 0:
                budgets = budgets.expand(batch_size)
        if budgets.shape != (batch_size,):
            raise ValueError(f"budget must resolve to shape ({batch_size},)")
        allowed = set(self.cfg.perceiver.budget_buckets)
        if any(int(x) not in allowed for x in budgets.detach().cpu()):
            raise ValueError(f"budgets must come from discrete buckets {sorted(allowed)}")
        return budgets, logits

    def readout_cached(
        self,
        doc_latents,
        document_mask,
        query_ids,
        query_mask,
        budget: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
        return_attn=False,
    ):
        """Online QuRO path with attention cost O(B*K*m)."""
        mode = self.cfg.perceiver.output_query_mode
        needs_query = mode != "agnostic" or self.cfg.perceiver.adaptive_budget
        query_emb = self._encode_query(query_ids, query_mask) if needs_query else None
        budgets, budget_logits = self._resolve_budgets(
            query_emb, query_mask, budget, doc_latents.size(0))
        max_budget = int(budgets.max().item())
        output_queries = self.output_query(
            query_emb, query_mask, batch_size=doc_latents.size(0), num_outputs=max_budget)
        memory, latent_mask = self.prepare_latent_memory(doc_latents, document_mask)
        readout, attention = self.decoder(
            output_queries, memory, latent_mask=latent_mask, return_attn=return_attn)
        soft_tokens = self.projector(readout)
        output_mask = (torch.arange(max_budget, device=soft_tokens.device)[None, :]
                       < budgets[:, None])
        return {
            "soft_tokens": soft_tokens,
            "soft_token_mask": output_mask,
            "budgets": budgets,
            "budget_logits": budget_logits,
            "attention": attention,
            "latent_mask": latent_mask,
        }

    def compress_with_query(self, latents, query_ids=None, query_mask=None,
                            return_attn=False):
        """Backward-compatible one-document prototype wrapper."""
        if query_ids is None or query_mask is None:
            raise ValueError("query_ids and query_mask are required")
        doc_mask = torch.ones(latents.size(0), 1, dtype=torch.bool, device=latents.device)
        result = self.readout_cached(latents[:, None], doc_mask, query_ids, query_mask,
                                     return_attn=return_attn)
        return result["soft_tokens"], result["attention"]

    # Generator bridge -------------------------------------------------
    def _assemble(self, mem, mem_mask, prefix_ids, target_ids, add_bos=True):
        emb_layer = self.generator.get_input_embeddings()
        device, dtype = mem.device, self.gen_dtype
        seqs, labels = [], []
        for i in range(mem.size(0)):
            valid_mem = mem[i][mem_mask[i]].to(dtype)
            parts, lab = [], []
            if add_bos and self.bos_id is not None:
                token = torch.tensor([self.bos_id], device=device)
                parts.append(emb_layer(token))
                lab.append(-100)
            parts.append(valid_mem)
            lab += [-100] * valid_mem.size(0)
            if prefix_ids[i]:
                ids = torch.tensor(prefix_ids[i], device=device)
                parts.append(emb_layer(ids))
                lab += [-100] * len(prefix_ids[i])
            if target_ids[i]:
                ids = torch.tensor(target_ids[i], device=device)
                parts.append(emb_layer(ids))
                lab += list(target_ids[i])
            seqs.append(torch.cat(parts))
            labels.append(lab)
        tmax = max(x.size(0) for x in seqs)
        inputs = torch.zeros(len(seqs), tmax, self.d_gen, device=device, dtype=dtype)
        mask = torch.zeros(len(seqs), tmax, device=device, dtype=torch.long)
        target = torch.full((len(seqs), tmax), -100, device=device, dtype=torch.long)
        for i, seq in enumerate(seqs):
            n = seq.size(0)
            inputs[i, :n] = seq
            mask[i, :n] = 1
            target[i, :n] = torch.tensor(labels[i], device=device)
        return {"inputs_embeds": inputs, "attention_mask": mask, "labels": target}

    def _batch_latents(self, batch):
        if "cached_latents" in batch:
            return batch["cached_latents"], batch["document_mask"]
        required = {"document_input_ids", "document_token_mask", "document_mask"}
        if not required.issubset(batch):
            raise KeyError("batch needs cached_latents or prototype document tensors")
        return self.encode_documents(
            batch["document_input_ids"], batch["document_token_mask"],
            batch["document_mask"]), batch["document_mask"]

    def qa_loss(self, batch, return_attn=False):
        latents, doc_mask = self._batch_latents(batch)
        result = self.readout_cached(
            latents, doc_mask, batch["query_ids"], batch["query_mask"],
            budget=batch.get("budget"), return_attn=return_attn)
        packed = self._assemble(
            result["soft_tokens"], result["soft_token_mask"],
            batch["prompt_ids"], batch["target_ids"])
        return self.generator(**packed).loss, result["attention"], result

    def forward(self, batch, beta=1.0):
        qa, _, result = self.qa_loss(batch)
        total = beta * qa
        output = {"qa_loss": qa.detach(),
                  "mean_budget": result["budgets"].float().mean().detach()}
        if result["budget_logits"] is not None and "budget" in batch:
            buckets = self.budget_selector.buckets
            targets = (batch["budget"][:, None] == buckets[None, :]).long().argmax(-1)
            budget_loss = torch.nn.functional.cross_entropy(result["budget_logits"], targets)
            total = total + self.cfg.train.budget_loss_weight * budget_loss
            output["budget_loss"] = budget_loss.detach()
        output["loss"] = total
        return output

    @torch.no_grad()
    def generate_answer(self, batch, max_new_tokens=24):
        was_training, gen_was_training = self.training, self.generator.training
        self.eval()
        self.generator.eval()
        try:
            latents, doc_mask = self._batch_latents(batch)
            result = self.readout_cached(
                latents, doc_mask, batch["query_ids"], batch["query_mask"],
                budget=batch.get("budget"))
            outputs = []
            for i in range(result["soft_tokens"].size(0)):
                packed = self._assemble(
                    result["soft_tokens"][i:i+1], result["soft_token_mask"][i:i+1],
                    [batch["prompt_ids"][i]], [[]])
                ids = self.generator.generate(
                    inputs_embeds=packed["inputs_embeds"],
                    attention_mask=packed["attention_mask"],
                    max_new_tokens=max_new_tokens, do_sample=False,
                    eos_token_id=getattr(self.tok, "eos_token_id", None),
                    pad_token_id=getattr(self.tok, "pad_token_id", None))
                outputs.append(self.tok.decode(ids[0].tolist(), skip_special_tokens=True))
            return outputs
        finally:
            self.train(was_training)
            self.generator.train(gen_was_training)

    # Checkpointing -----------------------------------------------------
    def trainable_state_dict(self):
        keep = {name for name, value in self.named_parameters() if value.requires_grad}
        keep |= {name for name, _ in self.named_buffers()
                 if not name.startswith("doc_encoder.backbone.")}
        return {name: value for name, value in self.state_dict().items() if name in keep}

    def save(self, path, optimizer=None, scheduler=None, step=None):
        trainable_names = {name for name, p in self.generator.named_parameters()
                           if p.requires_grad}
        generator_trainable = {
            name: value.detach().cpu() for name, value in self.generator.state_dict().items()
            if name in trainable_names
        }
        payload = {"format_version": 1, "state_dict": self.trainable_state_dict(),
                   "generator_trainable": generator_trainable,
                   "config": asdict(self.cfg), "step": step}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, path)

    def load(self, path, strict=False, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=strict)
        if ckpt.get("generator_trainable"):
            self.generator.load_state_dict(ckpt["generator_trainable"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        missing = [x for x in missing if not x.startswith("doc_encoder.backbone.")]
        return missing, unexpected, ckpt.get("step")


def build_model(cfg):
    from .generator import build_tokenizer_and_generator
    tok, generator = build_tokenizer_and_generator(cfg)
    enc_tok = None
    if cfg.doc_encoder.kind == "hf_encoder":
        from .hf_encoder import build_encoder_tokenizer
        enc_tok = build_encoder_tokenizer(cfg.doc_encoder)
    return tok, generator, PerceiverRAGCompressor(cfg, generator, tok, enc_tok)
