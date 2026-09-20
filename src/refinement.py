"""Full-length, identity-initialised query refinement of PISCO latents.

This experiment changes representations, not the token budget. The identity
path preserves each original latent and its order; only the correction attends
to query tokens and other latents. No norm, projection or learned scale touches
the identity path. Zero output projection gives E=Z exactly at initialisation.
The answer CE trains the correction; there is no target residual and no KD.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from .baselines import _flatten, PiscoDirectReadout
from .perceiver import AttentionBlock
from .query_writeback import PiscoQueryWritebackReadout

# Settings that decide whether two caches hold the same kind of vector. A latent
# produced by a different compressor, rate or dtype is not a drop-in for P's, and
# loading P's decoder on top of it compares nothing.
_CACHE_PROTOCOL = ("compressor", "latent_size", "hidden_size", "dtype",
                   "compr_rate", "doc_max_length", "checkpoint")


def _check_cache_compatible(baseline_dir, run_dir):
    """A replacement cache must encode the same way and cover the same documents.

    Equal ``cache_dir`` strings used to stand in for this. The string is neither
    necessary nor sufficient: a rebuilt cache at a new path can be a valid
    superset, and the same path can be repacked with a different compressor.
    """
    manifests = {}
    for role, path in (("baseline", baseline_dir), ("run", run_dir)):
        manifest_path = os.path.join(str(path), "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(
                f"cannot verify the {role} cache: {manifest_path} is missing")
        with open(manifest_path, encoding="utf-8") as f:
            manifests[role] = json.load(f)
    for field in _CACHE_PROTOCOL:
        was, now = manifests["baseline"].get(field), manifests["run"].get(field)
        if was != now:
            raise ValueError(
                f"replacement cache encodes differently: {field} {was!r} -> {now!r}")
    missing = set(manifests["baseline"]["documents"]) - set(manifests["run"]["documents"])
    if missing:
        raise ValueError(
            f"replacement cache is not a superset of the baseline's: "
            f"{len(missing)} documents absent, e.g. {sorted(missing)[:3]}")


class PiscoResidualReadout(nn.Module):
    needs_query = True

    def __init__(self, cache_hidden, gen_hidden, query_dim, d_readout=256,
                 num_heads=8, num_blocks=1, dropout=0.0):
        super().__init__()
        if cache_hidden != gen_hidden:
            raise ValueError("identity refinement requires cache_hidden == gen_hidden")
        if num_blocks < 1:
            raise ValueError("refinement needs at least one block")
        self.latent_proj = nn.Sequential(nn.LayerNorm(cache_hidden),
                                         nn.Linear(cache_hidden, d_readout))
        self.query_proj = nn.Sequential(nn.LayerNorm(query_dim),
                                        nn.Linear(query_dim, d_readout))
        self.query_blocks = nn.ModuleList([
            AttentionBlock(d_readout, num_heads=num_heads, widening=1, dropout=dropout)
            for _ in range(num_blocks)])
        self.memory_blocks = nn.ModuleList([
            AttentionBlock(d_readout, num_heads=num_heads, widening=1, dropout=dropout)
            for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(d_readout)
        self.out_proj = nn.Linear(d_readout, gen_hidden)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, doc_latents, document_mask, query_emb=None, query_mask=None,
                budget=None, return_attn=False):
        # Like P, return ALL valid latents regardless of the requested budget.
        # token_mask, not the budget label, controls the actual decoder length.
        raw, mask = _flatten(doc_latents, document_mask)
        raw = raw.float()
        if query_emb is None:
            raise ValueError("pisco_residual requires query embeddings")
        if not bool(mask.any(-1).all()):
            raise ValueError("each row needs at least one document")
        if query_mask is not None and not bool(query_mask.any(-1).all()):
            raise ValueError("each row needs at least one query token")
        # Mask before projection as well as in attention: padded cache entries
        # must never affect valid outputs, even if their stored values are NaN.
        hidden = self.latent_proj(raw.masked_fill(~mask[..., None], 0))
        query_input = (query_emb if query_mask is None else
                       query_emb.masked_fill(~query_mask[..., None], 0))
        query = self.query_proj(query_input)
        attention = None
        for qblock, mblock in zip(self.query_blocks, self.memory_blocks):
            hidden, _ = qblock(hidden, x_kv=query, mask=query_mask)
            hidden, attention = mblock(hidden, mask=mask, return_attn=return_attn)
        delta = self.out_proj(self.out_norm(hidden)).masked_fill(~mask[..., None], 0)
        return raw + delta, {"token_mask": mask, "latent_mask": mask,
                             "attention": attention,
                             "refinement_rms": delta.detach().square().mean().sqrt()}


def initialize_from_pisco(model, checkpoint, allow_data_change=False):
    """Load only the P decoder and query weights, never its optimiser/step.

    The caller must inherit the source config. A generic non-strict model.load
    would hide an incompatible readout and could leave the query adapter copied
    from the published decoder rather than the actual trained P checkpoint.

    ``cache_dir`` and ``train_file`` are the exception: which rows the new arm
    trains on is an experimental variable, not a property of the weights being
    loaded, and the residual results made it the next variable worth moving.
    They still may not move silently -- ``allow_data_change`` has to be asked
    for, the deviation is recorded in ``baseline_initialization``, and a new
    cache is checked for compressor compatibility, which is the substantive
    thing the blanket equality was standing in for.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["config"]
    if source["readout"]["kind"] != "pisco_direct":
        raise ValueError("baseline checkpoint must use pisco_direct")
    if source["decoder"]["input_mode"] != "D0":
        raise ValueError("baseline checkpoint must use D0")
    # These change tensor shapes or the backbone itself, so they are never an
    # experimental variable for an arm that loads P's weights.
    for section, fields in {
        "generator": ("kind", "name_or_path", "n_mem_tokens", "toy_d_model", "toy_n_layer"),
        "data": ("max_docs", "max_query_len", "max_answer_len"),
    }.items():
        for field in fields:
            if field in source[section] and source[section][field] != getattr(getattr(model.cfg, section), field):
                raise ValueError(f"baseline protocol mismatch: {section}.{field}")
    deviation = {}
    for field in ("cache_dir", "train_file"):
        was, now = source["data"].get(field), getattr(model.cfg.data, field)
        if field in source["data"] and was != now:
            if not allow_data_change:
                raise ValueError(
                    f"baseline protocol mismatch: data.{field} "
                    f"({was!r} -> {now!r}); pass --allow_data_change to make the "
                    f"training data the variable, and say so when reporting")
            deviation[field] = {"baseline": was, "run": now}
    if "cache_dir" in deviation:
        _check_cache_compatible(deviation["cache_dir"]["baseline"],
                                deviation["cache_dir"]["run"])
    state = payload.get("generator_trainable", {})
    if not state:
        raise ValueError("baseline checkpoint has no saved decoder weights")
    current = model.lm.state_dict()
    bad = [name for name, value in state.items()
           if name not in current or current[name].shape != value.shape]
    if bad:
        raise ValueError(f"incompatible baseline decoder tensors: {bad[:5]}")
    # Require the complete adapter, not just a few compatible keys.
    adapter = getattr(model.query_encoder, "decoder_adapter", None)
    expected = ({n for n, _ in model.lm.named_parameters() if f".{adapter}." in n}
                if adapter else set(current))
    if not expected.issubset(state):
        raise ValueError("baseline checkpoint is missing decoder parameters")
    model.lm.load_state_dict(state, strict=False)
    model.baseline_decoder_names = sorted(state)
    # Restore a frozen query snapshot if the source had one; otherwise snapshot
    # the just-loaded decoder. P did not use a query encoder, but the new arm does.
    query_adapter = getattr(model.query_encoder, "query_adapter", None)
    if query_adapter is not None:
        saved = payload.get("query_adapter")
        if saved:
            qstate = {n.replace(f".{saved['name']}.", f".{query_adapter}."): v
                      for n, v in saved["state"].items()}
        else:
            qstate = {n.replace(f".{adapter}.", f".{query_adapter}."): v
                      for n, v in state.items() if f".{adapter}." in n}
        expected_query = {n for n, _ in model.lm.named_parameters()
                          if f".{query_adapter}." in n}
        if set(qstate) != expected_query:
            raise ValueError("baseline query snapshot is incomplete")
        model.lm.load_state_dict(qstate, strict=False)
        model.query_encoder.query_adapter_hash = model.query_encoder.adapter_hash(
            model.lm, query_adapter)
    elif model.cfg.query_encoder.kind == "toy":
        qstate = {n.removeprefix("query_encoder."): v
                  for n, v in payload["state_dict"].items()
                  if n.startswith("query_encoder.")}
        model.query_encoder.load_state_dict(qstate, strict=True)
    model.baseline_initialization = {"checkpoint": str(checkpoint), "step": payload.get("step")}
    if deviation:
        model.baseline_initialization["data_deviation"] = deviation
    return model.baseline_initialization


@torch.no_grad()
def verify_pisco_identity(model, batch):
    """Compare latent/mask/answer logits with P on the same decoder in eval mode."""
    if not isinstance(model.readout, (PiscoResidualReadout, PiscoQueryWritebackReadout)):
        raise ValueError("identity check requires an identity residual readout")
    readout, training = model.readout, model.training
    model.eval()
    try:
        loss, actual = model.qa_loss(batch, return_logits=True)
        model.readout = PiscoDirectReadout(model.cache_hidden, model.d_gen)
        ref_loss, expected = model.qa_loss(batch, return_logits=True)
        for key in ("soft_tokens", "soft_token_mask", "answer_targets", "answer_logits"):
            if not torch.equal(actual[key], expected[key]):
                raise RuntimeError(f"initial P identity failed: {key}")
        if not torch.isfinite(loss) or not torch.equal(loss, ref_loss):
            raise RuntimeError("initial P identity failed: CE")
        return {"identity": True, "ce": float(loss),
                "tokens_per_row": actual["soft_token_mask"].sum(-1).tolist()}
    finally:
        model.readout = readout
        model.train(training)
