"""Direct-CE / Direct-State / W-CE, with a frozen, external document cache.

Layer numbers in ReaderSpec are ONE-based decoder block outputs (not final norm).
The activation context MUST span backward when gradient checkpointing is enabled.
No changes to the historical QuRO model or training entry point are required.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import _flatten
from .causal_order import decoder_layers
from .prompt import BuiltPrompt, assemble_inputs


@dataclass
class ReaderSpec:
    arm: str = "direct-ce"
    layers: tuple = (8, 16, 24)
    state_weight: float = 0.1
    workspace_tokens: int = 16
    cross_dim: int = 512
    cross_heads: int = 8
    gate_init: float = 0.1

    def validate(self, depth):
        self.layers = tuple(self.layers)
        if self.arm not in {"direct-ce", "direct-state", "w-ce", "direct-read", "direct-mlp"}:
            raise ValueError("unknown reader arm")
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("layers must be nonempty, unique and increasing")
        if self.layers[0] < 1 or self.layers[-1] > depth:
            raise ValueError(f"one-based layer numbers must be within 1..{depth}")
        if self.arm == "w-ce" and self.layers[-1] >= depth:
            raise ValueError("W needs a later decoder block to transmit its update to S/A")
        if self.state_weight < 0 or self.workspace_tokens < 1:
            raise ValueError("invalid loss weight / workspace size")
        if self.cross_heads < 1 or self.cross_dim < 1 or self.cross_dim % self.cross_heads:
            raise ValueError("cross_dim must be divisible by cross_heads")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row):
    # Deliberately contains no answer. Document order and question are part of identity.
    return json_hash({"id": str(row["id"]), "query": row["query"],
                      "docs": row["retrieved_doc_ids"]})


def tokenizer_hash(tok):
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else tok.stoi
    return json_hash({"vocab": vocab, "chat_template": getattr(tok, "chat_template", None),
                      "specials": getattr(tok, "special_tokens_map", None)})


def cosine_state_loss(student, teacher):
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("state targets must match [batch, layers, hidden]")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("non-finite reader states")
    return (1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1,
                                    eps=1e-8)).mean()


class CrossRead(nn.Module):
    def __init__(self, hidden, dim, heads, gate):
        super().__init__()
        self.heads = heads
        self.norm_w = nn.LayerNorm(hidden)
        self.norm_z = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, dim, bias=False)
        self.k = nn.Linear(hidden, dim, bias=False)
        self.v = nn.Linear(hidden, dim, bias=False)
        self.out = nn.Linear(dim, hidden, bias=False)
        nn.init.zeros_(self.out.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate)))

    def forward(self, w, z, mask):
        if not mask.any(dim=1).all():
            raise ValueError("every example needs at least one valid memory slot")
        # New parameters and attention math are FP32; cast only the residual back.
        w0, z0 = self.norm_w(w.float()), self.norm_z(z.detach().float())
        def heads(x):
            return x.reshape(x.size(0), x.size(1), self.heads, -1).transpose(1, 2)
        q, k, v = heads(self.q(w0)), heads(self.k(z0)), heads(self.v(z0))
        scores = (q @ k.transpose(-1, -2)) / q.size(-1) ** 0.5
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), -torch.inf)
        read = (scores.softmax(-1) @ v).transpose(1, 2).reshape(w.size(0), w.size(1), -1)
        return w + (self.gate * self.out(read)).to(w.dtype)


class AnswerMLP(nn.Module):
    """Approximately parameter-matched control: no extra access to Z.

    Its input can already contain evidence through the original D0 backbone.
    Thus this is an adaptation-capacity control, NOT a document-free model.
    """
    def __init__(self, hidden, dim, gate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * dim, bias=False)
        self.out = nn.Linear(2 * dim, hidden, bias=False)
        nn.init.zeros_(self.out.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate)))

    def forward(self, h, _z, _mask):
        return h + (self.gate * self.out(F.silu(self.up(self.norm(h.float()))))).to(h.dtype)


class ReaderExperiment(nn.Module):
    """Hold the LM normally so .train(), .to() and parameter groups are explicit."""
    def __init__(self, base, spec):
        super().__init__()
        self.lm = base.lm
        self.tok = base.tok
        self.builders = base.prompt_builders
        self.pad_id = base.pad_id
        self.spec = spec
        self.blocks = list(decoder_layers(self.lm))  # plain list, no double registration
        spec.validate(len(self.blocks))
        self.hidden = int(self.lm.config.hidden_size)
        self.lora_names = tuple(n for n, p in self.lm.named_parameters() if p.requires_grad)
        self.cross = nn.ModuleDict()
        if spec.arm == "w-ce":
            self.workspace = nn.Parameter(torch.empty(spec.workspace_tokens, self.hidden))
            nn.init.normal_(self.workspace, std=0.02)
            self.cross.update({str(l): CrossRead(self.hidden, spec.cross_dim,
                                                spec.cross_heads, spec.gate_init)
                               for l in spec.layers})
        elif spec.arm in {"direct-read", "direct-mlp"}:
            for layer in spec.layers:
                self.cross[str(layer)] = (CrossRead(self.hidden, spec.cross_dim,
                    spec.cross_heads, spec.gate_init) if spec.arm == "direct-read"
                    else AnswerMLP(self.hidden, spec.cross_dim, spec.gate_init))
        self.decoder_frozen = False
        self._active = False

    def freeze_decoder(self):
        self.decoder_frozen = True
        self.lm.requires_grad_(False)
        self.lm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.decoder_frozen:
            self.lm.eval()  # no LoRA dropout in the frozen reference path
        return self

    def workspace_prompt(self, query):
        builder = self.builders["AG"]
        # Existing special IDs are placeholders only; every W embedding is replaced.
        slots = builder.mem_tokens[0] * self.spec.workspace_tokens
        text = builder._chat(f"Question:{query}\n\n{slots}")
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        positions = [i for i, t in enumerate(ids) if t in builder.mem_token_ids]
        if len(ids) >= builder.max_prompt_tokens:
            raise ValueError("workspace prompt exceeds prompt limit")
        if len(positions) != self.spec.workspace_tokens or positions[-1] >= len(ids) - 1:
            raise ValueError("workspace slots must be after Q and before a real template suffix")
        return BuiltPrompt(ids, positions)

    def pack(self, batch, targets=True, pad_side="right", raw=False):
        z, mask = _flatten(batch["cached_latents"], batch["document_mask"])
        z, mask = z.detach(), mask.bool()
        if z.size(-1) != self.hidden or not mask.any(1).all():
            raise ValueError("invalid document cache hidden size or empty evidence")
        if raw:
            if any(len(texts) != len(ids) for texts, ids in
                   zip(batch["document_texts"], batch["retrieved_doc_ids"])):
                raise ValueError("raw teacher is missing document text")
            prompts = [self.builders["RG"].build(q, 0, docs)
                       for q, docs in zip(batch["queries"], batch["document_texts"])]
            soft, soft_mask = z, mask
        elif self.spec.arm == "w-ce":
            prompts = [self.workspace_prompt(q) for q in batch["queries"]]
            soft = self.workspace[None].expand(len(prompts), -1, -1)
            soft_mask = torch.ones(soft.shape[:2], dtype=torch.bool, device=z.device)
        else:
            prompts = [self.builders["D0"].build(q, int(m.sum()))
                       for q, m in zip(batch["queries"], mask)]
            soft, soft_mask = z, mask
        packed = assemble_inputs(self.lm.get_input_embeddings(), prompts, soft, soft_mask,
                                 batch["target_ids"] if targets else None,
                                 pad_token_id=self.pad_id, pad_side=pad_side)
        lengths = [len(p.input_ids) for p in prompts]
        full_lengths = [n + (len(t) if targets else 0)
                        for n, t in zip(lengths, batch["target_ids"])]
        width = packed["inputs_embeds"].size(1)
        offsets = [width - n if pad_side == "left" else 0 for n in full_lengths]
        anchor = torch.tensor([o + n - 1 for o, n in zip(offsets, lengths)], device=z.device)
        wpos = None
        if self.spec.arm == "w-ce" and not raw:
            wpos = torch.tensor([[o + p for p in prompt.slot_positions]
                                 for o, prompt in zip(offsets, prompts)], device=z.device)
        return {"inputs": packed, "anchor": anchor, "wpos": wpos, "z": z,
                "zmask": mask, "prompt_lengths": lengths,
                "prompt_ids": [p.input_ids for p in prompts]}

    @contextmanager
    def activate(self, packed, capture=False):
        """Hooks remain installed through backward, including non-reentrant recompute.

        HF cached decode has sequence length one and skips W writes. At prefill,
        a write after block l affects memory K/V at l+1, exactly as in full forward.
        """
        if self._active:
            raise RuntimeError("reader contexts cannot nest")
        if self.training and getattr(self.lm, "is_gradient_checkpointing", False):
            for block in self.blocks:
                fn = getattr(block, "_gradient_checkpointing_func", None)
                if fn is not None and getattr(fn, "keywords", {}).get("use_reentrant", True):
                    raise ValueError("reader experiments require use_reentrant=False")
        self._active = True
        states, handles = {}, []
        def hook(layer):
            def run(_module, _args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                wpos = packed["wpos"]
                changed = False
                if self.spec.arm in {"direct-read", "direct-mlp"}:
                    # Prefill/teacher forcing: modify only the predictor of the first
                    # answer token and subsequent answer-side positions. Cached
                    # generation: its single new token is also an answer predictor.
                    if h.size(1) == 1 and int(packed["anchor"].min()) > 0:
                        active = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
                    else:
                        active = torch.arange(h.size(1), device=h.device)[None] >= packed["anchor"][:, None]
                        active = active & packed["inputs"]["attention_mask"][:, :h.size(1)].bool()
                    # Pack answer positions per example. Project Z once per
                    # example, not once per answer token (important at H=4096).
                    counts = active.sum(1)
                    width = int(counts.max())
                    if width:
                        offsets = torch.arange(width, device=h.device)[None]
                        starts = active.long().argmax(1)[:, None]
                        cols = (starts + offsets).clamp_max(h.size(1) - 1)
                        rows = torch.arange(h.size(0), device=h.device)[:, None].expand_as(cols)
                        keep = offsets < counts[:, None]
                        z = packed.get("branch_z", packed["z"])
                        zm = packed.get("branch_zmask", packed["zmask"])
                        values = self.cross[str(layer)](h[rows, cols], z, zm)
                        h = h.clone()
                        h[rows[keep], cols[keep]] = values[keep]
                        changed = True
                if wpos is not None and h.size(1) > int(wpos.max()):
                    rows = torch.arange(h.size(0), device=h.device)[:, None]
                    w = self.cross[str(layer)](h[rows, wpos], packed["z"], packed["zmask"])
                    h = h.clone()
                    h[rows, wpos] = w
                    changed = True
                if capture and h.size(1) > int(packed["anchor"].max()):
                    rows = torch.arange(h.size(0), device=h.device)
                    states[layer] = h[rows, packed["anchor"]]
                if changed:
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    if isinstance(output, list):
                        return [h] + output[1:]
                    return h
                return None
            return run
        try:
            for layer in self.spec.layers:
                if capture or packed["wpos"] is not None or self.spec.arm in {"direct-read", "direct-mlp"}:
                    handles.append(self.blocks[layer - 1].register_forward_hook(hook(layer)))
            yield states
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def stack_states(self, states):
        if set(states) != set(self.spec.layers):
            raise RuntimeError("missing block captures")
        return torch.stack([states[l] for l in self.spec.layers], dim=1)

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=32):
        previous = self.training
        self.eval()
        try:
            packed = self.pack(batch, targets=False, pad_side="left")
            with self.activate(packed):
                ids = self.lm.generate(**packed["inputs"], use_cache=True, do_sample=False,
                                       max_new_tokens=max_new_tokens,
                                       eos_token_id=self.tok.eos_token_id,
                                       pad_token_id=self.pad_id)
            return [self.tok.decode(row.tolist(), skip_special_tokens=True).strip() for row in ids]
        finally:
            self.train(previous)

    def checkpoint_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: v.detach().cpu().clone() for n, v in self.state_dict().items() if n in keep}

    def restore_state(self, state):
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError(f"trainable checkpoint mismatch: missing={expected-set(state)}, "
                             f"unexpected={set(state)-expected}")
        self.load_state_dict(state, strict=False)


class StateTargets:
    """FP16 memory-mapped targets: no full dataset tensor resident on GPU/RAM."""
    def __init__(self, root, expected=None):
        import numpy as np
        self.root = Path(root)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        if self.meta.get("complete") is not True or self.meta.get("split") != "train":
            raise ValueError("state cache must be complete and built on train")
        for key, value in (expected or {}).items():
            if self.meta.get(key) != value:
                raise ValueError(f"state cache mismatch: {key}")
        for filename, key in (("index.json", "index_sha256"), ("states.bin", "states_sha256")):
            if key in self.meta and file_hash(self.root / filename) != self.meta[key]:
                raise ValueError(f"state cache checksum mismatch: {filename}")
        index = json.loads((self.root / "index.json").read_text())
        self.index = index
        shape = tuple(self.meta["shape"])
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError("invalid state-cache shape")
        if len(index) != shape[0] or set(index.values()) != set(range(shape[0])):
            raise ValueError("invalid state-cache index")
        if (self.root / "states.bin").stat().st_size != 2 * int(np.prod(shape)):
            raise ValueError("truncated state-cache file")
        self.data = np.memmap(self.root / "states.bin", dtype="float16", mode="r", shape=shape)

    def gather(self, rows, device):
        import numpy as np
        try:
            indices = [self.index[row_key(row)] for row in rows]
        except KeyError as error:
            raise ValueError("teacher cache missing this ID/query/ordered-documents tuple") from error
        return torch.from_numpy(np.array(self.data[indices], copy=True)).to(device=device)
