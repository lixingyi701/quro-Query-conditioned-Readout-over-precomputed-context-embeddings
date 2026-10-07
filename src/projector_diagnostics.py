"""Eval-only interventions on aligned projector inputs; no trainable modules."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

MODES = ("original", "scale_only", "direction_only", "full")


def input_variants(z, e, mask, eps=1e-8):
    """Factor E into per-token length and direction, retaining the same slots.

    A zero source vector has no direction. For those exceptional tokens, use
    the original vector in scale_only and E in direction_only, and report them.
    Padded values (including NaN) are discarded before any arithmetic.
    """
    if z.ndim != 3 or z.shape != e.shape or z.shape[:-1] != mask.shape or not math.isfinite(eps) or eps <= 0:
        raise ValueError("interventions need aligned (B,T,H) inputs and a positive epsilon")
    mask = mask.bool()
    z = torch.where(mask[..., None], z.detach().float(), 0.)
    e = torch.where(mask[..., None], e.detach().float(), 0.)
    if not torch.isfinite(z).all() or not torch.isfinite(e).all():
        raise ValueError("nonfinite valid memory vectors")
    nz, ne = z.norm(dim=-1), e.norm(dim=-1)
    good_z, good_e = nz > eps, ne > eps
    scale = torch.where(good_z[..., None],
                        z * (ne / nz.clamp_min(eps))[..., None], z)
    direction = torch.where(good_e[..., None],
                            e * (nz / ne.clamp_min(eps))[..., None], e)
    return dict(zip(MODES, (z, scale, direction, e))), {
        "valid_tokens": int(mask.sum()),
        "original_near_zero_tokens": int((mask & ~good_z).sum()),
        "full_near_zero_tokens": int((mask & ~good_e).sum()),
    }


def geometry_values(z, e, mask, eps=1e-8):
    """Scalar measurements only; angles/ratios exclude undefined denominators."""
    variants, counts = input_variants(z, e, mask, eps)
    z, e = variants["original"][mask], variants["full"][mask]
    h = z.size(-1)
    nz, ne = z.norm(dim=-1), e.norm(dim=-1)
    delta = e-z
    nd = delta.norm(dim=-1)
    good_z, good_e = nz > eps, ne > eps
    dot = (delta*z).sum(-1) / nz.clamp_min(eps)
    tangential_sq = (nd.square()-dot.square()).clamp_min(0.)
    return {
        "original_rms": nz/math.sqrt(h), "full_rms": ne/math.sqrt(h),
        "delta_rms": nd/math.sqrt(h),
        "relative_delta_l2": (nd/nz.clamp_min(eps))[good_z],
        "cosine_z_e": ((z*e).sum(-1)/(nz*ne).clamp_min(eps**2))[good_z & good_e].clamp(-1, 1),
        "signed_radial_over_z": (dot/nz.clamp_min(eps))[good_z],
        "radial_delta_rms": (dot.abs()/math.sqrt(h))[good_z],
        "tangential_delta_rms": (tangential_sq.sqrt()/math.sqrt(h))[good_z],
        "radial_delta_energy_fraction": (dot.square()/nd.square().clamp_min(eps**2))[
            good_z & (nd > eps)].clamp(0, 1),
    }, counts


class ScalarDistributions:
    """Pool scalar token statistics, rather than averaging batch RMS values."""
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, values):
        for key, tensor in values.items():
            value = tensor.detach().float().flatten().cpu()
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite statistic: {key}")
            self.values[key].append(value)

    def summary(self):
        result = {}
        for key, parts in self.values.items():
            x = torch.cat(parts).double()
            result[key] = {"n": x.numel(), "mean": float(x.mean()) if x.numel() else None,
                           "pooled_rms": float(x.square().mean().sqrt()) if x.numel() else None}
            for name, q in (("p05", .05), ("p50", .5), ("p95", .95)):
                result[key][name] = float(torch.quantile(x, q)) if x.numel() else None
        return result


def question_positions(prompt, empty_question_prompt):
    """Actual prompt token span changed by inserting the question (D0).

    Includes boundary tokens changed by tokenization; excludes the shared chat
    suffix. This avoids mistaking contextual query-encoder outputs for word
    embeddings, or using separately tokenized/truncated question IDs.
    """
    a, b = prompt.input_ids, empty_question_prompt.input_ids
    left = 0
    while left < min(len(a), len(b)) and a[left] == b[left]:
        left += 1
    right = 0
    while right < min(len(a), len(b))-left and a[-1-right] == b[-1-right]:
        right += 1
    return list(range(left, len(a)-right))


def prompt_regions(prompts, question_spans, labels):
    """Masks for a right-padded teacher-forced forward pass."""
    memory = torch.zeros_like(labels, dtype=torch.bool)
    question = memory.clone()
    prompt_text = memory.clone()
    for i, (prompt, spans) in enumerate(zip(prompts, question_spans)):
        memory[i, prompt.slot_positions] = True
        question[i, spans] = True
        prompt_text[i, :len(prompt.input_ids)] = True
    prompt_text &= ~memory
    return {"memory": memory, "question": question, "prompt_text": prompt_text,
            "answer": labels != -100}


def answer_scores(logits, labels, eos_id):
    """Causally shifted gold content score and a separate after-gold EOS score.

    EOS must occur once at the end of each target. Answer-length normalization
    applies only to content, not to EOS or prompt/padding tokens.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels do not align")
    rows = []
    for i in range(labels.size(0)):
        pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if pos.numel() < 2 or int(pos[0]) < 1:
            raise ValueError("need a nonempty answer followed by EOS after a prompt")
        ids = labels[i, pos]
        if int(ids[-1]) != eos_id or bool((ids[:-1] == eos_id).any()):
            raise ValueError("targets must have exactly one terminal EOS")
        # Compute log-softmax only at supervised positions, not on 4096d prompts.
        logits_i = logits[i, pos-1].float()
        logp = F.log_softmax(logits_i, dim=-1)
        selected = logp.gather(-1, ids[:, None]).squeeze(-1)
        if not torch.isfinite(selected).all():
            raise ValueError("nonfinite answer score")
        eos_probs = logp[:, eos_id].exp()
        rows.append({"content_tokens": int(ids.numel()-1),
                     "content_logprob_sum": float(selected[:-1].sum()),
                     "content_logprob_mean": float(selected[:-1].mean()),
                     "eos_after_gold_logprob": float(selected[-1]),
                     "eos_after_gold_probability": float(eos_probs[-1]),
                     "eos_during_gold_mean_probability": float(eos_probs[:-1].mean())})
    return rows


def generation_record(token_ids, tokenizer, cap):
    ids = [int(x) for x in token_ids]
    if not ids or len(ids) > cap:
        raise ValueError("generate(inputs_embeds=...) must return only newly generated tokens")
    eos = tokenizer.eos_token_id
    eos_at = ids.index(eos) if eos in ids else None
    content = ids if eos_at is None else ids[:eos_at]
    return {"prediction": tokenizer.decode(content, skip_special_tokens=True).strip(),
            "generated_token_ids": ids[:eos_at+1] if eos_at is not None else ids,
            "generated_content_tokens": len(content), "eos_reached": eos_at is not None,
            "hit_generation_cap": eos_at is None and len(ids) == cap}


class DecoderTrace:
    """Hooks on real decoder residuals/updates, never hidden_states differences.

    Supports the Mistral/Llama pre-norm block layout and the CPU toy reader.
    Collects CPU scalar sums only, in the gold teacher-forced pass. Final norm
    input/output are separate measurements, not labelled residual updates.
    """
    def __init__(self, lm, regions):
        self.lm, self.regions = lm, regions
        self.handles, self.stats = [], {}

    def _record(self, name, value):
        if isinstance(value, tuple):
            value = value[0]
        if not torch.is_tensor(value) or value.shape[:2] != next(iter(self.regions.values())).shape:
            raise ValueError(f"unsupported decoder activation shape: {name}")
        self.stats[name] = {}
        for region, mask in self.regions.items():
            x = value.detach()[mask].float()
            if not torch.isfinite(x).all():
                raise ValueError(f"nonfinite decoder activation: {name}/{region}")
            self.stats[name][region] = {"sum_sq": float(x.double().square().sum()),
                                         "elements": x.numel()}

    def _pre(self, name):
        def hook(module, args, kwargs):
            self._record(name, args[0] if args else kwargs["hidden_states"])
        return hook

    def _post(self, name):
        def hook(module, args, output):
            self._record(name, output)
        return hook

    def __enter__(self):
        modules = dict(self.lm.named_modules())
        final_names, blocks = set(), 0
        try:
            for name, block in modules.items():
                if all(hasattr(block, a) for a in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")):
                    if ".layers." not in name:
                        raise ValueError("unsupported decoder layer path")
                    final_names.add(name.rsplit(".layers.", 1)[0]+".norm")
                    attn, mlp, post_attn_norm = block.self_attn, block.mlp, block.post_attention_layernorm
                elif all(hasattr(block, a) for a in ("ln1", "ln2", "attn", "ff")):
                    final_names.add("ln_f")
                    attn, mlp, post_attn_norm = block.attn, block.ff, block.ln2
                else:
                    continue
                blocks += 1
                self.handles.extend([
                    block.register_forward_pre_hook(self._pre(name+"/block_input"), with_kwargs=True),
                    block.register_forward_hook(self._post(name+"/block_output")),
                    post_attn_norm.register_forward_pre_hook(self._pre(name+"/after_attention_residual"), with_kwargs=True),
                    attn.register_forward_hook(self._post(name+"/attention_update")),
                    mlp.register_forward_hook(self._post(name+"/mlp_update")),
                ])
            if not blocks or not final_names:
                raise ValueError("--trace_decoder needs a supported pre-norm Mistral/Llama or toy decoder")
            for name in final_names:
                norm = modules[name]
                self.handles.extend([
                    norm.register_forward_pre_hook(self._pre(name+"/final_norm_input"), with_kwargs=True),
                    norm.register_forward_hook(self._post(name+"/final_norm_output")),
                ])
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def merge_trace(totals, stats):
    for stage, regions in stats.items():
        for region, values in regions.items():
            dest = totals.setdefault(stage, {}).setdefault(region, {"sum_sq": 0., "elements": 0})
            for key in dest:
                dest[key] += values[key]


def summarize_trace(totals):
    result = {}
    for stage, regions in totals.items():
        result[stage] = {}
        for region, values in regions.items():
            n = values["elements"]
            rms = math.sqrt(values["sum_sq"]/n) if n else None
            record = {**values, "rms": rms}
            if stage.endswith(("/attention_update", "/mlp_update")):
                base = stage.rsplit("/", 1)[0]
                denominator = "/block_input" if stage.endswith("/attention_update") else "/after_attention_residual"
                dv = totals[base+denominator][region]
                drms = math.sqrt(dv["sum_sq"]/dv["elements"]) if dv["elements"] else 0.
                record["update_over_residual_rms"] = rms/drms if rms is not None and drms else None
            result[stage][region] = record
    return result
