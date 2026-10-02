"""Utilities for the D0/D2 causal-order diagnostic.

The question is not whether a PISCO decoder that was trained with memory-before-
question immediately scores better when its prompt is reversed.  The first-stage
experiment asks a narrower mechanistic question: does causal order move query-
conditioned computation from the query states to the memory states?

This module deliberately works with *decoder block inputs/outputs* rather than
``output.hidden_states`` so the final RMSNorm cannot be mistaken for a decoder
block update (see ``LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md`` W3).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BlockTrace:
    """Prompt-position residual states captured around every decoder block.

    Each tensor has shape ``[L, S, H]`` where ``S`` is the number of positions in
    the named group.  Only memory/query positions are retained; answer positions
    are never stored, which keeps one 7B diagnostic row small enough to run in a
    loop without materialising all hidden states.
    """

    memory_in: torch.Tensor
    memory_out: torch.Tensor
    query_in: torch.Tensor
    query_out: torch.Tensor

    @property
    def n_layers(self) -> int:
        return int(self.memory_in.size(0))


def decoder_layers(lm) -> Sequence[torch.nn.Module]:
    """Return the causal decoder blocks for bare or PEFT-wrapped Mistral models."""
    if hasattr(lm, "get_decoder"):
        decoder = lm.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    base = getattr(lm, "base_model", None)
    if base is not None:
        model = getattr(base, "model", None)
        if model is not None and hasattr(model, "layers"):
            return model.layers
    raise ValueError("cannot locate decoder layers on the supplied language model")


def _hidden_from_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@contextmanager
def capture_block_trace(lm, memory_positions: Sequence[int], query_positions: Sequence[int]):
    """Capture real block input/output states at the two causal groups.

    The context yields a mutable dict.  After a single forward pass it contains a
    ``trace`` key holding :class:`BlockTrace`.  Hooks are removed even if the
    forward raises.
    """
    layers = list(decoder_layers(lm))
    if not memory_positions:
        raise ValueError("the causal-order probe needs at least one memory position")
    if not query_positions:
        raise ValueError("the causal-order probe needs at least one query position")

    store: Dict[str, object] = {"memory_in": [], "memory_out": [],
                               "query_in": [], "query_out": []}
    handles = []

    def take(hidden: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        index = torch.as_tensor(list(positions), device=hidden.device, dtype=torch.long)
        return hidden[0].index_select(0, index).detach().float().cpu()

    def pre_hook(_module, args):
        hidden = args[0]
        store["memory_in"].append(take(hidden, memory_positions))
        store["query_in"].append(take(hidden, query_positions))

    def post_hook(_module, _args, output):
        hidden = _hidden_from_output(output)
        store["memory_out"].append(take(hidden, memory_positions))
        store["query_out"].append(take(hidden, query_positions))

    for layer in layers:
        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))

    box: Dict[str, object] = {}
    try:
        yield box
    finally:
        for handle in handles:
            handle.remove()
        lengths = {k: len(store[k]) for k in store}
        if any(v != len(layers) for v in lengths.values()):
            raise RuntimeError(
                f"captured block counts {lengths}, expected {len(layers)} each; "
                "the model did not execute every decoder block exactly once")
        box["trace"] = BlockTrace(**{
            k: torch.stack(store[k], dim=0) for k in
            ("memory_in", "memory_out", "query_in", "query_out")
        })


def relative_update(before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    """Mean ``||h_out-h_in||/||h_in||`` over positions, one value per block."""
    if before.shape != after.shape:
        raise ValueError(f"update tensors differ: {before.shape} vs {after.shape}")
    delta = (after - before).norm(dim=-1)
    denom = before.norm(dim=-1).clamp_min(1e-9)
    return (delta / denom).mean(dim=-1).cpu().numpy().astype(np.float64)


def cosine_sensitivity(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Mean ``1-cos`` between matched positions, one value per decoder block."""
    if a.shape != b.shape:
        raise ValueError(f"sensitivity tensors differ: {a.shape} vs {b.shape}")
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean(dim=-1
            ).cpu().numpy().astype(np.float64)


def relative_l2_sensitivity(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Symmetric relative L2 distance between matched states, per block."""
    if a.shape != b.shape:
        raise ValueError(f"sensitivity tensors differ: {a.shape} vs {b.shape}")
    numerator = (a - b).norm(dim=-1)
    denominator = 0.5 * (a.norm(dim=-1) + b.norm(dim=-1)).clamp_min(1e-9)
    return (numerator / denominator).mean(dim=-1).cpu().numpy().astype(np.float64)


def trace_statistics(original: BlockTrace, query_swap: BlockTrace,
                     memory_swap: BlockTrace) -> Dict[str, np.ndarray]:
    """The four measurements pre-registered for each D0/D2 row.

    ``memory_query_*`` changes the query while holding the memory fixed.  It is
    the primary D2 test.  ``query_memory_*`` changes the memory while holding the
    query fixed.  It is the complementary direction check.
    """
    stats = {
        "delta_memory": relative_update(original.memory_in, original.memory_out),
        "delta_query": relative_update(original.query_in, original.query_out),
        "memory_query_cos": cosine_sensitivity(original.memory_out,
                                                query_swap.memory_out),
        "memory_query_rel_l2": relative_l2_sensitivity(original.memory_out,
                                                      query_swap.memory_out),
        "query_memory_cos": cosine_sensitivity(original.query_out,
                                                memory_swap.query_out),
        "query_memory_rel_l2": relative_l2_sensitivity(original.query_out,
                                                        memory_swap.query_out),
    }
    for name, values in stats.items():
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite trace statistic {name}; rerun the diagnostic")
    return stats


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 4000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired 95% bootstrap CI for mean ``a-b`` across examples."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired bootstrap needs equal shapes, got {x.shape} and {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired bootstrap received non-finite values; rerun the diagnostic")
    if x.size == 0:
        raise ValueError("paired bootstrap needs at least one pair")
    d = x - y
    rng = np.random.default_rng(seed)
    # A full 30k-row target audit otherwise allocates ~1.9GB for index + d[index].
    # Chunking preserves the RNG stream and the per-resample reduction order.
    means = np.concatenate([
        d[rng.integers(0, d.size, size=(min(64, n_resamples-start), d.size))].mean(axis=1)
        for start in range(0, n_resamples, 64)
    ])
    return {"delta": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(d.size)}


def topology_gate(d0: np.ndarray, d2: np.ndarray, *, min_pairs: int = 32,
                  min_sensitivity: float = 1e-4, min_fold: float = 3.0,
                  min_layer_fraction: float = 0.25, seed: int = 0) -> Dict[str, object]:
    """Diagnostic gate for a D2 query-conditioned memory-state signal.

    ``d0`` and ``d2`` are ``[N,L]`` memory-query cosine sensitivities.  QA is intentionally absent: zero-shot D2 is interface-mismatched evidence.
    Passing this gate licenses a task-functional intervention, not training.

    GO requires all of:
      1. at least ``min_pairs`` position-matched counterfactual pairs;
      2. D2's layer-averaged sensitivity exceeds D0 with paired 95% CI > 0;
      3. the mean D2 signal clears both an absolute floor and ``min_fold`` over D0;
      4. the paired CI is > 0 in at least ``min_layer_fraction`` of decoder blocks.

    Anything else is HOLD rather than a negative mechanism claim.  Positional
    matching and per-trace finite-value checks are performed by the caller.
    This function also rejects non-finite sensitivity arrays rather than dropping pairs.
    """
    d0 = np.asarray(d0, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    if d0.shape != d2.shape or d0.ndim != 2:
        raise ValueError(f"expected matched [N,L] arrays, got {d0.shape} and {d2.shape}")

    if not np.isfinite(d0).all() or not np.isfinite(d2).all():
        return {
            "decision": "RERUN_INVALID",
            "reason": "Non-finite sensitivity values; inspect traces and rerun.",
            "n_pairs": int(d0.shape[0]),
            "thresholds": {
                "min_pairs": int(min_pairs),
                "min_sensitivity": float(min_sensitivity),
                "min_fold": float(min_fold),
                "min_layer_fraction": float(min_layer_fraction),
            },
            "qa_is_gate": False,
        }

    per_row_d0 = d0.mean(axis=1)
    per_row_d2 = d2.mean(axis=1)
    overall = paired_bootstrap(per_row_d2, per_row_d0, seed=seed)
    mean_d0 = float(np.mean(per_row_d0))
    mean_d2 = float(np.mean(per_row_d2))
    fold = mean_d2 / max(mean_d0, 1e-12)

    layer_ci = [paired_bootstrap(d2[:, l], d0[:, l], seed=seed + l + 1)
                for l in range(d0.shape[1])]
    positive_layers = sum(int(ci["lo"] > 0) for ci in layer_ci)
    layer_fraction = positive_layers / max(1, d0.shape[1])

    enough = int(overall["n"]) >= int(min_pairs)
    robust = bool(overall["lo"] > 0)
    magnitude = bool(mean_d2 >= min_sensitivity and fold >= min_fold)
    distributed = bool(layer_fraction >= min_layer_fraction)
    go = enough and robust and magnitude and distributed

    return {
        "decision": "GO_FUNCTION_TEST" if go else "HOLD_DIAGNOSTIC",
        "reason": ("D2 creates a robust, position-controlled query-conditioned memory "
                   "signal; test whether this state affects answer quality before training."
                   if go else
                   "The topology signal is not established in this checkpoint; inspect "
                   "controls and do not infer that a trained D2 model cannot use it."),
        "n_pairs": int(overall["n"]),
        "mean_d0_memory_query_cos": mean_d0,
        "mean_d2_memory_query_cos": mean_d2,
        "d2_over_d0_fold": float(fold),
        "paired_d2_minus_d0": overall,
        "positive_layer_fraction": float(layer_fraction),
        "positive_layers": int(positive_layers),
        "n_layers": int(d0.shape[1]),
        "thresholds": {
            "min_pairs": int(min_pairs),
            "min_sensitivity": float(min_sensitivity),
            "min_fold": float(min_fold),
            "min_layer_fraction": float(min_layer_fraction),
        },
        "layerwise_d2_minus_d0": layer_ci,
        "qa_is_gate": False,
    }
