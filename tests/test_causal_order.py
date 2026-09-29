"""CPU contract tests for the D0/D2 causal-order diagnostic.

Run with ``python tests/test_causal_order.py``.  No model weights or downloads are
needed; these tests protect the statistic definitions and the test->train gate.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import causal_order as co

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'} {name} {detail}")


def make_trace(memory, query, memory_delta=0.1, query_delta=0.2):
    memory = torch.as_tensor(memory, dtype=torch.float32)
    query = torch.as_tensor(query, dtype=torch.float32)
    return co.BlockTrace(
        memory_in=memory,
        memory_out=memory + memory_delta,
        query_in=query,
        query_out=query + query_delta,
    )


def test_statistics():
    # [L,S,H] with non-collinear states so cosine changes are measurable.
    memory = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0]],
    ])
    query = torch.tensor([
        [[1.0, 2.0], [2.0, 1.0]],
        [[2.0, 3.0], [3.0, 2.0]],
    ])
    original = make_trace(memory, query)

    # D2-like query intervention: memory states change when the query changes.
    query_swap = make_trace(memory, query)
    query_swap.memory_out = query_swap.memory_out + torch.tensor([0.3, -0.2])
    # D0-like memory intervention: query states change when memory changes.
    memory_swap = make_trace(memory, query)
    memory_swap.query_out = memory_swap.query_out + torch.tensor([-0.4, 0.1])

    stats = co.trace_statistics(original, query_swap, memory_swap)
    check("delta_memory has one value per layer", stats["delta_memory"].shape == (2,))
    check("delta_query has one value per layer", stats["delta_query"].shape == (2,))
    check("query swap changes memory sensitivity", bool((stats["memory_query_cos"] > 0).all()))
    check("memory swap changes query sensitivity", bool((stats["query_memory_cos"] > 0).all()))

    invariant = co.cosine_sensitivity(original.memory_out, original.memory_out.clone())
    check("identical causal prefix has zero cosine sensitivity",
          bool((np.abs(invariant) < 2e-7).all()), str(invariant))


def test_gate():
    rng = np.random.default_rng(7)
    d0 = np.abs(rng.normal(1e-6, 2e-7, size=(48, 8)))
    d2 = np.abs(rng.normal(2e-3, 2e-4, size=(48, 8)))
    go = co.topology_gate(d0, d2, min_pairs=32, min_sensitivity=1e-4,
                          min_fold=3.0, min_layer_fraction=0.25, seed=3)
    check("robust D2 query-conditioning opens the function-test gate",
          go["decision"] == "GO_FUNCTION_TEST", str(go["decision"]))
    check("QA is explicitly absent from the gate", go["qa_is_gate"] is False)

    hold = co.topology_gate(d0, d0 * 1.01, min_pairs=32, min_sensitivity=1e-4,
                            min_fold=3.0, min_layer_fraction=0.25, seed=3)
    check("near-floor effect does not justify a training run",
          hold["decision"] == "HOLD_DIAGNOSTIC", str(hold["decision"]))

    invalid = d2.copy()
    invalid[0, 0] = float("nan")
    rerun = co.topology_gate(d0, invalid)
    check("non-finite sensitivity invalidates the run",
          rerun["decision"] == "RERUN_INVALID", str(rerun["decision"]))
    try:
        co.paired_bootstrap([1.0, float("inf")], [0.0, 0.0])
    except ValueError:
        check("bootstrap rejects non-finite pairs", True)
    else:
        check("bootstrap rejects non-finite pairs", False)

    small = co.topology_gate(d0[:8], d2[:8], min_pairs=32, min_sensitivity=1e-4,
                             min_fold=3.0, min_layer_fraction=0.25, seed=3)
    check("too few pairs cannot open the training gate",
          small["decision"] == "HOLD_DIAGNOSTIC", str(small["decision"]))


if __name__ == "__main__":
    test_statistics()
    test_gate()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
        raise SystemExit(1)
