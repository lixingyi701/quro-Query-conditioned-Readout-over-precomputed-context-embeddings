"""CPU-only document ranking diagnostics; no torch/model/cache dependency."""
import math


def rank_support(logits, labels, mask):
    if not (len(logits) == len(labels) == len(mask)):
        raise ValueError("support logits, labels and mask must have equal lengths")
    valid = [i for i, keep in enumerate(mask) if keep]
    if any(labels[i] not in (0, 1) for i in valid):
        raise ValueError("support labels must be binary at labelled positions")
    if any(not math.isfinite(logits[i]) for i in valid):
        raise ValueError("nonfinite support logit at a labelled document")
    gold = {i for i in valid if labels[i] == 1}
    ranked = sorted(valid, key=lambda i: (-logits[i], i))
    return ranked, gold


def support_scores(logits, labels, mask, topks=(2, 4, 6)):
    """Recall@k and both@k over ORIGINAL labelled golds, not the loss mask.

    both@k is defined only for exactly two labelled golds. For k>N all N
    valid documents are selected. Ties use original document index.
    """
    topks = tuple(topks)
    if any(type(k) is not int or k < 1 for k in topks):
        raise ValueError("top-k values must be positive integers")
    ranked, gold = rank_support(logits, labels, mask)
    if not gold:
        return None
    result = {}
    for k in topks:
        chosen = set(ranked[:k])
        result[f"recall_at_{k}"] = len(chosen & gold)/len(gold)
        result[f"both_at_{k}"] = float(gold <= chosen) if len(gold) == 2 else None
    result["exact_at_gold_k"] = float(set(ranked[:len(gold)]) == gold)
    return result
