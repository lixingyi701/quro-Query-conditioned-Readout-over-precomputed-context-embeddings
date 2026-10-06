"""Aggregate masked fusion statistics without averaging microbatch RMS values."""
import math


def add_fusion_stats(total, stats):
    """Sum detached scalar tensors (or numbers); no host sync is required here."""
    for key, value in stats.items():
        total[key] = total.get(key, 0) + value


def fusion_metrics(stats):
    if not stats:
        return {}
    n = float(stats["elements"])
    if not math.isfinite(n) or n <= 0:
        raise ValueError("fusion statistics need positive valid element count")
    rms = {}
    for name in ("h", "b", "gamma", "product", "update"):
        squared_sum = float(stats[name + "_sum_sq"])
        if not math.isfinite(squared_sum) or squared_sum < 0:
            raise ValueError(f"nonfinite or negative fusion statistic: {name}")
        rms[name] = math.sqrt(squared_sum / n)
    result = {"fusion_" + name + "_rms": value for name, value in rms.items()}
    # An exactly zero context has no meaningful relative amplitude. JSON null
    # conveys that fact without inventing a denominator or writing Infinity.
    for name in ("product", "update"):
        result[f"fusion_{name}_over_b_rms"] = rms[name] / rms["b"] if rms["b"] else None
    return result


def paired_change_stats(aux, reference_aux, memory, document_mask):
    """Sufficient statistics for the change of gamma and E against a reference readout.

    Both readouts share the cached memory Z, so E - E_ref = delta - delta_ref.
    Only valid documents count; sums are pooled across batches, never averaged.
    """
    import torch
    dm = document_mask.bool()
    with torch.no_grad():
        diff = (aux["delta"] - reference_aux["delta"]).float()
        stats = {"documents": dm.sum(),
                 "delta_diff_sum_sq": diff[dm].square().sum(),
                 "delta_ref_sum_sq": reference_aux["delta"].float()[dm].square().sum(),
                 "memory_sum_sq": memory.float()[dm].square().sum()}
        gamma, gamma_ref = aux.get("gamma"), reference_aux.get("gamma")
        if gamma is not None and gamma_ref is not None:
            g, r = gamma.float()[dm], gamma_ref.float()[dm]
            stats["gamma_diff_sum_sq"] = (g - r).square().sum()
            stats["gamma_ref_sum_sq"] = r.square().sum()
            norms = g.norm(dim=-1) * r.norm(dim=-1)
            cosine = torch.where(norms > 0, (g * r).sum(-1) / norms.clamp_min(1e-12),
                                 torch.full_like(norms, float("nan")))
            stats["gamma_cosine_sum"] = torch.nan_to_num(cosine, nan=0.0).sum()
            stats["gamma_cosine_count"] = (norms > 0).sum()
    return stats


def paired_change_metrics(stats):
    """Relative changes; null where the reference is exactly zero (e.g. gamma at init)."""
    if not stats:
        return {}

    def ratio(numerator, denominator):
        den = float(stats[denominator])
        return math.sqrt(float(stats[numerator]) / den) if den > 0 else None

    result = {"change_documents": int(stats["documents"]),
              "change_e_over_delta_ref": ratio("delta_diff_sum_sq", "delta_ref_sum_sq"),
              "change_e_over_memory": ratio("delta_diff_sum_sq", "memory_sum_sq")}
    if "gamma_diff_sum_sq" in stats:
        result["change_gamma_rel"] = ratio("gamma_diff_sum_sq", "gamma_ref_sum_sq")
        count = int(stats["gamma_cosine_count"])
        result["change_gamma_cosine"] = (float(stats["gamma_cosine_sum"]) / count
                                         if count else None)
    return result
