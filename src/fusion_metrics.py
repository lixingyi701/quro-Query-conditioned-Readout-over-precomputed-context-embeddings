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
