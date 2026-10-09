"""Training loss and dev-F1 curves for the public_qa_90k seed-42 arms.

Reads the per-step logs copied next to this script; writes train_loss.png and dev_f1.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ARMS = ["SQX", "SQ", "S0X", "S0"]
COLOR = {"SQX": "#2a78d6", "SQ": "#eb6834", "S0X": "#1baf7a", "S0": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EPOCH = 90000 / 16  # optimizer steps per pass over 90k rows at effective batch 16
WINDOW = 10         # log rows are every 20 steps -> 200-step moving mean


def load(arm):
    rows = [json.loads(line) for line in open(HERE / f"{arm}_s42_train_log.jsonl")]
    train = [r for r in rows if "step" in r]
    return (np.array([r["step"] for r in train]), np.array([r["loss"] for r in train]),
            [r["validation"] for r in rows if "validation" in r])


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=26)
    ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.axvline(EPOCH, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(EPOCH + 80, 0.02, "1 epoch (5,625 steps)", color=INK2, fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0),
              labelcolor=INK, handlelength=1.6)


def end_labels(ax, x, ends, fmt, gap):
    """Direct labels at the line ends, nudged apart so they never overlap."""
    placed = []
    for arm in sorted(ends, key=ends.get):
        y = ends[arm] if not placed else max(ends[arm], placed[-1] + gap)
        placed.append(y)
        ax.text(x, y, fmt(arm, ends[arm]), color=INK, fontsize=8, va="center")


data = {arm: load(arm) for arm in ARMS}

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ends = {}
for arm in ARMS:
    steps, loss, _ = data[arm]
    mean = np.convolve(loss, np.ones(WINDOW) / WINDOW, mode="valid")
    ax.plot(steps, loss, color=COLOR[arm], lw=0.6, alpha=0.15)
    ax.plot(steps[WINDOW - 1:], mean, color=COLOR[arm], lw=2, label=arm)
    ends[arm] = mean[-1]
ax.set_xlim(0, 9900)
ax.set_ylim(0, 4.2)
style(ax, "Training loss (answer CE), public_qa_90k, seed 42",
      "loss (faint: every 20 steps; solid: 200-step mean)")
end_labels(ax, 9080, ends, lambda a, v: f"{a} {v:.2f}", 0.13)
fig.tight_layout()
fig.savefig(HERE / "train_loss.png", facecolor=SURF)

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ends = {}
for arm in ARMS:
    val = data[arm][2][1:]  # step 0 (F1 12.5) would flatten the axis
    xs, ys = [r["step"] for r in val], [r["f1"] * 100 for r in val]
    ax.plot(xs, ys, color=COLOR[arm], lw=2, marker="o", ms=4, mec=SURF, mew=1.5, label=arm)
    ends[arm] = ys[-1]
ax.set_xlim(0, 9900)
ax.set_ylim(47, 55)
style(ax, "Dev F1 during training (mixed dev, first 500; step 0 = 12.5 not shown)", "F1 (%)")
end_labels(ax, 9150, ends, lambda a, v: f"{a} {v:.1f}", 0.32)
fig.tight_layout()
fig.savefig(HERE / "dev_f1.png", facecolor=SURF)
