"""Figures for the full-compression infeasibility replication.

Reads only what ``scripts/diagnose_full_compression_infeasibility.py`` wrote --
never the model -- so a figure can always be rebuilt from a run directory, and a
plot can never quietly come from a different forward pass than the numbers beside
it.

Five figures.  Four are fixed by
``docs/SELECOM_FULL_COMPRESSION_INFEASIBILITY_INSTRUCTION.md`` §8.4/§9; the fifth
puts behaviour next to the attention ratios, because §2 orders the argument
behaviour-first and a dominance plot with no failure under it establishes
nothing.

``figure2_conceptual_replication.png``
    SeleCom's Figure 2 layout: source groups across, target tokens down, one
    panel per (document representation x instruction).  Mass and density are on
    two rows with a *shared* scale per row, because the whole question is whether
    the memory's pull survives dividing by its token count -- mass alone would
    confirm SeleCom on 80-vs-20 positions.

``layer_group_heatmap.png``   where in the stack, with a bootstrap band
``generation_step_heatmap.png``  first token or pulled back as output accumulates
``norm_logit_diagnostics.png``   the quantities SeleCom's Appendix A.1 blames
``behaviour_and_dominance.png``  did the model actually fail, in the same rows

    python scripts/plot_full_compression_infeasibility.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import infeasibility as inf

# --------------------------------------------------------------------------------------
# Palette.  Sequential = one hue light->dark for magnitude; three categorical
# slots for the line panels, which is the number that clears the all-pairs check
# (worst CVD dE 9.2, worst normal-vision dE 24.0 on the light surface).  Aqua
# sits below 3:1 against this surface, so every series is also direct-labelled.
# --------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("quro_blue", BLUE_RAMP)

#: Fixed slot order; never cycled.  Only these three are ever drawn together.
SERIES: Dict[str, str] = {"document": "#2a78d6", "instruction": "#eb6834",
                          "query": "#1baf7a"}

#: The scaffolding groups carry no claim and would be four more lines competing
#: for the same axes, so they fold into one column.
FORMAT_GROUPS = ("document_delimiter", "query_delimiter", "answer_prefix")
PANEL_GROUPS = ("prefix", "document", "query", "instruction", "output_history")
PANEL_LABELS = {"prefix": "Prefix", "document": "Document", "query": "Query",
                "instruction": "Instruction", "output_history": "Output",
                "format": "Format"}


def style_axes(ax, *, grid_axis: Optional[str] = "y") -> None:
    """Recessive chrome: hairline grid behind the data, two spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        ax.grid(False, axis="x" if grid_axis == "y" else "y")
        ax.set_axisbelow(True)


def draw_series(ax, x: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray,
                names: Sequence[str], *, legend: bool = False,
                log: bool = False) -> None:
    """One line per claim-bearing group, with a band and a de-collided end label.

    A series with no finite values -- the query group in Level A, where there is
    no question -- is dropped rather than drawn as an empty legend entry that
    implies a measurement was made.
    """
    drawn: List[Tuple[float, str, str]] = []
    for name, colour in SERIES.items():
        if name not in names:
            continue
        j = list(names).index(name)
        finite = np.isfinite(mean[:, j])
        if not finite.any():
            continue
        ax.plot(x, mean[:, j], color=colour, linewidth=2.0,
                label=PANEL_LABELS.get(name, name), zorder=3)
        if np.isfinite(low[:, j]).any() and not np.allclose(low[:, j], high[:, j],
                                                            equal_nan=True):
            ax.fill_between(x, low[:, j], high[:, j], color=colour, alpha=0.16,
                            linewidth=0, zorder=2)
        drawn.append((float(mean[finite, j][-1]), PANEL_LABELS.get(name, name), colour))
    if log:
        ax.set_yscale("log")
    if not drawn:
        return

    # Direct labels for every series (the palette's relief rule), nudged apart
    # when two endpoints land on top of each other.
    low_y, high_y = ax.get_ylim()
    if log:
        span = np.log10(max(high_y, 1e-12)) - np.log10(max(low_y, 1e-12))
        place = lambda v: np.log10(max(v, 1e-12))
        unplace = lambda v: 10 ** v
    else:
        span = high_y - low_y
        place, unplace = (lambda v: v), (lambda v: v)
    gap = 0.055 * span if span else 1.0
    # Room for the end labels inside the axes; placed outside they get clipped by
    # the neighbouring panel, which is how "Instructio" ends up on the page.
    width = float(x[-1] - x[0]) or 1.0
    ax.set_xlim(float(x[0]) - 0.02 * width, float(x[-1]) + 0.30 * width)
    # The padding must not grow the tick range into values the data never had.
    ax.set_xticks([t for t in ax.get_xticks() if x[0] <= t <= x[-1]])
    drawn.sort(key=lambda item: item[0])
    previous = -np.inf
    for value, label, colour in drawn:
        target = max(place(value), previous + gap)
        previous = target
        ax.annotate(label, (x[-1], unplace(target)), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_SECONDARY,
                    va="center")
    if legend:
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
class Run:
    """One run directory, indexed by condition."""

    def __init__(self, run_dir: str):
        self.dir = os.path.abspath(run_dir)
        self.manifest = self._json("manifest.json")
        self.behaviour = self._json("behavioral_metrics.json")
        self.attention = np.load(os.path.join(self.dir, "grouped_attention_stats.npz"),
                                 allow_pickle=False)
        self.norms = np.load(os.path.join(self.dir, "norm_logit_stats.npz"),
                             allow_pickle=False)
        self.groups = [str(g) for g in self.attention["groups"]]
        self.condition = [str(c) for c in self.attention["condition"]]
        self.conditions = sorted(set(self.condition))

    def _json(self, name: str):
        with open(os.path.join(self.dir, name), encoding="utf-8") as f:
            return json.load(f)

    def get(self, key: str, condition: str) -> Optional[np.ndarray]:
        source = self.attention if key in self.attention else self.norms
        if key not in source:
            return None
        index = np.array([i for i, c in enumerate(self.condition) if c == condition])
        if index.size == 0:
            return None
        return source[key][index].astype(np.float32)

    def present(self, *names: str) -> List[str]:
        return [n for n in names if n in self.conditions]


def fold_format(values: np.ndarray, groups: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Collapse the scaffolding groups into one column, keeping the claim-bearing ones."""
    keep = [g for g in PANEL_GROUPS if g in groups]
    folded = [values[..., groups.index(g)] for g in keep]
    other = [groups.index(g) for g in FORMAT_GROUPS if g in groups]
    if other:
        stacked = np.stack([values[..., i] for i in other], -1)
        with quiet():
            summed = np.nansum(stacked, axis=-1)
        # nansum turns an all-NaN row into 0.0, which is finite -- and a finite
        # zero in a target row that does not exist stretches every panel to the
        # padded height.  Absent stays absent.
        folded.append(np.where(np.all(np.isnan(stacked), axis=-1), np.nan, summed))
        keep = keep + ["format"]
    return np.stack(folded, axis=-1), keep


class quiet:
    """All-NaN reductions are expected: absent groups have nothing to average."""

    def __enter__(self):
        self._errstate = np.errstate(invalid="ignore", divide="ignore")
        self._errstate.__enter__()
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        self._warnings.__exit__(*exc)
        self._errstate.__exit__(*exc)
        return False


def bootstrap_band(values: np.ndarray, n_resamples: int = 1000,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and a 95% interval over the *example* axis (axis 0).

    Examples are the independent draws; the layers and heads of one forward pass
    are not, and resampling those would shrink every band until nothing looked
    uncertain.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    with quiet():
        mean = np.nanmean(values, axis=0)
        if n < 3:
            return mean, mean, mean
        index = rng.integers(0, n, size=(n_resamples, n))
        draws = np.nanmean(values[index], axis=1)
        return mean, np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def finite_rows(matrix: np.ndarray) -> int:
    rows = np.isfinite(matrix).any(axis=1)
    return int(np.nonzero(rows)[0][-1]) + 1 if rows.any() else matrix.shape[0]


def clipped_norm(panels: Sequence[np.ndarray], names: Sequence[str]) -> Normalize:
    """A shared scale set by the claim-bearing groups, letting the prefix saturate.

    Position 0 is a well-known attention sink: it takes most of the mass in every
    condition and would compress the document/instruction contrast -- the only
    thing this figure is for -- into the bottom of the ramp.  It stays on the
    axis, clipped, rather than being hidden.
    """
    keep = [i for i, n in enumerate(names) if n != "prefix"]
    values = np.concatenate([p[..., keep].ravel() for p in panels])
    values = values[np.isfinite(values)]
    top = float(np.percentile(values, 99)) if values.size else 1.0
    return Normalize(0.0, top or 1.0)


# --------------------------------------------------------------------------------------
# Figure 1: SeleCom's Figure 2 layout
# --------------------------------------------------------------------------------------
def figure2(run: Run, out_dir: str, max_steps: int = 24) -> Optional[str]:
    # SeleCom's 2x2 is (compressed | raw) x (grounded task | conflict).  The
    # grounded task is reconstruction at Level A and QA at Level B, so pick
    # whichever pair this run actually has rather than falling back only when the
    # conflict pair is missing too.
    grounded = (run.present("memory/reconstruct", "raw/reconstruct")
                or run.present("memory/qa", "raw/qa"))
    conflicting = (run.present("memory/conflict", "raw/conflict")
                   or run.present("memory/qa_conflict", "raw/qa_conflict"))
    columns = grounded + conflicting
    if len(columns) < 2:
        return None

    rows = [("mass_sg", "mass\n(sums to 1 per target token)"),
            ("density_sg", "density\n(mass per source token)")]
    data: Dict[str, Dict[str, np.ndarray]] = {}
    names: List[str] = []
    for key, _ in rows:
        data[key] = {}
        for condition in columns:
            values = run.get(key, condition)
            if values is None:
                continue
            folded, names = fold_format(values, run.groups)
            with quiet():
                data[key][condition] = np.nanmean(folded, axis=0)[:max_steps]
    if not any(data.values()):
        return None

    scale = {key: clipped_norm(list(block.values()), names) for key, block in data.items()
             if block}
    # Each condition keeps its own target length: a 60-token reconstruction and a
    # 10-token nonce padded to a common height would be two thirds empty box.
    height = {c: max((finite_rows(block[c]) for block in data.values() if c in block),
                     default=1) for c in columns}

    fig, axes = plt.subplots(len(rows), len(columns),
                             figsize=(2.55 * len(columns) + 1.2, 3.4 * len(rows)),
                             squeeze=False, facecolor=SURFACE)
    for r, (key, row_label) in enumerate(rows):
        image = None
        for c, condition in enumerate(columns):
            ax = axes[r][c]
            style_axes(ax, grid_axis=None)
            matrix = data[key].get(condition)
            if matrix is None:
                ax.axis("off")
                continue
            image = ax.imshow(matrix[: height[condition]], aspect="auto",
                              cmap=SEQUENTIAL, norm=scale[key], interpolation="nearest")
            ax.set_xticks(range(len(names)))
            if r == len(rows) - 1:
                ax.set_xticklabels([PANEL_LABELS.get(n, n) for n in names],
                                   rotation=45, ha="right", fontsize=8,
                                   color=INK_SECONDARY)
            else:
                ax.set_xticklabels([])
                ax.set_title(condition, fontsize=9.5, color=INK, pad=6)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=8.5, color=INK)
        if image is not None:
            bar = fig.colorbar(image, ax=list(axes[r]), fraction=0.028, pad=0.015,
                               extend="max")
            bar.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle("Attention on each source group while predicting the target tokens "
                 "(rows: target token, top to bottom)\nshared scale per row, clipped at "
                 "the 99th percentile of the non-prefix cells; position 0 is an "
                 "attention sink and saturates", fontsize=10, color=INK, y=0.995)
    path = os.path.join(out_dir, "figure2_conceptual_replication.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 2: layer x source group
# --------------------------------------------------------------------------------------
def layer_figure(run: Run, out_dir: str) -> Optional[str]:
    columns = run.present("memory/reconstruct", "raw/reconstruct",
                          "memory/conflict", "raw/conflict",
                          "memory/qa", "raw/qa")[:4]
    if not columns:
        return None

    prepared = []
    for condition in columns:
        density = run.get("density_lhg", condition)
        if density is None:
            continue
        folded, names = fold_format(density, run.groups)          # (N, L, H, G')
        with quiet():
            per_layer = np.nanmean(folded, axis=2)                # (N, L, G')
        prepared.append((condition, names) + bootstrap_band(per_layer))
    if not prepared:
        return None

    names = prepared[0][1]
    scale = clipped_norm([p[2] for p in prepared], names)
    ceiling = max(float(np.nanmax(p[4][:, [names.index(n) for n in SERIES if n in names]]))
                  for p in prepared)

    fig, axes = plt.subplots(2, len(prepared), figsize=(3.1 * len(prepared) + 1.0, 7.0),
                             squeeze=False, facecolor=SURFACE)
    image = None
    for c, (condition, names, mean, low, high) in enumerate(prepared):
        ax = axes[0][c]
        style_axes(ax, grid_axis=None)
        image = ax.imshow(mean.T, aspect="auto", cmap=SEQUENTIAL, norm=scale,
                          interpolation="nearest")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([PANEL_LABELS.get(n, n) for n in names] if c == 0 else [],
                           fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("layer", fontsize=8, color=INK_MUTED)
        ax.set_title(condition, fontsize=9.5, color=INK)

        ax = axes[1][c]
        style_axes(ax)
        ax.set_ylim(0, ceiling * 1.12)
        draw_series(ax, np.arange(mean.shape[0]), mean, low, high, names,
                    legend=(c == 0))
        ax.set_xlabel("layer", fontsize=8, color=INK_MUTED)
        if c == 0:
            ax.set_ylabel("attention density (mass / source token)", fontsize=8,
                          color=INK_MUTED)
        else:
            ax.set_yticklabels([])
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes[0]), fraction=0.026, pad=0.015, extend="max")
        bar.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle("Per-layer attention density by source group, 95% interval over examples",
                 fontsize=10.5, color=INK, y=0.995)
    path = os.path.join(out_dir, "layer_group_heatmap.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 3: generation step x source group
# --------------------------------------------------------------------------------------
def step_figure(run: Run, out_dir: str, max_steps: int = 24) -> Optional[str]:
    columns = run.present("memory/conflict", "raw/conflict", "none/conflict",
                          "memory/qa_conflict", "raw/qa_conflict")[:4]
    if not columns:
        return None

    prepared = []
    for condition in columns:
        values = run.get("density_sg", condition)
        if values is None:
            continue
        folded, names = fold_format(values[:, :max_steps], run.groups)
        prepared.append((condition, names) + bootstrap_band(folded))
    if not prepared:
        return None

    fig, axes = plt.subplots(1, len(prepared), figsize=(3.3 * len(prepared) + 0.8, 3.6),
                             squeeze=False, sharey=True, facecolor=SURFACE)
    for c, (condition, names, mean, low, high) in enumerate(prepared):
        ax = axes[0][c]
        style_axes(ax)
        rows = finite_rows(mean)
        draw_series(ax, np.arange(rows), mean[:rows], low[:rows], high[:rows], names,
                    legend=(c == 0))
        ax.set_title(condition, fontsize=9.5, color=INK)
        ax.set_xlabel("teacher-forced target position", fontsize=8, color=INK_MUTED)
        if c == 0:
            ax.set_ylabel("attention density", fontsize=8, color=INK_MUTED)

    fig.suptitle("Does the document's pull grow as the output accumulates?",
                 fontsize=10.5, color=INK)
    path = os.path.join(out_dir, "generation_step_heatmap.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 4: the mechanism candidates
# --------------------------------------------------------------------------------------
def norm_figure(run: Run, out_dir: str) -> Optional[str]:
    rows = run.present("memory/conflict", "raw/conflict") or \
        run.present("memory/qa", "raw/qa")
    if not rows:
        return None

    # The residual-stream norm spans two orders of magnitude between a memory slot
    # and a text token, so it gets a log axis; on a linear one the whole panel is
    # the final-layer spike and the layer-0 gap -- the thing SeleCom predicts --
    # is invisible.
    measures = [("qk_mean_lhg", "pre-softmax QK logit", 2, False),
                ("k_norm_lhg", "key norm", 2, False),
                ("v_norm_lhg", "value norm", 2, False),
                ("hidden_norm_lg", "residual-stream norm (log)", None, True)]

    prepared: Dict[Tuple[str, str], Tuple] = {}
    for condition in rows:
        for key, _, head_axis, _ in measures:
            values = run.get(key, condition)
            if values is None:
                continue
            folded, names = fold_format(values, run.groups)
            if head_axis is not None:
                with quiet():
                    folded = np.nanmean(folded, axis=head_axis)
            prepared[(condition, key)] = (names,) + bootstrap_band(folded)

    fig, axes = plt.subplots(len(rows), len(measures),
                             figsize=(3.3 * len(measures), 3.0 * len(rows)),
                             squeeze=False, facecolor=SURFACE)
    for column, (key, title, _, log) in enumerate(measures):
        # One scale per measure across conditions: panels that do not share a
        # scale cannot be compared, and comparing them is the point.
        block = [prepared[(c, key)] for c in rows if (c, key) in prepared]
        if not block:
            continue
        slots = [block[0][0].index(n) for n in SERIES if n in block[0][0]]
        finite = np.concatenate([p[1][:, slots].ravel() for p in block])
        finite = finite[np.isfinite(finite)]
        limits = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        pad = 0.12 * (limits[1] - limits[0] or 1.0)

        for row, condition in enumerate(rows):
            ax = axes[row][column]
            style_axes(ax)
            ax.set_title(title if row == 0 else "", fontsize=9.5, color=INK)
            if column == 0:
                ax.set_ylabel(condition, fontsize=9, color=INK)
            if (condition, key) not in prepared:
                ax.axis("off")
                continue
            names, mean, low, high = prepared[(condition, key)]
            if log:
                ax.set_ylim(max(limits[0] * 0.5, 1e-3), limits[1] * 2.5)
            else:
                ax.set_ylim(limits[0] - pad, limits[1] + pad)
            draw_series(ax, np.arange(mean.shape[0]), mean, low, high, names,
                        legend=(row == 0 and column == 0), log=log)
            ax.set_xlabel("layer", fontsize=8, color=INK_MUTED)

    fig.suptitle("SeleCom's proposed mechanism, measured: scale and direction of the "
                 "memory channel against the instruction's", fontsize=10.5, color=INK)
    path = os.path.join(out_dir, "norm_logit_diagnostics.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 5: behaviour, which the mechanism figures only mean anything next to
# --------------------------------------------------------------------------------------
def behaviour_figure(run: Run, out_dir: str) -> Optional[str]:
    conditions = [c for c in run.conditions if c.endswith("conflict")]
    if not conditions:
        return None
    per = run.behaviour.get("per_condition", {})

    def value(condition: str, key: str) -> float:
        # A missing key is missing, not zero: reading `or 0.0` off an absent
        # metric would draw a bar where no measurement exists.
        raw = per.get(condition, {}).get(key)
        return float(raw) if isinstance(raw, (int, float)) else np.nan

    rates = [("exact", "output is exactly the nonce", "#2a78d6"),
             ("leading", "output starts with the nonce", "#eb6834"),
             ("mentions", "nonce appears anywhere (upper bound)", "#1baf7a")]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 0.62 * len(conditions) + 2.6),
                             facecolor=SURFACE, gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    style_axes(ax, grid_axis="x")
    y = np.arange(len(conditions))
    height = 0.26
    for k, (key, label, colour) in enumerate(rates):
        values = np.array([value(c, key) for c in conditions])
        bars = ax.barh(y + (1 - k) * height, np.nan_to_num(values), height=height * 0.92,
                       color=colour, label=label, zorder=3)
        for bar, raw in zip(bars, values):
            if not np.isfinite(raw):
                continue
            ax.annotate(f"{raw:.0%}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", fontsize=7,
                        color=INK_SECONDARY, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("rate over examples", fontsize=8, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower right")
    ax.set_title("Instruction following under a conflicting instruction",
                 fontsize=9.5, color=INK)

    # Dots, not bars: the ratios span two decades and want a log axis, where a
    # bar's length no longer encodes its value.
    ax = axes[1]
    style_axes(ax, grid_axis="x")
    for key, label, colour, marker in (
            ("dominance.mass_ratio", "mass ratio", "#2a78d6", "o"),
            ("dominance.density_ratio", "density ratio", "#eb6834", "D")):
        values = np.array([value(c, key) for c in conditions])
        ax.scatter(values, y, s=54, color=colour, marker=marker, label=label,
                   zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("parity", (1.0, len(conditions) - 0.35), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(conditions) - 0.4)
    ax.set_xscale("log")
    ax.set_xlabel("document / instruction  (log scale)", fontsize=8, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower right")
    ax.set_title("Attention dominance in the same rows", fontsize=9.5, color=INK)

    path = os.path.join(out_dir, "behaviour_and_dominance.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def separability_figure(run: Run, out_dir: str) -> Optional[str]:
    """Instruction following against document utility, one point per intervention.

    The question the subspace sweep asks is whether the suppressing component can
    be removed *without* taking the document with it.  That is a statement about
    two numbers at once, so it has to be drawn in two dimensions: anything that
    separates them would sit above and to the right of the curve the others trace.
    """
    per = run.behaviour.get("per_condition", {})

    def point(conflict: str, grounded: Optional[str]):
        a = per.get(conflict, {}).get("leading")
        b = per.get(grounded or "", {}).get("rouge_l")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return float(b), float(a)

    family = [("row mean", "memory@rmean"), ("global mean", "memory@gmean"),
              ("−μ", "memory@dec"), ("−μ renorm", "memory@decn"),
              ("−top1", "memory@proj1"), ("−top2", "memory@proj2"),
              ("−top4", "memory@proj4"), ("−top8", "memory@proj8"),
              ("−top16", "memory@proj16")]
    swept = [(label, point(f"{tag}/conflict", f"{tag}/reconstruct")) for label, tag in family]
    swept = [(label, p) for label, p in swept if p]
    anchors = [("PISCO memory", point("memory/conflict", "memory/reconstruct"), "#2a78d6"),
               ("raw text", point("raw/conflict", "raw/reconstruct"), "#1baf7a"),
               ("no memory", point("none/conflict", "none/reconstruct"), "#898781")]
    anchors = [(n, p, c) for n, p, c in anchors if p]
    if not swept or not anchors:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 5.4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)

    # Only the deproject family is an ordered sequence (k = 1, 2, 4, 8, 16), so
    # only it gets a connecting line; joining the others in list order would draw
    # a trajectory through points that have no order.
    ordered = [(label, p) for label, p in swept if label.startswith("−top")]
    if len(ordered) > 1:
        ax.plot([p[0] for _, p in ordered], [p[1] for _, p in ordered],
                color="#eb6834", linewidth=1.4, alpha=0.45, zorder=2)
    ax.scatter([p[0] for _, p in swept], [p[1] for _, p in swept], s=58, color="#eb6834",
               zorder=3, edgecolor=SURFACE, linewidth=1.2,
               label="latent edits; line = removing the top k directions (k = 1…16)")
    for label, (x, y) in swept:
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color=INK_SECONDARY)
    for name, (x, y), colour in anchors:
        ax.scatter([x], [y], s=120, marker="*", color=colour, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(name, (x, y), xytext=(7, -10), textcoords="offset points",
                    fontsize=8.5, color=INK)

    ax.set_xlabel("document utility  (reconstruction ROUGE-L)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("instruction following  (output starts with the nonce)",
                  fontsize=9, color=INK_MUTED)
    ax.set_title("Can the suppressing component be removed without the document?\n"
                 "every latent edit trades one for the other; raw text is off the curve",
                 fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper right")
    path = os.path.join(out_dir, "separability_tradeoff.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/figures")
    ap.add_argument("--max_steps", type=int, default=24)
    args = ap.parse_args()

    run = Run(args.run_dir)
    out_dir = args.out_dir or os.path.join(run.dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    written = [
        figure2(run, out_dir, args.max_steps),
        layer_figure(run, out_dir),
        step_figure(run, out_dir, args.max_steps),
        norm_figure(run, out_dir),
        behaviour_figure(run, out_dir),
        separability_figure(run, out_dir),
    ]
    for path in written:
        print("wrote" if path else "skipped", path or "(condition absent from this run)")

    with open(os.path.join(out_dir, "figures.json"), "w", encoding="utf-8") as f:
        json.dump({"run": run.dir, "level": run.manifest.get("level"),
                   "conditions": run.conditions,
                   "figures": [os.path.basename(p) for p in written if p]},
                  f, indent=2)


if __name__ == "__main__":
    main()
