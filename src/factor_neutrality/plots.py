"""Publication figures. One consistent style: fixed categorical order, a
blue<->red diverging scale with a gray midpoint for signed quantities, a single-hue
blue ramp for magnitudes, thin marks, recessive grid, never a second y-axis.
Each figure is saved as PNG (README) and PDF (paper).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32020"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
RECESSIONS: pd.Series | None = None  # set by the pipeline for shading

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


def _shade_recessions(ax: plt.Axes, index: pd.Index) -> None:
    if RECESSIONS is None:
        return
    rec = RECESSIONS.reindex(index).fillna(0).astype(bool)
    starts = rec & ~rec.shift(1, fill_value=False)
    ends = rec & ~rec.shift(-1, fill_value=False)
    for s, e in zip(index[starts.to_numpy()], index[ends.to_numpy()]):
        ax.axvspan(s, e, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)


def cumulative(returns: pd.DataFrame, title: str, log: bool = True, ylabel: str = "Growth of $1",
               recessions: bool = True) -> plt.Figure:
    """Growth of $1 for each column (log axis so equal % moves look equal)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    wealth = (1 + returns.fillna(0)).cumprod()
    for i, c in enumerate(wealth):
        ax.plot(wealth.index, wealth[c], color=SERIES[i % 8], label=c)
        ax.annotate(c, (wealth.index[-1], wealth[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
    if log:
        ax.set_yscale("log")
    if recessions:
        _shade_recessions(ax, wealth.index)
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(5, len(wealth.columns)))
    return fig


def drawdowns(dd: pd.DataFrame, title: str) -> plt.Figure:
    fig, axes = plt.subplots(len(dd.columns), 1, figsize=(7.2, 1.5 * len(dd.columns)),
                             sharex=True, sharey=True)
    for i, (ax, c) in enumerate(zip(np.atleast_1d(axes), dd)):
        ax.fill_between(dd.index, dd[c] * 100, 0, color=SERIES[i % 8], lw=0, alpha=0.85)
        ax.set_ylabel("%")
        ax.text(0.01, 0.08, f"{c}  (max {dd[c].min() * 100:.0f}%)", transform=ax.transAxes,
                fontsize=8, color=INK)
    np.atleast_1d(axes)[0].set_title(title)
    return fig


def heatmaps(grids: dict[str, pd.DataFrame], title: str, diverging: bool = True,
             fmt: str = "{:.2f}", vmax: float | None = None) -> plt.Figure:
    """Side-by-side 5x5 grids (size rows x BE/ME columns) with values printed."""
    fig, axes = plt.subplots(1, len(grids), figsize=(3.0 * len(grids), 2.9))
    axes = np.atleast_1d(axes)
    vm = vmax or max(np.nanmax(np.abs(g.to_numpy())) for g in grids.values())
    for ax, (name, g) in zip(axes, grids.items()):
        kw = dict(cmap=DIVERGING, vmin=-vm, vmax=vm) if diverging else dict(cmap=SEQUENTIAL, vmin=0, vmax=vm)
        im = ax.imshow(g.to_numpy(), **kw)
        for (i, j), v in np.ndenumerate(g.to_numpy()):
            shade = abs(v) / vm if diverging else v / vm
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if shade > 0.6 else INK)
        ax.set_xticks(range(5), ["Low", "2", "3", "4", "High"])
        ax.set_yticks(range(5), ["Small", "2", "3", "4", "Big"] if ax is axes[0] else [])
        ax.set_xlabel("BE/ME quintile")
        ax.set_title(name, fontsize=9)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    return fig


def scatter_45(x: pd.Series, y: pd.Series, labels: list[str] | None, title: str,
               xlabel: str, ylabel: str, groups: pd.Series | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    if groups is None:
        ax.scatter(x, y, s=22, color=SERIES[0], edgecolor=SURFACE, lw=1)
    else:
        for i, g in enumerate(groups.unique()):
            m = (groups == g).to_numpy()
            ax.scatter(x[m], y[m], s=22, color=SERIES[i], edgecolor=SURFACE, lw=1, label=g)
        ax.legend()
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.8, ls="--")
    if labels:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def coef_bars(est: pd.DataFrame, se: pd.DataFrame, title: str, ylabel: str) -> plt.Figure:
    """Grouped bars: rows = coefficients, columns = models; whiskers = +/-1.96 SE."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    k = len(est.columns)
    width = 0.8 / k
    x = np.arange(len(est.index))
    for i, m in enumerate(est.columns):
        ax.bar(x + i * width - 0.4 + width / 2, est[m], width * 0.92, color=SERIES[i], label=m,
               yerr=1.96 * se[m], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, est.index)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=k, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return fig


def rolling_lines(df: pd.DataFrame, title: str, ylabel: str, band: pd.DataFrame | None = None,
                  zero: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _shade_recessions(ax, df.index)
    for i, c in enumerate(df):
        ax.plot(df.index, df[c], color=SERIES[i % 8], label=c)
        if band is not None and c in band:
            ax.fill_between(df.index, df[c] - band[c], df[c] + band[c], color=SERIES[i % 8],
                            alpha=0.18, lw=0)
    if zero:
        ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(df.columns) > 1:
        ax.legend(ncol=min(6, len(df.columns)), loc="upper left")
    return fig


def stacked_shares(shares: pd.DataFrame, title: str, xlabel: str) -> plt.Figure:
    """Horizontal 100% bars (rows = strategies/models, columns = components)."""
    fig, ax = plt.subplots(figsize=(7.2, 0.5 + 0.45 * len(shares)))
    left_pos = np.zeros(len(shares))
    left_neg = np.zeros(len(shares))
    for i, c in enumerate(shares.columns):
        v = shares[c].to_numpy()
        color = "#b5b4ad" if c == "Residual" else SERIES[i % 8]
        left = np.where(v >= 0, left_pos, left_neg)
        ax.barh(shares.index, v, left=left, color=color, label=c, edgecolor=SURFACE, lw=1)
        left_pos += np.where(v >= 0, v, 0)
        left_neg += np.where(v < 0, v, 0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(ncol=min(7, len(shares.columns)), loc="upper center", bbox_to_anchor=(0.5, -0.45 if len(shares) < 4 else -0.22))
    return fig


def waterfall(parts: pd.Series, title: str, ylabel: str) -> plt.Figure:
    """Mean-return attribution: factor contributions + alpha stack to the total."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    cum = 0.0
    for i, (k, v) in enumerate(parts.items()):
        color = SERIES[0] if v >= 0 else SERIES[7]
        ax.bar(i, v, bottom=cum, color=color, width=0.65)
        ax.text(i, cum + v + (0.15 if v >= 0 else -0.15), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
        cum += v
    ax.bar(len(parts), cum, color=INK2, width=0.65)
    ax.text(len(parts), cum + 0.15, f"{cum:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(parts) + 1), [*parts.index, "Total"])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig


def grid_heatmap(table: pd.DataFrame, title: str, fmt: str = "{:.2f}", ax: plt.Axes | None = None,
                 vmax: float | None = None) -> plt.Axes:
    ax = ax or plt.subplots(figsize=(4, 3))[1]
    vm = vmax or np.nanmax(np.abs(table.to_numpy()))
    ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-vm, vmax=vm, aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) / vm > 0.6 else INK)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    return ax
