# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
"""Publication-quality figures for BB-code correlated-noise simulations.

Figures (workbook Chapter 9, minimum main-text set)
----------------------------------------------------
S1  Crossing-kernel J₀ sweep (theorem validation, Ch. 3–5)
S2  Power-law J₀ sweep (main distance-decay figure)
S3  Power-law α sweep at J₀ = 0.04 (summability story)
S4  Exponential ξ sweep at J₀ = 0.04
S5  PER sweep at J₀ = 0.04, α = 3 (transition regime)
S6  Phase diagram at p = 3 × 10⁻³ (two variants)
G3  Weighted exposure vs LER scatter + regression (meta-analysis)
T1  BB144 power-law J₀ sweep (scaling check)
T2  BB144 PER sweep at J₀ = 0.04, α = 3 (scaling check)
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterSciNotation  # noqa: E402
from scipy.stats import beta as _beta_dist  # noqa: E402

_HAS_LATEX = bool(shutil.which("latex") and shutil.which("dvipng"))

# ── Okabe–Ito palette ──
_BLUE      = "#0072B2"
_VERMILION = "#D55E00"
_GREEN     = "#009E73"
_ORANGE    = "#E69F00"
_SKY       = "#56B4E9"
_PURPLE    = "#CC79A7"
_BLACK     = "#000000"

EMB_STYLE: dict[str, tuple[str, str, str]] = {
    "monomial_column": ("Single-layer (monomial)", _VERMILION, "s"),
    "ibm_biplanar":    ("Thickness-2 (biplanar)",  _BLUE,      "o"),
    "logical_aware":   ("Logical-aware",            _GREEN,     "^"),
}

COL1 = 3.375
COL2 = 7.0

_LINE_W     = 0.6
_MK_SIZE    = 3.5
_CAP_SIZE   = 1.8
_CAP_THICK  = 0.6   # match vertical bar thickness
_EBAR_W     = 0.6

STYLE: dict[str, Any] = {
    "text.usetex":          _HAS_LATEX,
    "font.family":          "serif",
    "font.serif":           ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset":     "cm",
    "axes.formatter.use_mathtext": True,
    "axes.labelsize":       9,
    "font.size":            8,
    "legend.fontsize":      6.5,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.frameon":       True,
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "0.82",
    "legend.handlelength":  1.6,
    "legend.borderpad":     0.45,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.5,
    "legend.labelspacing":  0.45,
    "figure.dpi":           200,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.linewidth":       0.5,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.size":     1.8,
    "ytick.minor.size":     1.8,
    "xtick.major.width":    0.45,
    "ytick.major.width":    0.45,
    "xtick.minor.width":    0.3,
    "ytick.minor.width":    0.3,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "grid.linewidth":       0.3,
    "grid.alpha":           0.22,
    "lines.linewidth":      _LINE_W,
    "lines.markersize":     _MK_SIZE,
    "errorbar.capsize":     _CAP_SIZE,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _clopper_pearson(k: int, n: int, alpha: float = 0.3173) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(_beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _get_yerr(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = df["primary_ler_total"].values.astype(float)
    lo = df["primary_ler_total_lo"].values.astype(float)
    hi = df["primary_ler_total_hi"].values.astype(float)
    return np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)


def _save(fig: plt.Figure, outdir: Path, stem: str,
          force_log_axes: bool = True) -> None:
    """Save figure, optionally forcing decade labels on all log axes."""
    outdir.mkdir(parents=True, exist_ok=True)
    if force_log_axes:
        for ax in fig.get_axes():
            axes_to_fix = ""
            if ax.get_xscale() == "log":
                axes_to_fix += "x"
            if ax.get_yscale() == "log":
                axes_to_fix += "y"
            if axes_to_fix:
                # Map single chars to the axis parameter _force_log_labels expects
                if axes_to_fix == "xy":
                    _force_log_labels(ax, "both")
                else:
                    _force_log_labels(ax, axes_to_fix)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)


def _log_ticks(ax, axis: str = "both") -> None:
    subs = np.arange(2, 10)
    if axis in ("y", "both"):
        ax.yaxis.set_minor_locator(LogLocator(subs=subs, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if axis in ("x", "both"):
        ax.xaxis.set_minor_locator(LogLocator(subs=subs, numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())


def _force_log_labels(ax, axis: str = "both") -> None:
    """Ensure at least 2 labeled ticks on every log axis.

    For narrow ranges (<2 decades), extends the axis limits to include
    the nearest decade boundaries.  Then places labeled ticks at every
    10^n within the view.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter

    def _fmt_pow10(x, pos):
        if x <= 0:
            return ""
        exp = np.log10(x)
        if abs(exp - round(exp)) < 0.01:
            return rf"$10^{{{int(round(exp))}}}$"
        # Sub-decade labels
        exp_floor = int(np.floor(exp))
        ratio = x / 10.0 ** exp_floor
        for mult in [2, 3, 5]:
            if abs(ratio - mult) < 0.15:
                return rf"${mult}\!\times\!10^{{{exp_floor}}}$"
        return ""

    def _fix_axis(axis_obj, set_lim_fn):
        lo, hi = axis_obj.get_view_interval()
        if lo <= 0 or hi <= 0:
            return
        exp_lo = int(np.floor(np.log10(lo)))
        exp_hi = int(np.ceil(np.log10(hi)))
        ticks = [10.0 ** e for e in range(exp_lo, exp_hi + 1)]
        # Filter to ticks within (or very near) the view
        margin = 0.15  # allow ticks slightly outside data range
        ticks_in_view = [t for t in ticks
                         if t >= lo * (1 - margin) and t <= hi * (1 + margin)]
        if len(ticks_in_view) < 2:
            # Narrow range — add sub-decade labels at 2× and 5× each decade
            for e in range(exp_lo - 1, exp_hi + 2):
                for mult in [1, 2, 5]:
                    ticks.append(mult * 10.0 ** e)
            ticks = sorted(set(ticks))
            ticks_in_view = [t for t in ticks
                             if t >= lo * 0.9 and t <= hi * 1.1]
        axis_obj.set_major_locator(FixedLocator(ticks_in_view))
        axis_obj.set_major_formatter(FuncFormatter(_fmt_pow10))

    ax.figure.canvas.draw()
    if axis in ("y", "both"):
        _fix_axis(ax.yaxis, ax.set_ylim)
    if axis in ("x", "both"):
        _fix_axis(ax.xaxis, ax.set_xlim)


def _grid(ax) -> None:
    ax.grid(True, which="major", linewidth=0.3, alpha=0.25, color="0.60")
    ax.grid(True, which="minor", linewidth=0.15, alpha=0.12, color="0.70")


def _emb_plot(ax, df: pd.DataFrame, x_col: str, *, detected_only: bool = True,
              show_baseline: bool = False):
    """Plot one curve per embedding with error bars.

    Points with zero failures are dropped when detected_only=True
    (avoids log(0) on log-y axes).  Baseline J₀=0 points are shown
    as horizontal bands when show_baseline=True.
    """
    for emb, edf in df.groupby("embedding"):
        if detected_only:
            edf = edf[edf["primary_failures"] > 0]
        if edf.empty:
            continue
        label, color, marker = EMB_STYLE.get(emb, (emb, _BLACK, "D"))

        if show_baseline:
            baseline = edf[edf[x_col] == 0.0]
            edf = edf[edf[x_col] > 0.0]
            if not baseline.empty:
                y0 = baseline["primary_ler_total"].iloc[0]
                if y0 > 0:
                    ax.axhline(y=y0, color=color, ls=":", lw=0.5, alpha=0.35)

        edf = edf.sort_values(x_col)
        if edf.empty:
            continue
        yerr = _get_yerr(edf)
        eb = ax.errorbar(
            edf[x_col], edf["primary_ler_total"], yerr=yerr,
            marker=marker, label=label, color=color,
            markeredgewidth=0,
            capsize=_CAP_SIZE, elinewidth=_EBAR_W,
        )
        _fix_caps(eb)


def _fix_caps(errorbar_result) -> None:
    """Force cap thickness after errorbar() call.

    Caps are rendered as marker ticks, so their visible thickness
    is controlled by markeredgewidth, not linewidth.
    """
    _, caps, bars = errorbar_result
    for cap in caps:
        cap.set_markeredgewidth(_CAP_THICK)
    for bar in bars:
        bar.set_linewidth(_EBAR_W)


def _parse_kernel_params(df: pd.DataFrame) -> pd.DataFrame:
    if "alpha" in df.columns and "xi" in df.columns:
        return df
    df = df.copy()
    def _extract(kp, key):
        if pd.isna(kp) or kp in ("", "{}"):
            return np.nan
        try:
            return ast.literal_eval(kp).get(key, np.nan)
        except (ValueError, SyntaxError):
            return np.nan
    df["alpha"] = df["kernel_params"].apply(lambda x: _extract(x, "alpha"))
    df["xi"] = df["kernel_params"].apply(lambda x: _extract(x, "xi"))
    return df


def _sem(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    sub = df.copy()
    for col, val in kwargs.items():
        sub = sub[sub[col] == val]
    if sub.empty:
        return sub
    dedup_cols = ["code", "embedding", "kernel", "alpha", "xi", "J0", "p_cnot"]
    cols = [c for c in dedup_cols if c in sub.columns]
    sub = (
        sub.sort_values("primary_shots", ascending=False)
        .drop_duplicates(subset=cols, keep="first")
        .reset_index(drop=True)
    )
    return sub


# ═══════════════════════════════════════════════════════════════════════
#  S1 — Crossing-kernel J₀ sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_s1_crossing_j0(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="crossing", p_cnot=1e-3)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    # Biplanar crossing has many zero-failure points — show only detections
    _emb_plot(ax, sub, "J0", detected_only=True, show_baseline=True)
    pos = sub[(sub["J0"] > 0) & (sub["primary_failures"] > 0)]
    if not pos.empty:
        ax.set_xlim(pos["J0"].min() * 0.5, pos["J0"].max() * 1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, outdir, "bb72_crossing_j0_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  S2 — Power-law J₀ sweep (single panel, full range)
# ═══════════════════════════════════════════════════════════════════════

def plot_s2_powerlaw_j0(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw", p_cnot=1e-3)
    sub = sub[sub["alpha"] == 3.0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "J0", show_baseline=True)
    pos = sub[sub["J0"] > 0]
    if not pos.empty:
        ax.set_xlim(pos["J0"].min() * 0.6, pos["J0"].max() * 1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, outdir, "bb72_powerlaw_j0_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  S3 — Power-law α sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_s3_alpha_sweep(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw",
               J0=0.04, p_cnot=1e-3)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "alpha", detected_only=False)
    ax.axvline(x=2.0, ls=":", lw=0.8, color=_ORANGE, zorder=0,
               label=r"$\alpha_c = 2$ (summability)")
    ax.set_yscale("log")
    _log_ticks(ax, "y")
    _grid(ax)
    ax.set_xlabel(r"Decay exponent $\alpha$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="center right", fontsize=6)
    fig.tight_layout()
    _save(fig, outdir, "bb72_alpha_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  S4 — Exponential ξ sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_s4_xi_sweep(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="exponential",
               J0=0.04, p_cnot=1e-3)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "xi")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Correlation length $\xi$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, outdir, "bb72_xi_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  S5 — PER sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_s5_per_sweep(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw", J0=0.04)
    sub = sub[sub["alpha"] == 3.0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "p_cnot")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, outdir, "bb72_per_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  S6 — Phase diagram (two variants)
# ═══════════════════════════════════════════════════════════════════════

def _phase_data(df):
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw", p_cnot=3e-3)
    alphas = sorted(sub["alpha"].dropna().unique())
    j0s = sorted(sub["J0"].unique())
    if len(alphas) < 2 or len(j0s) < 2:
        return None, None, None, None
    emb_order = [("monomial_column", r"Single-layer"),
                 ("ibm_biplanar", r"Thickness-2")]
    present = [e for e, _ in emb_order if e in sub["embedding"].values]
    if not present:
        return None, None, None, None
    return sub, alphas, j0s, [(e, dict(emb_order)[e]) for e in present]


def plot_s6_phase_diagram(df: pd.DataFrame, outdir: Path) -> None:
    """Bubble plot — circle area ∝ p_L, colour from inferno, values inside large circles."""
    _apply_style()
    sub, alphas, j0s, embs = _phase_data(df)
    if sub is None:
        return

    pos_vals = sub.loc[sub["primary_ler_total"] > 0, "primary_ler_total"]
    vmin, vmax = pos_vals.min(), min(pos_vals.max(), 1.0)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, len(embs), figsize=(COL2 * 0.85, COL1 * 0.92),
                              squeeze=False,
                              gridspec_kw={"right": 0.87})

    for idx, (emb, emb_label) in enumerate(embs):
        ax = axes[0, idx]
        edf = sub[sub["embedding"] == emb]

        # Threshold: place text inside circle if circle is large enough
        inside_threshold_col = 2 if idx == 0 else 3  # 0-indexed column

        for _, row in edf.iterrows():
            val = row["primary_ler_total"]
            if val <= 0 or np.isnan(val):
                continue
            ji = j0s.index(row["J0"])
            ai = alphas.index(row["alpha"])
            area = 60 + 420 * (val / vmax)
            ax.scatter(ji, ai, s=area, c=[val], cmap="inferno", norm=norm,
                       edgecolors="0.4", linewidths=0.2, zorder=3)
            txt = f"{val:.2f}" if val >= 0.1 else (f"{val:.3f}" if val >= 0.01 else f"{val:.4f}")
            if ji >= inside_threshold_col:
                # Inside the circle
                ax.text(ji, ai, txt, ha="center", va="center", fontsize=3.5,
                        color="white" if val < 0.4 else "black", zorder=4)
            else:
                # Below the circle
                ax.text(ji, ai - 0.3, txt, ha="center", va="top", fontsize=3.3,
                        color="0.25", zorder=4)

        # Summability line — yellow, no label
        if 2.0 in alphas:
            y = alphas.index(2.0)
            ax.axhline(y=y, color=_ORANGE, ls="--", lw=0.7, alpha=0.5, zorder=1)

        ax.set_xticks(range(len(j0s)))
        ax.set_xticklabels([f"{j:.3f}" if j < 0.01 else f"{j:.2f}" for j in j0s],
                            rotation=40, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"${a:.1f}$" for a in alphas], fontsize=6)
        ax.set_xlabel(r"Coupling amplitude $J_0$")
        if idx == 0:
            ax.set_ylabel(r"Decay exponent $\alpha$")
        ax.set_title(emb_label, fontsize=7.5)
        ax.set_xlim(-0.6, len(j0s) - 0.4)
        ax.set_ylim(-0.6, len(alphas) - 0.4)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.82,
                         pad=0.02, aspect=22)
    cbar.ax.set_title(r"$p_\mathrm{L}$", fontsize=7, pad=4)
    cbar.ax.tick_params(labelsize=5.5)
    _save(fig, outdir, "bb72_phase_diagram_bubble")


def plot_s6_phase_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Clean heatmap — no summability line, external colorbar."""
    _apply_style()
    sub, alphas, j0s, embs = _phase_data(df)
    if sub is None:
        return

    pos_vals = sub.loc[sub["primary_ler_total"] > 0, "primary_ler_total"]
    vmin, vmax = pos_vals.min(), min(pos_vals.max(), 1.0)

    fig, axes = plt.subplots(1, len(embs), figsize=(COL2 * 0.85, COL1 * 0.82),
                              squeeze=False,
                              gridspec_kw={"right": 0.87})

    for idx, (emb, emb_label) in enumerate(embs):
        ax = axes[0, idx]
        edf = sub[sub["embedding"] == emb]

        grid = np.full((len(alphas), len(j0s)), np.nan)
        for _, row in edf.iterrows():
            ai = alphas.index(row["alpha"])
            ji = j0s.index(row["J0"])
            grid[ai, ji] = row["primary_ler_total"]

        norm = Normalize(vmin=0, vmax=vmax)
        im = ax.imshow(
            grid, origin="lower", aspect="auto",
            norm=norm, cmap="inferno",
            extent=[-0.5, len(j0s) - 0.5, -0.5, len(alphas) - 0.5],
            interpolation="nearest",
        )

        for ai_idx in range(len(alphas)):
            for ji_idx in range(len(j0s)):
                val = grid[ai_idx, ji_idx]
                if np.isnan(val) or val <= 0:
                    continue
                normed = val / vmax
                txt_color = "black" if normed > 0.5 else "white"
                txt = f"{val:.2f}" if val >= 0.1 else f"{val:.3f}"
                ax.text(ji_idx, ai_idx, txt, ha="center", va="center",
                        fontsize=4.2, color=txt_color)

        ax.set_xticks(range(len(j0s)))
        ax.set_xticklabels([f"{j:.3f}" if j < 0.01 else f"{j:.2f}" for j in j0s],
                            rotation=40, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"${a:.1f}$" for a in alphas], fontsize=6)
        if idx == 0:
            ax.set_ylabel(r"Decay exponent $\alpha$")
        ax.set_title(emb_label, fontsize=7.5)

    # Shared centered x label (below rotated tick labels)
    fig.text(0.42, -0.06, r"Coupling amplitude $J_0$", ha="center", fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82,
                         pad=0.02, aspect=22)
    cbar.ax.set_title(r"$p_\mathrm{L}$", fontsize=7, pad=4)
    cbar.ax.tick_params(labelsize=5.5)
    _save(fig, outdir, "bb72_phase_diagram_heatmap", force_log_axes=False)


# ═══════════════════════════════════════════════════════════════════════
#  G3 — Exposure vs LER (scatter + per-embedding regression)
# ═══════════════════════════════════════════════════════════════════════

def plot_g3_exposure_vs_ler(df: pd.DataFrame, outdir: Path) -> None:
    from scipy.stats import spearmanr as _spearmanr

    _apply_style()
    exp_col = "reference_weighted_exposure"
    sub = df.dropna(subset=[exp_col, "primary_ler_total"]).copy()
    sub = sub[(sub[exp_col] > 0) & (sub["primary_failures"] >= 10)]
    if len(sub) < 5:
        return

    _kernel_marker = {"crossing": "s", "powerlaw": "o", "exponential": "D"}

    fig, ax = plt.subplots(figsize=(COL1, COL1))
    for (emb, kernel), gdf in sub.groupby(["embedding", "kernel"]):
        yerr = _get_yerr(gdf)
        _, color, _ = EMB_STYLE.get(emb, (emb, _BLACK, "o"))
        m = _kernel_marker.get(kernel, "^")
        short_emb = "Bi" if "biplanar" in emb else ("LA" if "logical" in emb else "Mono")
        eb = ax.errorbar(
            gdf[exp_col], gdf["primary_ler_total"], yerr=yerr,
            fmt=m, color=color, markersize=3.0, alpha=0.5,
            markeredgewidth=0, linestyle="none",
            capsize=_CAP_SIZE, elinewidth=_EBAR_W,
            label=f"{short_emb}, {kernel.capitalize()}",
            zorder=3,
        )
        _fix_caps(eb)

    # Per-group regressions: crossing (mono only), distance-decay mono, distance-decay bi
    x_fit = np.logspace(np.log10(sub[exp_col].min()) - 0.15,
                         np.log10(sub[exp_col].max()) + 0.15, 100)

    reg_groups = [
        ("monomial_column", "crossing", _VERMILION, "mono cross."),
        ("monomial_column", None, _VERMILION, "mono dist.-decay"),  # powerlaw+exponential
        ("ibm_biplanar", None, _BLUE, "bi dist.-decay"),
    ]
    for emb_name, kernel_filter, color, label_prefix in reg_groups:
        mask = sub["embedding"] == emb_name
        if kernel_filter:
            mask = mask & (sub["kernel"] == kernel_filter)
        else:
            mask = mask & (sub["kernel"] != "crossing")
        if mask.sum() < 3:
            continue
        lx = np.log10(sub.loc[mask, exp_col].values)
        ly = np.log10(sub.loc[mask, "primary_ler_total"].values)
        coeffs = np.polyfit(lx, ly, 1)
        y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
        ls = "-." if kernel_filter == "crossing" else "--"
        ax.plot(x_fit, y_fit, ls=ls, lw=0.5, color=color, alpha=0.4, zorder=2,
                label=rf"{label_prefix}: $\beta={coeffs[0]:.2f}$")

    rho, _ = _spearmanr(sub[exp_col], sub["primary_ler_total"])
    ax.text(
        0.03, 0.97, rf"Spearman $\rho = {rho:.2f}$",
        transform=ax.transAxes, fontsize=6, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.88),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Weighted exposure $\mathcal{E}_\varphi^X$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(fontsize=4.5, ncol=2, loc="lower right",
              borderpad=0.5, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    _save(fig, outdir, "bb72_exposure_vs_ler")

    # ── By-kernel subpanels ──
    kernels = sorted(sub["kernel"].unique())
    if len(kernels) < 2:
        return
    fig2, axes2 = plt.subplots(1, len(kernels), figsize=(COL2, COL1 * 0.88),
                                squeeze=False, sharey=True)
    for ki, kernel in enumerate(kernels):
        ax2 = axes2[0, ki]
        ksub = sub[sub["kernel"] == kernel]
        for emb, gdf in ksub.groupby("embedding"):
            _, color, _ = EMB_STYLE.get(emb, (emb, _BLACK, "o"))
            m = _kernel_marker.get(kernel, "o")
            short = "Thickness-2" if "biplanar" in emb else "Single-layer"
            yerr = _get_yerr(gdf)
            eb2 = ax2.errorbar(
                gdf[exp_col], gdf["primary_ler_total"], yerr=yerr,
                fmt=m, color=color, markersize=3.0, alpha=0.5,
                markeredgewidth=0, linestyle="none",
                capsize=_CAP_SIZE, elinewidth=_EBAR_W,
                label=short, zorder=3,
            )
            _fix_caps(eb2)
        if len(ksub) >= 3:
            lx = np.log10(ksub[exp_col].values)
            ly = np.log10(ksub["primary_ler_total"].values)
            coeffs = np.polyfit(lx, ly, 1)
            xr = np.logspace(lx.min() - 0.1, lx.max() + 0.1, 50)
            ax2.plot(xr, 10 ** np.polyval(coeffs, np.log10(xr)),
                     ls="--", lw=0.5, color="0.4", alpha=0.4, zorder=2)
            rk, _ = _spearmanr(ksub[exp_col], ksub["primary_ler_total"])
            ax2.text(0.04, 0.96,
                     rf"$\rho={rk:.2f}$, $\beta={coeffs[0]:.2f}$",
                     transform=ax2.transAxes, fontsize=5.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.85))

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        _log_ticks(ax2)
        _grid(ax2)
        ax2.set_xlabel(r"Weighted exposure $\mathcal{E}_\varphi^X$")
        if ki == 0:
            ax2.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
        ax2.set_title(kernel.capitalize(), fontsize=7)
        ax2.legend(fontsize=5, loc="lower right")

    fig2.tight_layout(w_pad=0.8)
    _save(fig2, outdir, "bb72_exposure_vs_ler_by_kernel")


# ═══════════════════════════════════════════════════════════════════════
#  T1 — BB144 J₀ sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_t1_bb144_j0(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB144", experiment_kind="z_memory", kernel="powerlaw", p_cnot=1e-3)
    sub = sub[sub["alpha"] == 3.0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "J0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, outdir, "bb144_j0_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  T2 — BB144 PER sweep
# ═══════════════════════════════════════════════════════════════════════

def plot_t2_bb144_per(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB144", experiment_kind="z_memory", kernel="powerlaw", J0=0.04)
    sub = sub[sub["alpha"] == 3.0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "p_cnot")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, outdir, "bb144_per_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  Extra: BB72+BB144 biplanar scaling comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_scaling_comparison(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    bb72 = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw",
                p_cnot=1e-3, embedding="ibm_biplanar")
    bb72 = bb72[(bb72["alpha"] == 3.0) & (bb72["J0"] > 0) & (bb72["primary_failures"] > 0)]
    bb144 = _sem(df, code="BB144", experiment_kind="z_memory", kernel="powerlaw",
                 p_cnot=1e-3, embedding="ibm_biplanar")
    bb144 = bb144[(bb144["alpha"] == 3.0) & (bb144["J0"] > 0) & (bb144["primary_failures"] > 0)]
    if bb72.empty and bb144.empty:
        return

    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    for code_df, code_name, color, marker in [
        (bb72, r"$[\![72,12,6]\!]$", _BLUE, "o"),
        (bb144, r"$[\![144,12,12]\!]$", _SKY, "D"),
    ]:
        code_df = code_df.sort_values("J0")
        if code_df.empty:
            continue
        yerr = _get_yerr(code_df)
        eb = ax.errorbar(code_df["J0"], code_df["primary_ler_total"], yerr=yerr,
                         marker=marker, color=color, markeredgewidth=0,
                         capsize=_CAP_SIZE, elinewidth=_EBAR_W,
                         label=f"Biplanar {code_name}")
        _fix_caps(eb)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, outdir, "bb72_bb144_biplanar_scaling")


# ═══════════════════════════════════════════════════════════════════════
#  Extra: LER ratio mono/bi vs J₀
# ═══════════════════════════════════════════════════════════════════════

def plot_ler_ratio(df: pd.DataFrame, outdir: Path) -> None:
    """LER ratio mono/bi with Gaussian error propagation on the ratio."""
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw", p_cnot=1e-3)
    sub = sub[sub["alpha"] == 3.0]
    mono_df = sub[sub["embedding"] == "monomial_column"].set_index("J0")
    bi_df = sub[sub["embedding"] == "ibm_biplanar"].set_index("J0")
    common = sorted(set(mono_df.index) & set(bi_df.index))
    common = [j for j in common if j > 0
              and mono_df.loc[j, "primary_ler_total"] > 0
              and bi_df.loc[j, "primary_ler_total"] > 0]
    if len(common) < 2:
        return

    ratios, ratio_lo, ratio_hi = [], [], []
    for j in common:
        pm = mono_df.loc[j, "primary_ler_total"]
        pb = bi_df.loc[j, "primary_ler_total"]
        r = pm / pb
        ratios.append(r)
        # Error propagation: δ(a/b) = (a/b) √((δa/a)² + (δb/b)²)
        pm_lo = mono_df.loc[j, "primary_ler_total_lo"]
        pm_hi = mono_df.loc[j, "primary_ler_total_hi"]
        pb_lo = bi_df.loc[j, "primary_ler_total_lo"]
        pb_hi = bi_df.loc[j, "primary_ler_total_hi"]
        # Use half-width of CI as σ estimate
        sig_m = (pm_hi - pm_lo) / 2
        sig_b = (pb_hi - pb_lo) / 2
        sig_r = r * np.sqrt((sig_m / pm) ** 2 + (sig_b / pb) ** 2)
        ratio_lo.append(sig_r)
        ratio_hi.append(sig_r)

    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    eb = ax.errorbar(common, ratios, yerr=[ratio_lo, ratio_hi],
                     marker="o", color=_PURPLE, markeredgewidth=0,
                     markersize=4,
                     capsize=_CAP_SIZE, elinewidth=_EBAR_W)
    _fix_caps(eb)
    ax.axhline(y=1, ls=":", lw=0.5, color="0.5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"$p_\mathrm{L}^\mathrm{mono} / p_\mathrm{L}^\mathrm{bi}$")
    fig.tight_layout()
    _save(fig, outdir, "bb72_ler_ratio_mono_bi")


# ═══════════════════════════════════════════════════════════════════════
#  LA J₀ window (novelty figure)
# ═══════════════════════════════════════════════════════════════════════

def plot_la_j0_window(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw", p_cnot=1e-3)
    sub = sub[(sub["alpha"] == 3.0) & (sub["J0"].isin([0.02, 0.03, 0.04, 0.06]))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "J0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax)
    _grid(ax)
    ax.set_xlabel(r"Coupling amplitude $J_0$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, outdir, "bb72_la_j0_window")


# ═══════════════════════════════════════════════════════════════════════
#  LA α mini-slice (supplementary)
# ═══════════════════════════════════════════════════════════════════════

def plot_la_alpha_minislice(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    sub = _sem(df, code="BB72", experiment_kind="z_memory", kernel="powerlaw",
               J0=0.04, p_cnot=1e-3)
    sub = sub[sub["alpha"].isin([2.0, 3.0, 5.0])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(COL1, COL1 / 1.4))
    _emb_plot(ax, sub, "alpha", detected_only=False)
    ax.axvline(x=2.0, ls=":", lw=0.8, color=_ORANGE, zorder=0,
               label=r"$\alpha_c = 2$")
    ax.set_yscale("log")
    _log_ticks(ax, "y")
    _grid(ax)
    ax.set_xlabel(r"Decay exponent $\alpha$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.legend(loc="center right", fontsize=5.5)
    fig.tight_layout()
    _save(fig, outdir, "bb72_la_alpha_minislice")


# ═══════════════════════════════════════════════════════════════════════
#  BB90 + BB108 combined (appendix)
# ═══════════════════════════════════════════════════════════════════════

def plot_bb90_bb108_combined(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    fig, (ax90, ax108) = plt.subplots(1, 2, figsize=(COL2, COL2 / 2.8), sharey=True)

    for ax, code, title in [(ax90, "BB90", r"$[\![90,8,10]\!]$"),
                             (ax108, "BB108", r"$[\![108,8,12]\!]$")]:
        sub = _sem(df, code=code, experiment_kind="z_memory", kernel="powerlaw",
                   p_cnot=1e-3)
        sub = sub[sub["alpha"] == 3.0]
        if sub.empty:
            continue
        _emb_plot(ax, sub, "J0")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_ticks(ax)
        _grid(ax)
        ax.set_xlabel(r"$J_0$")
        ax.set_title(title, fontsize=8)
        if ax is ax90:
            ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
            ax.legend(loc="upper left", fontsize=5.5)

    fig.tight_layout(w_pad=0.5)
    _save(fig, outdir, "bb90_bb108_j0_slices_combined")


# ═══════════════════════════════════════════════════════════════════════
#  BB144 combined scaling (main text)
# ═══════════════════════════════════════════════════════════════════════

def plot_bb144_combined(df: pd.DataFrame, outdir: Path) -> None:
    _apply_style()
    fig, (ax_j0, ax_per) = plt.subplots(1, 2, figsize=(COL2, COL2 / 2.8))

    # T1: J0 sweep
    sub_j0 = _sem(df, code="BB144", experiment_kind="z_memory", kernel="powerlaw",
                  p_cnot=1e-3)
    sub_j0 = sub_j0[sub_j0["alpha"] == 3.0]
    _emb_plot(ax_j0, sub_j0, "J0")
    ax_j0.set_xscale("log")
    ax_j0.set_yscale("log")
    _log_ticks(ax_j0)
    _grid(ax_j0)
    ax_j0.set_xlabel(r"$J_0$")
    ax_j0.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax_j0.legend(loc="upper left", fontsize=5.5)

    # T2: PER sweep
    sub_per = _sem(df, code="BB144", experiment_kind="z_memory", kernel="powerlaw",
                   J0=0.04)
    sub_per = sub_per[sub_per["alpha"] == 3.0]
    _emb_plot(ax_per, sub_per, "p_cnot")
    ax_per.set_xscale("log")
    ax_per.set_yscale("log")
    _log_ticks(ax_per)
    _grid(ax_per)
    ax_per.set_xlabel(r"Physical error rate $p$")
    ax_per.legend(loc="lower right", fontsize=5.5)

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "bb144_scaling_combined")


# ═══════════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════════

def plot_v3_suite(df: pd.DataFrame, outdir: Path) -> None:
    """Generate all publication figures."""
    outdir = Path(outdir)
    df = _parse_kernel_params(df)
    # Main text
    plot_s1_crossing_j0(df, outdir)
    plot_s2_powerlaw_j0(df, outdir)
    plot_s3_alpha_sweep(df, outdir)
    plot_s4_xi_sweep(df, outdir)
    plot_s5_per_sweep(df, outdir)
    plot_s6_phase_diagram(df, outdir)
    plot_s6_phase_heatmap(df, outdir)
    plot_g3_exposure_vs_ler(df, outdir)
    plot_t1_bb144_j0(df, outdir)
    plot_t2_bb144_per(df, outdir)
    plot_bb144_combined(df, outdir)
    plot_la_j0_window(df, outdir)
    plot_la_alpha_minislice(df, outdir)
    # Appendix
    plot_scaling_comparison(df, outdir)
    plot_ler_ratio(df, outdir)
    plot_bb90_bb108_combined(df, outdir)
