#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate the many-layout BB72 exposure-vs-LER figure (advice Item 8)."""

from __future__ import annotations
import ast
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HAS_LATEX = bool(shutil.which("latex") and shutil.which("dvipng"))

_BLUE = "#4C72B0"
_RED = "#C44E52"
_GREEN = "#55A868"
_PURPLE = "#8172B2"
_ORANGE = "#E17C05"
_GRAY = "#999999"

STYLE = {
    "text.usetex": _HAS_LATEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.55,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.2,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.82",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.18,
}

DATA = Path(__file__).parent / "data"
RESULTS_OUT = Path(__file__).parent.parent.parent / "code" / "results-out"
LAYOUTS_CSV = Path(__file__).parent.parent.parent / "code" / "configs" / "random_layouts" / "layout_summary.csv"
OUTDIR = Path(__file__).parent.parent / "figures"


def load_many_layout():
    files = sorted(RESULTS_OUT.glob("results_many_layout_*.csv"))
    ml = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ml["layout"] = ml["experiment_id"].str.replace("many_layout_", "")
    summary = pd.read_csv(LAYOUTS_CSV)
    ml = ml.merge(summary, on="layout", how="left")

    # Compute crossing count for each layout from the configs
    # (We have this in the layout configs but let's just use the data we have)
    return ml


def main():
    plt.rcParams.update(STYLE)

    ml = load_many_layout()
    ml = ml.sort_values("J_kappa")

    # Classify layouts
    ml["category"] = "Random"
    ml.loc[ml["layout"] == "monomial", "category"] = "Monomial"
    ml.loc[ml["layout"] == "logical_aware", "category"] = "Logical-aware"

    cat_color = {"Monomial": _RED, "Logical-aware": _GREEN, "Random": _GRAY}
    cat_marker = {"Monomial": "o", "Logical-aware": "D", "Random": "s"}
    cat_zorder = {"Monomial": 10, "Logical-aware": 10, "Random": 5}
    cat_size = {"Monomial": 6.5, "Logical-aware": 6.5, "Random": 4.5}

    fig, ax = plt.subplots(figsize=(4.9, 3.9))

    for cat in ["Random", "Monomial", "Logical-aware"]:
        sub = ml[ml["category"] == cat]
        y = sub["primary_ler_total"].to_numpy(dtype=float)
        lo = sub["primary_ler_total_lo"].to_numpy(dtype=float)
        hi = sub["primary_ler_total_hi"].to_numpy(dtype=float)
        ax.errorbar(
            sub["J_kappa"],
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt=cat_marker[cat],
            markersize=cat_size[cat],
            linestyle="none",
            mfc=cat_color[cat],
            mec="black",
            mew=0.4,
            ecolor=cat_color[cat],
            elinewidth=0.8,
            alpha=0.85,
            zorder=cat_zorder[cat],
            label=cat,
        )

    # Spearman annotation
    rho, pval = spearmanr(ml["J_kappa"], ml["primary_ler_total"])
    pval_exp = int(np.floor(np.log10(pval)))
    pval_man = pval / 10**pval_exp
    ax.text(
        0.03, 0.97,
        rf"Spearman $\rho_{{\mathrm{{S}}}}={rho:.3f}$"
        + f"\n($n={len(ml)}$, "
        + rf"$p={pval_man:.1f}\times10^{{{pval_exp}}}$)",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_xlabel(r"Maximum weighted exposure $J_{\kappa}(\phi;\,\mathcal{R}_X)$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.grid(True, which="both", alpha=0.18, linestyle="--")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_RED,
               markeredgecolor="black", markeredgewidth=0.4, markersize=6.5,
               label="Monomial"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=_GREEN,
               markeredgecolor="black", markeredgewidth=0.4, markersize=6.5,
               label="Logical-aware"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=_GRAY,
               markeredgecolor="black", markeredgewidth=0.4, markersize=4.5,
               label="Random permutation"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=7, loc="lower right")

    fig.savefig(OUTDIR / "bb72_many_layout.pdf")
    print(f"Saved: {OUTDIR / 'bb72_many_layout.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
