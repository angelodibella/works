#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate the proxy-kernel hardware-anchoring figure for Appendix E."""

from __future__ import annotations
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HAS_LATEX = bool(shutil.which("latex") and shutil.which("dvipng"))

_BLUE = "#4C72B0"
_RED  = "#C44E52"

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
    "lines.markersize": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.82",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.18,
}

DATADIR = Path(__file__).resolve().parent.parent.parent / "code" / "results-out"
OUTDIR  = Path(__file__).resolve().parent.parent / "figures"


def main():
    plt.rcParams.update(STYLE)

    df = pd.read_csv(DATADIR / "results_proxy_kernel.csv")

    mono = df[df["experiment_id"].str.contains("mono")].sort_values("kernel_params")
    bi   = df[df["experiment_id"].str.contains("bi")].sort_values("kernel_params")

    # Extract xi from kernel_params string like "{'xi': 0.25}"
    import ast
    mono_xi = mono["kernel_params"].apply(lambda s: ast.literal_eval(s)["xi"]).values
    bi_xi   = bi["kernel_params"].apply(lambda s: ast.literal_eval(s)["xi"]).values

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.errorbar(mono_xi, mono["primary_ler_total"],
                yerr=[mono["primary_ler_total"] - mono["primary_ler_total_lo"],
                      mono["primary_ler_total_hi"] - mono["primary_ler_total"]],
                fmt="o-", color=_RED, markersize=5, linewidth=1.4,
                elinewidth=0.8, capsize=2, label="Monomial")
    ax.errorbar(bi_xi, bi["primary_ler_total"],
                yerr=[bi["primary_ler_total"] - bi["primary_ler_total_lo"],
                      bi["primary_ler_total_hi"] - bi["primary_ler_total"]],
                fmt="s-", color=_BLUE, markersize=5, linewidth=1.4,
                elinewidth=0.8, capsize=2, label="Biplanar bounded-thickness")

    # Mark hardware-informed regions
    ax.axvspan(0.15, 0.35, alpha=0.08, color=_RED, zorder=0)
    ax.annotate("flux-like", xy=(np.sqrt(0.15*0.35), 2e-2), fontsize=6, ha="center", color="0.4")
    ax.axvspan(8, 12, alpha=0.08, color=_BLUE, zorder=0)
    ax.annotate("drive", xy=(np.sqrt(8*12), 2e-2), fontsize=6, ha="center", color="0.4")
    ax.annotate(r"$\xi_{\rm drive}\geq 17$", xy=(11, 5e-2),
                fontsize=5.5, ha="right", color="0.5")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Exponential decay length $\xi$ (pitch units)")
    ax.set_ylabel(r"Logical error rate $p_L$")
    ax.set_xlim(0.15, 12)
    ax.set_ylim(0.01, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.18, linestyle="--")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "proxy_kernel_sweep.pdf")
    print(f"Saved {OUTDIR / 'proxy_kernel_sweep.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
