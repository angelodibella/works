#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate publication figures and numeric LaTeX macros from semantic results.

This script is intended to be run from the attached repository and treats
``data/results_semantic.csv`` as the publication source of truth.  It also
uses the repository audit CSVs to pull theorem-facing quantities into the
manuscript.

Example
-------
python scripts/generate_publication_assets.py \
  --repo-root . \
  --outdir ../paper/figures \
  --numbers-tex ../paper/generated/numbers.tex
"""
from __future__ import annotations

import argparse
import ast
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle, Polygon
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HAS_LATEX = bool(shutil.which("latex") and shutil.which("dvipng"))

_BLUE = "#4C72B0"
_RED = "#C44E52"
_GREEN = "#55A868"
_PURPLE = "#8172B2"
_ORANGE = "#E17C05"
_BLACK = "#000000"
_GRAY = "#666666"

EMB_ORDER = ["monomial_column", "logical_aware", "ibm_biplanar"]
EMB_LABEL = {
    "monomial_column": "Monomial",
    "logical_aware": "Logical-aware",
    "ibm_biplanar": "Biplanar bounded-thickness",
}
EMB_COLOR = {
    "monomial_column": _RED,
    "logical_aware": _GREEN,
    "ibm_biplanar": _BLUE,
}
EMB_MARKER = {
    "monomial_column": "o",
    "logical_aware": "D",
    "ibm_biplanar": "s",
}
KERNEL_MARKER = {"crossing": "o", "powerlaw": "s", "exponential": "^"}
KERNEL_LABEL = {"crossing": "Crossing", "powerlaw": "Power law", "exponential": "Exponential"}

STYLE: dict[str, Any] = {
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


# Core data helpers

def setup_style() -> None:
    plt.rcParams.update(STYLE)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["params"] = df["kernel_params"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.strip() else {}
    )
    df["alpha"] = df["params"].apply(lambda d: float(d.get("alpha", np.nan)))
    df["xi"] = df["params"].apply(lambda d: float(d.get("xi", np.nan)))
    df["embedding"] = df["embedding"].replace(
        {"logical_aware_fixed:configs/logical_aware_bb72_truefamily.json": "logical_aware"}
    )
    return df


def qfilter(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    sub = df.copy()
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in {"J0", "p_cnot", "alpha", "xi"}:
            sub = sub[np.isclose(sub[key].astype(float), float(value))]
        else:
            sub = sub[sub[key] == value]
    return sub.copy()


def detected_and_upper(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detected = sub[sub["primary_failures"] > 0].copy()
    upper = sub[sub["primary_failures"] == 0].copy()
    return detected, upper


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def fmt(value: float | int | str, digits: int = 4) -> str:
    if isinstance(value, str):
        return latex_escape(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    x = float(value)
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}" if digits >= 3 else f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    if ax >= 1e-3:
        return f"{x:.4f}"
    return f"{x:.3g}"


def macro_line(name: str, value: Any) -> str:
    return rf"\newcommand{{\{name}}}{{{fmt(value)}}}"


def select_one(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    sub = qfilter(df, **kwargs)
    if len(sub) != 1:
        raise ValueError(f"Expected one row for {kwargs}, found {len(sub)}")
    return sub.iloc[0]


# Plot helpers

def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=9)


def style_ler_axis(ax: plt.Axes, xlabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.2)
    ax.grid(True, which="both", alpha=0.18, linestyle="--")


def add_series(ax: plt.Axes, sub: pd.DataFrame, xcol: str, *, label: str, color: str, marker: str,
               connect_detected: bool = True) -> None:
    sub = sub.sort_values(xcol)
    detected, upper = detected_and_upper(sub)
    if not detected.empty:
        if connect_detected:
            ax.plot(detected[xcol], detected["primary_ler_total"], color=color, lw=1.5, alpha=0.95)
        y = detected["primary_ler_total"].to_numpy(dtype=float)
        lo = detected["primary_ler_total_lo"].to_numpy(dtype=float)
        hi = detected["primary_ler_total_hi"].to_numpy(dtype=float)
        ax.errorbar(
            detected[xcol].to_numpy(dtype=float),
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt=marker,
            color=color,
            mfc=color,
            mec="black",
            mew=0.25,
            capsize=2.0,
            elinewidth=0.75,
            label=label,
            lw=0,
            zorder=3,
        )
    if not upper.empty:
        # Plot exact 95% upper bounds for zero-failure points as open markers.
        ax.scatter(
            upper[xcol].to_numpy(dtype=float),
            upper["primary_ler_total_hi"].to_numpy(dtype=float),
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            s=32,
            zorder=4,
        )


def set_log_x_if_positive(ax: plt.Axes, series: pd.Series) -> None:
    vals = np.asarray(series, dtype=float)
    vals = vals[vals > 0]
    if len(vals) > 0:
        ax.set_xscale("log")
        ax.set_xlim(vals.min() * 0.75, vals.max() * 1.35)


# Figures

def fig_concept_pipeline(outdir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10.4, 3.05))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x0, y0, w, h = 0.03, 0.40, 0.14, 0.46
    rect = Rectangle((x0, y0), w, h, facecolor="#F7F7F7", edgecolor="0.45", linewidth=0.8)
    ax.add_patch(rect)
    cols = np.linspace(x0 + 0.02, x0 + w - 0.02, 4)
    for idx, cx in enumerate(cols):
        ax.plot([cx, cx], [y0 + 0.07, y0 + h - 0.07], color="0.78", lw=0.9)
        ax.text(cx, y0 + h - 0.03, ["$q(X)$", "$q(L)$", "$q(R)$", "$q(Z)$"][idx], ha="center", va="top", fontsize=8)
    ys = np.linspace(y0 + 0.10, y0 + h - 0.13, 4)
    for y in ys:
        ax.plot([cols[1], cols[3]], [y, y + 0.07], color=_BLUE, lw=1.0)
    ax.plot([cols[1], cols[3]], [ys[0], ys[-1]], color=_RED, lw=1.4)
    ax.plot([cols[1], cols[3]], [ys[-1], ys[0]], color=_RED, lw=1.4)
    ax.text(x0 + w/2, y0 - 0.03, "Monomial\nsingle layer", ha="center", va="top", fontsize=8.2)

    x1 = 0.20
    rect = Rectangle((x1, y0), w, h, facecolor="#F7F7F7", edgecolor="0.45", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x1 + 0.055, y0 + h - 0.11, "Layer $G_A$", fontsize=7.2, color="0.35")
    ax.text(x1 + 0.055, y0 + 0.02, "Layer $G_B$", fontsize=7.2, color="0.35")
    cols2 = np.linspace(x1 + 0.02, x1 + w - 0.02, 4)
    for idx, cx in enumerate(cols2):
        ax.plot([cx, cx], [y0 + 0.07, y0 + h - 0.07], color="0.85", lw=0.8)
        ax.text(cx, y0 + h - 0.03, ["$q(X)$", "$q(L)$", "$q(R)$", "$q(Z)$"][idx], ha="center", va="top", fontsize=8)
    ysA = [y0 + 0.32, y0 + 0.25, y0 + 0.18]
    ysB = [y0 + 0.14, y0 + 0.09, y0 + 0.05]
    for y in ysA:
        ax.plot([cols2[1], cols2[3]], [y, y + 0.015], color=_BLUE, lw=1.05)
    for y in ysB:
        ax.plot([cols2[1], cols2[3]], [y, y - 0.015], color=_GREEN, lw=1.05)
    ax.text(x1 + w/2, y0 - 0.03, "Biplanar\nbounded thickness", ha="center", va="top", fontsize=8.0)

    boxes = [
        (0.38, "Routed geometry\n" r"$\varphi \mapsto d_\varphi(e,e')$", "#F3F6FA"),
        (0.54, "Microscopic model\n" r"$\hat H_{\times}^{(t)}$ and twirl", "#F6F7F2"),
        (0.70, "Retained sector model\n" r"$\mathcal N_\varphi^\sigma$", "#F8F4FA"),
        (0.86, "Correlation graph\n" r"$\nu$ and $\mathcal E$", "#F9F3F0"),
    ]
    bw, bh = 0.12, 0.20
    yb = 0.56
    for x, txt, fc in boxes:
        rect = Rectangle((x, yb), bw, bh, facecolor=fc, edgecolor="0.45", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + bw/2, yb + bh/2, txt, ha="center", va="center", fontsize=8.3)
    for i in range(len(boxes)-1):
        x_start = boxes[i][0] + bw
        x_end = boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x_start + 0.004, yb + bh/2), (x_end - 0.004, yb + bh/2),
                                     arrowstyle='-|>', mutation_scale=10, linewidth=1, color='0.35'))
    ax.add_patch(FancyArrowPatch((0.34, 0.62), (0.38 - 0.01, yb + bh/2),
                                 arrowstyle='-|>', mutation_scale=10, linewidth=1, color='0.35'))
    ax.text(0.92, 0.31, "BB72/BB144 validation\n" r"$\mathcal E$--$p_\mathrm{L}$ trend and layout hierarchy",
            fontsize=8.5, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((0.92, yb), (0.92, 0.38), arrowstyle='-|>', mutation_scale=10, linewidth=1, color='0.35'))
    fig.savefig(outdir / "concept_geometry_pipeline.pdf")
    plt.close(fig)


def fig_crossing_j0(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    sub = qfilter(results, code="BB72", decoded_sector="X", kernel="crossing", p_cnot=1e-3)
    for emb in ["monomial_column", "ibm_biplanar"]:
        s = sub[sub["embedding"] == emb]
        add_series(ax, s, "J0", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
    set_log_x_if_positive(ax, sub["J0"])
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(outdir / "bb72_crossing_j0_sweep.pdf")
    plt.close(fig)


def fig_powerlaw_main(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.95, 3.45))
    sub = qfilter(results, code="BB72", decoded_sector="X", kernel="powerlaw", p_cnot=1e-3)

    ax = axs[0]
    left = qfilter(sub, alpha=3.0)
    for emb in EMB_ORDER:
        s = left[left["embedding"] == emb]
        add_series(ax, s, "J0", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
    ax.set_xlim(-0.002, 0.083)
    add_panel_label(ax, "(a)")
    ax.legend(frameon=True, loc="lower right")

    ax = axs[1]
    right = qfilter(sub, J0=0.04)
    for emb in EMB_ORDER:
        s = right[right["embedding"] == emb].sort_values("alpha")
        if s.empty:
            continue
        add_series(ax, s, "alpha", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb], connect_detected=False)
        detected, _ = detected_and_upper(s)
        if len(detected) > 1:
            ax.plot(detected["alpha"], detected["primary_ler_total"], color=EMB_COLOR[emb], lw=1.2, alpha=0.9)
    ax.axvline(2.0, color=_ORANGE, ls=":", lw=0.9, zorder=1)
    style_ler_axis(ax, r"Decay exponent $\alpha$")
    ax.set_xlim(1.4, 5.1)
    ax.text(2.03, 1.4e-4, r"$\alpha=2$", fontsize=7.2, color=_ORANGE, va="bottom")
    add_panel_label(ax, "(b)")
    ax.legend(frameon=True, loc="lower right")

    fig.savefig(outdir / "bb72_powerlaw_main.pdf")
    plt.close(fig)


def fig_additional_bb72(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.35))

    ax = axs[0]
    sub = qfilter(results, code="BB72", decoded_sector="X", kernel="exponential", J0=0.04, p_cnot=1e-3)
    for emb in ["monomial_column", "ibm_biplanar"]:
        s = sub[sub["embedding"] == emb]
        add_series(ax, s, "xi", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Exponential range $\xi$")
    ax.set_xscale("log")
    add_panel_label(ax, "(a)")
    ax.legend(frameon=True, loc="lower right")

    ax = axs[1]
    sub = qfilter(results, code="BB72", decoded_sector="X", kernel="powerlaw", J0=0.04, alpha=3.0)
    for emb in ["monomial_column", "ibm_biplanar"]:
        s = sub[sub["embedding"] == emb]
        add_series(ax, s, "p_cnot", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Physical two-qubit depolarizing rate $p$")
    ax.set_xscale("log")
    add_panel_label(ax, "(b)")
    ax.legend(frameon=True, loc="lower right")

    ax = axs[2]
    mono = qfilter(results, code="BB72", decoded_sector="X", embedding="monomial_column", kernel="powerlaw", p_cnot=3e-3)
    bi = qfilter(results, code="BB72", decoded_sector="X", embedding="ibm_biplanar", kernel="powerlaw", p_cnot=3e-3)
    mono = mono.pivot(index="alpha", columns="J0", values="primary_ler_total").sort_index().sort_index(axis=1)
    bi = bi.pivot(index="alpha", columns="J0", values="primary_ler_total").sort_index().sort_index(axis=1)
    ratio = np.log10((mono / bi).to_numpy(dtype=float))
    im = ax.imshow(ratio, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mono.columns)), [fmt(v) for v in mono.columns])
    ax.set_yticks(range(len(mono.index)), [fmt(v) for v in mono.index])
    ax.set_xlabel(r"$J_0\tau$")
    ax.set_ylabel(r"$\alpha$")
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, va="top", ha="left", fontsize=9, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(r"$\log_{10}(p_\mathrm{L}^{\mathrm{mono}}/p_\mathrm{L}^{\mathrm{bi}})$")

    fig.savefig(outdir / "bb72_additional_sweeps.pdf")
    plt.close(fig)


def fig_exposure_vs_ler(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    baseline = qfilter(results, code="BB72", decoded_sector="X")
    baseline = baseline[(baseline["embedding"] != "logical_aware") & (baseline["primary_failures"] >= 10) & (baseline["reference_weighted_exposure"] > 0)]

    for (emb, kernel), sub in baseline.groupby(["embedding", "kernel"]):
        y = sub["primary_ler_total"].to_numpy(dtype=float)
        lo = sub["primary_ler_total_lo"].to_numpy(dtype=float)
        hi = sub["primary_ler_total_hi"].to_numpy(dtype=float)
        ax.errorbar(
            sub["reference_weighted_exposure"],
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt=KERNEL_MARKER[kernel],
            markersize=5.0,
            linestyle="none",
            mfc=EMB_COLOR[emb],
            mec="black",
            mew=0.3,
            ecolor=EMB_COLOR[emb],
            alpha=0.82,
        )

    la = qfilter(results, code="BB72", decoded_sector="X", embedding="logical_aware")
    la = la[(la["primary_failures"] >= 10) & (la["reference_weighted_exposure"] > 0)]
    if not la.empty:
        y = la["primary_ler_total"].to_numpy(dtype=float)
        lo = la["primary_ler_total_lo"].to_numpy(dtype=float)
        hi = la["primary_ler_total_hi"].to_numpy(dtype=float)
        ax.errorbar(
            la["reference_weighted_exposure"],
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="*",
            markersize=8,
            linestyle="none",
            mfc=EMB_COLOR["logical_aware"],
            mec="black",
            mew=0.35,
            ecolor=EMB_COLOR["logical_aware"],
            alpha=0.9,
        )

    rho, _ = spearmanr(baseline["reference_weighted_exposure"], baseline["primary_ler_total"])
    ax.text(
        0.03,
        0.97,
        rf"Spearman $\rho_\mathrm{{S}}={rho:.3f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Reference weighted exposure $\mathcal{E}^{X}_{\varphi}(L_{\mathrm{ref}})$")
    ax.set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    ax.set_ylim(1e-4, 1.2)
    ax.grid(True, which="both", alpha=0.18, linestyle="--")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EMB_COLOR["monomial_column"], markeredgecolor="black", markeredgewidth=0.3, label="Monomial"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EMB_COLOR["ibm_biplanar"], markeredgecolor="black", markeredgewidth=0.3, label="Biplanar bounded-thickness"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=EMB_COLOR["logical_aware"], markeredgecolor="black", markeredgewidth=0.3, label="Logical-aware"),
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="Crossing"),
        Line2D([0], [0], marker="s", color="black", linestyle="none", label="Power law"),
        Line2D([0], [0], marker="^", color="black", linestyle="none", label="Exponential"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=7, ncol=2, loc="lower right")
    fig.savefig(outdir / "bb72_exposure_vs_ler.pdf")
    plt.close(fig)


def fig_exposure_by_kernel(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.35), sharey=True)
    baseline = qfilter(results, code="BB72", decoded_sector="X")
    baseline = baseline[(baseline["embedding"] != "logical_aware") & (baseline["primary_failures"] >= 10) & (baseline["reference_weighted_exposure"] > 0)]
    for ax, kernel in zip(axs, ["crossing", "powerlaw", "exponential"]):
        sub = baseline[baseline["kernel"] == kernel]
        for emb in ["monomial_column", "ibm_biplanar"]:
            s = sub[sub["embedding"] == emb]
            if s.empty:
                continue
            y = s["primary_ler_total"].to_numpy(dtype=float)
            lo = s["primary_ler_total_lo"].to_numpy(dtype=float)
            hi = s["primary_ler_total_hi"].to_numpy(dtype=float)
            ax.errorbar(
                s["reference_weighted_exposure"],
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt=EMB_MARKER[emb],
                markersize=4.8,
                linestyle="none",
                color=EMB_COLOR[emb],
                mfc=EMB_COLOR[emb],
                mec="black",
                mew=0.25,
                ecolor=EMB_COLOR[emb],
                alpha=0.85,
                label=EMB_LABEL[emb],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.18, linestyle="--")
        ax.set_xlabel(r"$\mathcal{E}^{X}_{\varphi}(L_{\mathrm{ref}})$")
        ax.set_title(KERNEL_LABEL[kernel])
    axs[0].set_ylabel(r"Logical error rate $p_\mathrm{L}$")
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            wrapped = [l.replace("Biplanar bounded-thickness", "Biplanar\nbounded-thickness") for l in labels]
            ax.legend(handles, wrapped, frameon=True, loc="lower right")
    fig.savefig(outdir / "bb72_exposure_by_kernel.pdf")
    plt.close(fig)


def fig_bb144(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.95, 3.45))

    # Wrapped label to avoid legend overlapping data points
    bb144_label = {k: v for k, v in EMB_LABEL.items()}
    bb144_label["ibm_biplanar"] = "Biplanar\nbounded-thickness"

    ax = axs[0]
    sub = qfilter(results, code="BB144", decoded_sector="X", kernel="powerlaw", p_cnot=1e-3, alpha=3.0)
    for emb in ["monomial_column", "ibm_biplanar"]:
        s = sub[sub["embedding"] == emb]
        add_series(ax, s, "J0", label=bb144_label[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
    set_log_x_if_positive(ax, sub["J0"])
    add_panel_label(ax, "(a)")
    ax.legend(frameon=True, loc="lower right")

    ax = axs[1]
    sub = qfilter(results, code="BB144", decoded_sector="X", kernel="powerlaw", J0=0.04, alpha=3.0)
    for emb in ["monomial_column", "ibm_biplanar"]:
        s = sub[sub["embedding"] == emb]
        add_series(ax, s, "p_cnot", label=bb144_label[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Physical two-qubit depolarizing rate $p$")
    ax.set_xscale("log")
    add_panel_label(ax, "(b)")
    ax.legend(frameon=True, loc="lower right")

    fig.savefig(outdir / "bb144_scaling_main.pdf")
    plt.close(fig)


def fig_la_window(results: pd.DataFrame, la_audit: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.85, 3.35), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axs[0]
    sub = qfilter(results, code="BB72", decoded_sector="X", kernel="powerlaw", p_cnot=1e-3, alpha=3.0)
    for emb in EMB_ORDER:
        s = sub[sub["embedding"] == emb]
        if emb == "logical_aware":
            s = s[s["J0"].isin([0.02, 0.03, 0.04, 0.06])]
        add_series(ax, s, "J0", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
    style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
    ax.set_xlim(-0.002, 0.083)
    add_panel_label(ax, "(a)")
    ax.legend(frameon=True, loc="lower right")

    ax = axs[1]
    order = ["monomial", "logical_aware", "biplanar"]
    labels = ["Monomial", "Logical-aware", "Biplanar\nbounded-thickness"]
    colors = [_RED, _GREEN, _BLUE]
    vals = [float(la_audit.loc[la_audit["embedding"] == key, "max_exp"].iloc[0]) for key in order]
    bars = ax.bar(np.arange(3), vals, color=colors, edgecolor="black", linewidth=0.35)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.03, fmt(v), ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel(r"$\max_{L\in\mathcal R_X}\,\mathcal{E}^{X}_{\varphi}(L)$")
    ax.set_ylim(0, max(vals) * 1.35)
    add_panel_label(ax, "(b)")
    ax.grid(True, axis="y", alpha=0.18, linestyle="--")
    fig.savefig(outdir / "bb72_logical_aware_window.pdf")
    plt.close(fig)


def fig_phase(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.85, 3.45), sharey=True)
    embs = ["monomial_column", "ibm_biplanar"]
    titles = ["Monomial", "Biplanar bounded-thickness"]
    for ax, emb, title in zip(axs, embs, titles):
        sub = qfilter(results, code="BB72", decoded_sector="X", kernel="powerlaw", p_cnot=3e-3, embedding=emb)
        grid = sub.pivot(index="alpha", columns="J0", values="primary_ler_total").sort_index().sort_index(axis=1)
        im = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=np.nanmax(grid.to_numpy(dtype=float)))
        ax.set_xticks(range(len(grid.columns)), [fmt(v) for v in grid.columns], rotation=40, ha="right")
        ax.set_yticks(range(len(grid.index)), [fmt(v) for v in grid.index])
        ax.set_xlabel(r"$J_0\tau$")
        ax.set_title(title, fontsize=8)
        for iy in range(len(grid.index)):
            for ix in range(len(grid.columns)):
                val = grid.iloc[iy, ix]
                if np.isnan(val):
                    continue
                color = "black" if val > 0.45 else "white"
                ax.text(ix, iy, f"{val:.2f}" if val >= 0.1 else f"{val:.3f}", ha="center", va="center", fontsize=6.5, color=color)
    axs[0].set_ylabel(r"$\alpha$")
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(r"$p_\mathrm{L}$")
    fig.savefig(outdir / "bb72_phase_diagram_heatmap.pdf")
    plt.close(fig)


def fig_family_slices(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.85, 3.35), sharey=True)
    for ax, code, label in zip(axs, ["BB90", "BB108"], [r"$[\![90,8,8]\!]$ family support", r"$[\![108,8,10]\!]$ benchmark metadata"]):
        sub = qfilter(results, code=code, decoded_sector="X", kernel="powerlaw", p_cnot=1e-3, alpha=3.0)
        for emb in ["monomial_column", "ibm_biplanar"]:
            s = sub[sub["embedding"] == emb]
            add_series(ax, s, "J0", label=EMB_LABEL[emb], color=EMB_COLOR[emb], marker=EMB_MARKER[emb])
        style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
        ax.set_xlim(0.009, 0.042)
        ax.set_title(label, fontsize=8)
    axs[0].legend(frameon=True, loc="upper left")
    fig.savefig(outdir / "bb90_bb108_slices.pdf")
    plt.close(fig)


def fig_biplanar_scaling(results: pd.DataFrame, outdir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for code, color, marker in [("BB72", _BLUE, "o"), ("BB144", _PURPLE, "D")]:
        sub = qfilter(results, code=code, decoded_sector="X", embedding="ibm_biplanar", kernel="powerlaw", p_cnot=1e-3, alpha=3.0)
        add_series(ax, sub, "J0", label=code, color=color, marker=marker)
    style_ler_axis(ax, r"Coupling parameter $J_0\tau$")
    set_log_x_if_positive(ax, qfilter(results, embedding="ibm_biplanar", kernel="powerlaw", p_cnot=1e-3, alpha=3.0)["J0"])
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(outdir / "biplanar_scaling.pdf")
    plt.close(fig)




def _monomial_tex(label: str) -> str:
    if label == '1':
        return r'$1$'
    return '$' + label + '$'


def _proj_3d(x: float, y: float, z: float) -> tuple[float, float]:
    return (x + 0.55 * y, 0.28 * y + 0.95 * z)


def _draw_cnot(ax: plt.Axes, x: float, y_control: float, y_target: float, *, color: str, label: str | None = None,
               label_dx: float = 0.02, label_dy: float = 0.0) -> None:
    ax.plot([x, x], [y_target, y_control], color=color, lw=1.2, zorder=2)
    ax.add_patch(Circle((x, y_control), 0.016, facecolor=color, edgecolor='black', linewidth=0.3, zorder=3))
    ax.add_patch(Circle((x, y_target), 0.020, facecolor='white', edgecolor=color, linewidth=1.0, zorder=3))
    ax.plot([x, x], [y_target - 0.016, y_target + 0.016], color=color, lw=0.9, zorder=4)
    ax.plot([x - 0.016, x + 0.016], [y_target, y_target], color=color, lw=0.9, zorder=4)
    if label:
        ax.text(x + label_dx, 0.5 * (y_control + y_target) + label_dy, label, color=color, fontsize=7.2,
                va='center', ha='left')


def fig_embedding_geometry(repo: Path, outdir: Path) -> None:
    setup_style()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import networkx as nx
    from bbstim.bbcode import build_bb72
    from bbstim.embeddings import IBMBiplanarEmbedding, _support_crossing_graph_edges

    spec = build_bb72()
    support = [3, 12, 21, 24, 27, 33]
    support_A = support[:3]
    support_B = support[3:]

    fig = plt.figure(figsize=(10.8, 3.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.00, 1.35], wspace=0.30)

    # ------------------------------------------------------------------
    # (a) Exact monomial B3 support picture
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(-0.38, 3.38)
    ax.set_ylim(-1.5, 36.8)
    xcols = {'X': 0.0, 'L': 1.0, 'R': 2.0, 'Z': 3.0}
    for reg, x in xcols.items():
        ax.plot([x, x], [0, 35], color='0.82', lw=0.8, zorder=1)
        for row in range(spec.half):
            ax.add_patch(Circle((x, row), 0.028, facecolor='0.75', edgecolor='none', alpha=0.55, zorder=1))
        ax.text(x, 36.0, rf'$q({reg})$', ha='center', va='bottom', fontsize=8.4)

    targets_B3 = {i: spec.mapped_target_index(i, spec.B_terms[2], True, 'Z') for i in support}
    group_color = {i: _BLUE for i in support_A} | {i: _RED for i in support_B}

    for src in support:
        tgt = targets_B3[src]
        c = group_color[src]
        ax.plot([xcols['L'], xcols['Z']], [src, tgt], color=c, lw=1.45, alpha=0.88, zorder=2)

    # crossing markers for the K_{3,3} structure visible in round B3
    def intersection_y(s1, t1, s2, t2):
        m1 = (t1 - s1) / 2.0
        m2 = (t2 - s2) / 2.0
        if abs(m1 - m2) < 1e-12:
            return None
        x = 1.0 + (s2 - s1) / (m1 - m2)
        if x <= 1.0 or x >= 3.0:
            return None
        y = s1 + m1 * (x - 1.0)
        return x, y

    for a in support_A:
        for b in support_B:
            inter = intersection_y(a, targets_B3[a], b, targets_B3[b])
            if inter is None:
                continue
            x, y = inter
            ax.scatter([x], [y], marker='D', s=12, facecolor=_ORANGE, edgecolor='black', linewidth=0.25, zorder=4)

    for src in support:
        c = group_color[src]
        ax.scatter([xcols['L']], [src], s=34, marker='o', facecolor=c, edgecolor='black', linewidth=0.35, zorder=5)
        ax.text(xcols['L'] - 0.12, src, _monomial_tex(spec.monomial(src)), ha='right', va='center', fontsize=7.0, color=c)
        tgt = targets_B3[src]
        ax.scatter([xcols['Z']], [tgt], s=34, marker='o', facecolor=c, edgecolor='black', linewidth=0.35, zorder=5)
        ax.text(xcols['Z'] + 0.12, tgt, rf'${tgt}$', ha='left', va='center', fontsize=7.0, color=c)

    ax.text(2.00, -0.75, r'BB72 reference support in round $B_3$', ha='center', va='top', fontsize=7.6)
    ax.text(2.00, -1.30, r'visible projected crossings: $K_{3,3}$ contribution', ha='center', va='top', fontsize=7.1, color='0.35')
    add_panel_label(ax, '(a)')

    # ------------------------------------------------------------------
    # (b) Exact support-induced correlation graph and matching
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_xlim(-0.55, 4.70)
    ax.set_ylim(-0.95, 3.55)

    sigma = list(range(spec.half))
    edge_set = _support_crossing_graph_edges(spec, sigma, sigma, set(support), list(spec.B_terms))
    G = nx.Graph()
    G.add_nodes_from(support)
    G.add_edges_from(edge_set)
    matching = nx.max_weight_matching(G, maxcardinality=True)

    pos = {
        3: (0.0, 3.0), 12: (0.0, 1.5), 21: (0.0, 0.0),
        24: (3.0, 3.0), 27: (3.0, 1.5), 33: (3.0, 0.0),
    }
    for u, v in sorted(edge_set):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        rad = -0.18 if {u, v} == {24, 33} else 0.0
        patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-',
                                connectionstyle=f'arc3,rad={rad}', linewidth=1.0,
                                color='0.70', zorder=1)
        ax.add_patch(patch)
    for u, v in matching:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-',
                                connectionstyle='arc3,rad=0.0', linewidth=2.3,
                                color=_GREEN, zorder=2)
        ax.add_patch(patch)

    for node in support:
        x, y = pos[node]
        c = _BLUE if node in support_A else _RED
        ax.scatter([x], [y], s=62, marker='o', facecolor=c, edgecolor='black', linewidth=0.4, zorder=3)
        ax.text(x + (-0.16 if node in support_A else 0.16), y, _monomial_tex(spec.monomial(node)),
                ha='right' if node in support_A else 'left', va='center', fontsize=7.0, color=c)

    ax.text(1.5, 3.34, r'$C_{\varphi}^{X}[S_{\mathrm{ref}}]=K_{3,3}\cup \Delta_B$', ha='center', va='bottom', fontsize=7.8)
    ax.text(1.5, -0.58, r'one maximum matching: $\nu_{\varphi}^{X}(L_{\mathrm{ref}})=3$', ha='center', va='top', fontsize=7.5, color=_GREEN)
    add_panel_label(ax, '(b)')

    # ------------------------------------------------------------------
    # (c) Exact Biplanar bounded-thickness embedding schematic
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    bi = IBMBiplanarEmbedding(spec)
    subset_idx = [i for i in range(spec.half) if spec.ab(i)[0] <= 4 and spec.ab(i)[1] <= 2]
    subset_nodes = []
    for i in subset_idx:
        subset_nodes.extend([('L', i), ('X', i), ('Z', i), ('R', i)])
    base_xy = np.array([bi.base_coords[n] for n in subset_nodes])
    xmin, ymin = base_xy.min(axis=0) - 0.55
    xmax, ymax = base_xy.max(axis=0) + 0.55

    def plane_poly(z):
        corners = [(xmin, ymin, z), (xmax, ymin, z), (xmax, ymax, z), (xmin, ymax, z)]
        return [ _proj_3d(*p) for p in corners ]

    lower = Polygon(plane_poly(-bi.h), closed=True, facecolor=_RED, edgecolor=_RED, alpha=0.08, linewidth=0.8)
    base = Polygon(plane_poly(0.0), closed=True, facecolor='0.65', edgecolor='0.45', alpha=0.10, linewidth=0.8)
    upper = Polygon(plane_poly(+bi.h), closed=True, facecolor=_BLUE, edgecolor=_BLUE, alpha=0.08, linewidth=0.8)
    for patch in [lower, base, upper]:
        ax.add_patch(patch)

    def plot_route(poly, color, lw=1.4, ls='-'):
        pts = np.array([_proj_3d(*p) for p in poly])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, solid_capstyle='round', zorder=3)

    # Actual example routes taken directly from the implemented embedding.
    gA2 = bi.routing_geometry(control_reg='X', target_reg='L', term_name='A2', term=spec.A_terms[1], transpose=False)
    gB2 = bi.routing_geometry(control_reg='L', target_reg='Z', term_name='B2', term=spec.B_terms[1], transpose=True)
    gB3 = bi.routing_geometry(control_reg='L', target_reg='Z', term_name='B3', term=spec.B_terms[2], transpose=True)
    example_routes = [
        (gA2.edge_polylines[('X', 0, 'L', 1)], _BLUE, r'$A_2$'),
        (gB2.edge_polylines[('L', 6, 'Z', 12)], _RED, r'$B_2$'),
        (gB3.edge_polylines[('L', 24, 'Z', 0)], _BLUE, r'$B_3$'),
    ]
    for poly, color, lab in example_routes:
        plot_route(poly, color)
        mid = poly[len(poly) // 2]
        px, py = _proj_3d(*mid)
        ax.text(px + 0.10, py + 0.05, lab, color=color, fontsize=7.0, ha='left', va='bottom')

    for node in subset_nodes:
        x, y = bi.base_coords[node]
        px, py = _proj_3d(x, y, 0.0)
        reg = node[0]
        if reg in {'L', 'R'}:
            ax.scatter([px], [py], s=8, marker='o', facecolor=_GREEN, edgecolor='none', alpha=0.85, zorder=4)
        elif reg == 'X':
            ax.scatter([px], [py], s=12, marker='s', facecolor='white', edgecolor=_PURPLE, linewidth=0.7, zorder=4)
        else:
            ax.scatter([px], [py], s=12, marker='D', facecolor='white', edgecolor=_ORANGE, linewidth=0.7, zorder=4)

    ax.text(*_proj_3d(xmax + 0.35, ymax - 0.15, +bi.h), r'$\Pi_{+}: G_A=\{A_2,A_3,B_3\}$', fontsize=7.3, color=_BLUE, ha='left', va='center')
    ax.text(*_proj_3d(xmax + 0.35, ymax - 0.15, 0.0), r'$\Pi_{0}$: common toric base plane', fontsize=7.3, color='0.35', ha='left', va='center')
    ax.text(*_proj_3d(xmax + 0.35, ymax - 0.15, -bi.h), r'$\Pi_{-}: G_B=\{A_1,B_1,B_2\}$', fontsize=7.3, color=_RED, ha='left', va='center')
    ax.text(*_proj_3d(xmin - 0.10, ymin - 0.35, 0.0), r'base placement: $q(L),(q(X),q(Z)),q(R)$ on the toric unit cell', fontsize=7.0, color='0.35', ha='left', va='top')

    all_pts = []
    for z in (-bi.h, 0.0, +bi.h):
        all_pts.extend(plane_poly(z))
    all_pts = np.array(all_pts)
    ax.set_xlim(all_pts[:, 0].min() - 0.55, all_pts[:, 0].max() + 2.9)
    ax.set_ylim(all_pts[:, 1].min() - 0.65, all_pts[:, 1].max() + 0.55)
    add_panel_label(ax, '(c)')

    fig.savefig(outdir / 'embedding_geometry_story.pdf')
    plt.close(fig)


def fig_syndrome_extraction(repo: Path, outdir: Path) -> None:
    setup_style()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from bbstim.bbcode import build_bb72
    from bbstim.circuit import ibm_round_specs

    spec = build_bb72()
    round_specs = ibm_round_specs(spec)

    fig = plt.figure(figsize=(10.4, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.22)

    # ------------------------------------------------------------------
    # (a) Exact depth-8 IBM cycle used by the repository circuit builder
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0.15, 8.95)
    ax.set_ylim(-0.55, 3.75)
    yreg = {'X': 3.0, 'L': 2.0, 'R': 1.0, 'Z': 0.0}
    for reg, y in yreg.items():
        ax.plot([0.45, 8.65], [y, y], color='0.25', lw=0.8)
        ax.text(0.30, y, rf'$q({reg})$', ha='right', va='center', fontsize=8.2)

    for idx, (rname, ops, _idles) in enumerate(round_specs, start=1):
        if rname in {'R3', 'R4', 'R5'}:
            ax.add_patch(Rectangle((idx - 0.43, -0.30), 0.86, 3.55, facecolor=_RED, alpha=0.07, edgecolor='none', zorder=0))
        ax.text(idx, 3.50, rname, ha='center', va='bottom', fontsize=7.5)
        if len(ops) == 1:
            xvals = [idx]
        else:
            xvals = [idx - 0.11, idx + 0.11]
        for x, op in zip(xvals, ops):
            control_reg, target_reg, term_name, _term, _transpose = op
            color = _BLUE if term_name.startswith('A') else _RED
            _draw_cnot(ax, x, yreg[control_reg], yreg[target_reg], color=color, label=rf'${term_name}$', label_dx=0.05)

    # preparation/measurement boxes exactly as in build_bb_memory_experiment()
    boxprops = dict(boxstyle='round,pad=0.18', fc='white', ec='0.55', linewidth=0.7)
    ax.text(0.42, yreg['Z'] + 0.28, r'Init$_Z$', ha='left', va='bottom', fontsize=7.1, bbox=boxprops)
    ax.text(1.00, yreg['X'] + 0.28, r'Init$_X$', ha='center', va='bottom', fontsize=7.1, bbox=boxprops)
    ax.text(7.38, yreg['Z'] - 0.35, r'Meas$_Z$', ha='center', va='top', fontsize=7.1, bbox=boxprops)
    ax.text(8.00, yreg['X'] + 0.28, r'Meas$_X$', ha='center', va='bottom', fontsize=7.1, bbox=boxprops)
    ax.text(8.62, yreg['Z'] + 0.28, r'Init$_Z$', ha='right', va='bottom', fontsize=7.1, bbox=boxprops)
    ax.text(4.0, -0.45, r'All local circuit-level noise channels act on the full physical cycle; geometry channels are inserted only on sector-relevant routed operations.',
            ha='center', va='top', fontsize=7.0, color='0.35')
    add_panel_label(ax, '(a)')

    # ------------------------------------------------------------------
    # (b) Sector-resolved propagation picture
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # universal CNOT conjugation rules
    box = FancyBboxPatch((0.05, 0.71), 0.90, 0.23, boxstyle='round,pad=0.02', facecolor='#F7F7F7', edgecolor='0.7', linewidth=0.8)
    ax.add_patch(box)
    ax.text(0.50, 0.89, r'CNOT conjugation rules used throughout the sector reduction', ha='center', va='center', fontsize=8.0)
    ax.text(0.10, 0.80, r'$\mathrm{CX}^{\dagger}X_c\,\mathrm{CX}=X_cX_t$', fontsize=7.6, ha='left', va='center')
    ax.text(0.10, 0.74, r'$\mathrm{CX}^{\dagger}X_t\,\mathrm{CX}=X_t$', fontsize=7.6, ha='left', va='center')
    ax.text(0.56, 0.80, r'$\mathrm{CX}^{\dagger}Z_c\,\mathrm{CX}=Z_c$', fontsize=7.6, ha='left', va='center')
    ax.text(0.56, 0.74, r'$\mathrm{CX}^{\dagger}Z_t\,\mathrm{CX}=Z_cZ_t$', fontsize=7.6, ha='left', va='center')

    # X sector
    ax.text(0.27, 0.63, r'$X$ sector', color=_BLUE, fontsize=8.0, ha='center', va='center')
    ax.plot([0.07, 0.45], [0.56, 0.56], color='0.25', lw=0.8)
    ax.plot([0.07, 0.45], [0.44, 0.44], color='0.25', lw=0.8)
    ax.text(0.05, 0.56, r'$q(L)$', ha='right', va='center', fontsize=7.6)
    ax.text(0.05, 0.44, r'$q(Z)$', ha='right', va='center', fontsize=7.6)
    _draw_cnot(ax, 0.18, 0.56, 0.44, color=_RED, label=r'$B_r$', label_dx=0.04)

    # Z sector
    ax.text(0.27, 0.28, r'$Z$ sector', color=_PURPLE, fontsize=8.0, ha='center', va='center')
    ax.plot([0.07, 0.45], [0.21, 0.21], color='0.25', lw=0.8)
    ax.plot([0.07, 0.45], [0.09, 0.09], color='0.25', lw=0.8)
    ax.text(0.05, 0.21, r'$q(X)$', ha='right', va='center', fontsize=7.6)
    ax.text(0.05, 0.09, r'$q(R)$', ha='right', va='center', fontsize=7.6)
    _draw_cnot(ax, 0.18, 0.21, 0.09, color=_BLUE, label=r'$B_r$', label_dx=0.04)

    # repository scope box
    box2 = FancyBboxPatch((0.56, 0.08), 0.37, 0.49, boxstyle='round,pad=0.02', facecolor='white', edgecolor='0.75', linewidth=0.8)
    ax.add_patch(box2)
    ax.text(0.745, 0.53, r'Repository scope in the production data', fontsize=7.8, ha='center', va='center')
    ax.text(0.59, 0.45, r'full depth-$8$ circuit for local noise', fontsize=7.2, ha='left', va='center')
    ax.text(0.59, 0.38, r'theory-reduced geometry on sector-relevant $B$ rounds', fontsize=7.2, ha='left', va='center')
    ax.text(0.59, 0.25, r'$X$-sector rounds: $q(L)	o q(Z)$ with $B_1,B_2,B_3$', fontsize=7.2, ha='left', va='center')
    ax.text(0.59, 0.17, r'$Z$-sector rounds: $q(X)	o q(R)$ with $B_2,B_1,B_3$', fontsize=7.2, ha='left', va='center')
    add_panel_label(ax, '(b)')

    fig.savefig(outdir / 'syndrome_extraction_and_propagation.pdf')
    plt.close(fig)

# Number macros

def build_numbers(results: pd.DataFrame, geom: pd.DataFrame, la_audit: pd.DataFrame,
                  purel: pd.DataFrame, intermediate: pd.DataFrame, numbers_tex: Path) -> None:
    macros: dict[str, Any] = {}
    baseline = qfilter(results, code="BB72", decoded_sector="X")
    baseline = baseline[(baseline["embedding"] != "logical_aware") & (baseline["primary_failures"] >= 10) & (baseline["reference_weighted_exposure"] > 0)]
    rho, pval = spearmanr(baseline["reference_weighted_exposure"], baseline["primary_ler_total"])
    macros["nRawRows"] = len(pd.read_csv(numbers_tex.parents[1] / "data" / "results.csv")) if (numbers_tex.parents[1] / "data" / "results.csv").exists() else len(results)
    macros["nSemanticTotal"] = len(results)
    macros["nMergedSemanticPoints"] = int((results["n_merged"] > 1).sum()) if "n_merged" in results.columns else 0
    code_macro_names = {
        "BB72": "nBBSeventyTwoPoints",
        "BB90": "nBBNinetyPoints",
        "BB108": "nBBOneOhEightPoints",
        "BB144": "nBBOneFortyFourPoints",
    }
    for code_name, macro_name in code_macro_names.items():
        macros[macro_name] = int((results['code'] == code_name).sum())
    macros["baselineCorrelationN"] = len(baseline)
    macros["baselineSpearmanRho"] = f"{rho:.3f}"
    macros["baselineSpearmanP"] = f"{pval:.2e}".replace("e-", r"\times10^{-").rstrip("0").rstrip(".") + "}"

    mono_cross = select_one(results, code="BB72", embedding="monomial_column", kernel="crossing", decoded_sector="X", J0=0.04, p_cnot=1e-3)
    bi_cross = select_one(results, code="BB72", embedding="ibm_biplanar", kernel="crossing", decoded_sector="X", J0=0.04, p_cnot=1e-3)
    macros["bbSeventyTwoCrossMonoLER"] = mono_cross["primary_ler_total"]
    macros["bbSeventyTwoCrossMonoLERLo"] = mono_cross["primary_ler_total_lo"]
    macros["bbSeventyTwoCrossMonoLERHi"] = mono_cross["primary_ler_total_hi"]
    macros["bbSeventyTwoCrossBiLERHi"] = bi_cross["primary_ler_total_hi"]
    macros["bbSeventyTwoDistance"] = 6
    geom_mono_cross = geom[(geom["code"] == "BB72") & (geom["embedding"] == "monomial_column") & (geom["kernel"] == "crossing")].iloc[0]
    macros["bbSeventyTwoCrossSupportCrossings"] = int(geom_mono_cross["support_crossings"])
    macros["bbSeventyTwoCrossMatching"] = int(geom_mono_cross["matching_number"])
    macros["bbSeventyTwoCrossDeffMono"] = int(macros["bbSeventyTwoDistance"] - macros["bbSeventyTwoCrossMatching"])
    macros["bbSeventyTwoCrossDeffBi"] = 6

    mono_pow = select_one(results, code="BB72", embedding="monomial_column", kernel="powerlaw", decoded_sector="X", J0=0.04, p_cnot=1e-3, alpha=3.0)
    bi_pow = select_one(results, code="BB72", embedding="ibm_biplanar", kernel="powerlaw", decoded_sector="X", J0=0.04, p_cnot=1e-3, alpha=3.0)
    la_pow = select_one(results, code="BB72", embedding="logical_aware", kernel="powerlaw", decoded_sector="X", J0=0.04, p_cnot=1e-3, alpha=3.0)
    macros["bbSeventyTwoPowerMonoLER"] = mono_pow["primary_ler_total"]
    macros["bbSeventyTwoPowerMonoLERLo"] = mono_pow["primary_ler_total_lo"]
    macros["bbSeventyTwoPowerMonoLERHi"] = mono_pow["primary_ler_total_hi"]
    macros["bbSeventyTwoPowerBiLER"] = bi_pow["primary_ler_total"]
    macros["bbSeventyTwoPowerBiLERLo"] = bi_pow["primary_ler_total_lo"]
    macros["bbSeventyTwoPowerBiLERHi"] = bi_pow["primary_ler_total_hi"]
    macros["bbSeventyTwoPowerLAFourLER"] = la_pow["primary_ler_total"]
    macros["bbSeventyTwoPowerLAFourLERLo"] = la_pow["primary_ler_total_lo"]
    macros["bbSeventyTwoPowerLAFourLERHi"] = la_pow["primary_ler_total_hi"]
    macros["bbSeventyTwoPowerLERRatio"] = mono_pow["primary_ler_total"] / bi_pow["primary_ler_total"]

    geom_mono_pow = geom[(geom["code"] == "BB72") & (geom["embedding"] == "monomial_column") & (geom["kernel"] == "powerlaw_a3")].iloc[0]
    geom_bi_pow = geom[(geom["code"] == "BB72") & (geom["embedding"] == "ibm_biplanar") & (geom["kernel"] == "powerlaw_a3")].iloc[0]
    macros["bbSeventyTwoMonoExposure"] = geom_mono_pow["weighted_exposure"]
    macros["bbSeventyTwoBiExposure"] = geom_bi_pow["weighted_exposure"]
    macros["bbSeventyTwoPowerAggProbMono"] = geom_mono_pow["agg_pair_prob_max"]
    macros["bbSeventyTwoPowerAggProbBi"] = geom_bi_pow["agg_pair_prob_max"]

    mono_144 = select_one(results, code="BB144", embedding="monomial_column", kernel="powerlaw", decoded_sector="X", J0=0.04, p_cnot=1e-3, alpha=3.0)
    bi_144 = select_one(results, code="BB144", embedding="ibm_biplanar", kernel="powerlaw", decoded_sector="X", J0=0.04, p_cnot=1e-3, alpha=3.0)
    macros["bbOneFortyFourPowerMonoLER"] = mono_144["primary_ler_total"]
    macros["bbOneFortyFourPowerMonoLERLo"] = mono_144["primary_ler_total_lo"]
    macros["bbOneFortyFourPowerMonoLERHi"] = mono_144["primary_ler_total_hi"]
    macros["bbOneFortyFourPowerBiLER"] = bi_144["primary_ler_total"]
    macros["bbOneFortyFourPowerBiLERLo"] = bi_144["primary_ler_total_lo"]
    macros["bbOneFortyFourPowerBiLERHi"] = bi_144["primary_ler_total_hi"]
    macros["bbOneFortyFourPowerLERRatio"] = mono_144["primary_ler_total"] / bi_144["primary_ler_total"]

    macros["bbSeventyTwoPureLFamilyCount"] = len(purel)
    mono_la = la_audit[la_audit["embedding"] == "monomial"].iloc[0]
    la_la = la_audit[la_audit["embedding"] == "logical_aware"].iloc[0]
    bi_la = la_audit[la_audit["embedding"] == "biplanar"].iloc[0]
    macros["bbSeventyTwoLAMaxExposureMono"] = mono_la["max_exp"]
    macros["bbSeventyTwoLAMaxExposureLA"] = la_la["max_exp"]
    macros["bbSeventyTwoLAMaxExposureBi"] = bi_la["max_exp"]
    macros["bbSeventyTwoLAImprovementPct"] = 100.0 * (1.0 - la_la["max_exp"] / mono_la["max_exp"])
    macros["bbSeventyTwoBiImprovementPct"] = 100.0 * (1.0 - bi_la["max_exp"] / mono_la["max_exp"])

    bb90_mono = intermediate[(intermediate["code"] == "BB90") & (intermediate["embedding"] == "monomial_column")].iloc[0]
    bb90_bi = intermediate[(intermediate["code"] == "BB90") & (intermediate["embedding"] == "ibm_biplanar")].iloc[0]
    bb108_mono = intermediate[(intermediate["code"] == "BB108") & (intermediate["embedding"] == "monomial_column")].iloc[0]
    bb108_bi = intermediate[(intermediate["code"] == "BB108") & (intermediate["embedding"] == "ibm_biplanar")].iloc[0]
    bb144_mono = intermediate[(intermediate["code"] == "BB144") & (intermediate["embedding"] == "monomial_column")].iloc[0]
    bb144_bi = intermediate[(intermediate["code"] == "BB144") & (intermediate["embedding"] == "ibm_biplanar")].iloc[0]
    macros["bbNinetyMonoMaxExposure"] = bb90_mono["max_exposure"]
    macros["bbNinetyBiMaxExposure"] = bb90_bi["max_exposure"]
    macros["bbOneOhEightMonoMaxExposure"] = bb108_mono["max_exposure"]
    macros["bbOneOhEightBiMaxExposure"] = bb108_bi["max_exposure"]
    macros["bbOneFortyFourMonoMaxExposure"] = bb144_mono["max_exposure"]
    macros["bbOneFortyFourBiMaxExposure"] = bb144_bi["max_exposure"]

    numbers_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, value in macros.items():
        lines.append(macro_line(name, value))
    numbers_tex.write_text("\n".join(lines) + "\n")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--numbers-tex", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(repo / "data" / "results_semantic.csv")
    geom = pd.read_csv(repo / "data" / "geometry_audit.csv")
    la_audit = pd.read_csv(repo / "data" / "logical_aware_truefamily_optimization.csv")
    purel = pd.read_csv(repo / "data" / "bb72_pureL_minwt_logicals.csv")
    intermediate = pd.read_csv(repo / "data" / "intermediate_bb_geometry_audit.csv")

    fig_concept_pipeline(outdir)
    fig_embedding_geometry(repo, outdir)
    fig_syndrome_extraction(repo, outdir)
    fig_crossing_j0(results, outdir)
    fig_powerlaw_main(results, outdir)
    fig_additional_bb72(results, outdir)
    fig_exposure_vs_ler(results, outdir)
    fig_exposure_by_kernel(results, outdir)
    fig_bb144(results, outdir)
    fig_la_window(results, la_audit, outdir)
    fig_phase(results, outdir)
    fig_family_slices(results, outdir)
    fig_biplanar_scaling(results, outdir)
    build_numbers(results, geom, la_audit, purel, intermediate, args.numbers_tex)
    print(f"Wrote figures to {outdir}")
    print(f"Wrote numbers to {args.numbers_tex}")


if __name__ == "__main__":
    main()
