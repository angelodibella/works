#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate microscopic surrogate-validation figures.

Fig 1 (two-block decomposition):
  Two panels showing the Frobenius norms ||A||, ||B||, ||C|| from
  Proposition 1 as functions of theta.  Panel (a) is the stray-drive
  regime (C = 0 identically); panel (b) is the mixed regime where C
  is nonzero but subdominant.

Fig 2 (weight audit):
  Two panels.  Panel (a): crosstalk-induced weight-1 mass M_1(theta)
  for each coupling type, with sin^2(theta) reference.  Panel (b):
  absolute weight->=3 mass M_3(theta), showing monotone decrease from
  the baseline 0.5.

Both figures use the same style as the other publication plots.
Simulations are run inline (16x16 matrix exponentials, ~1 s total).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm, norm as la_norm

# ── Style ────────────────────────────────────────────────────────────

_HAS_LATEX = bool(shutil.which("latex") and shutil.which("dvipng"))

_BLUE   = "#4C72B0"
_RED    = "#C44E52"
_GREEN  = "#55A868"
_ORANGE = "#E17C05"
_GRAY   = "#999999"

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
    "lines.linewidth": 1.4,
    "lines.markersize": 5.2,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.82",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.18,
}

OUTDIR = Path(__file__).resolve().parent.parent / "figures"

# ── Pauli / circuit primitives ───────────────────────────────────────

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
I4  = np.eye(4, dtype=complex)
I16 = np.eye(16, dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out


# ── Two-block decomposition (Proposition 1) ──────────────────────────

def decompose(K):
    """K on 4x4 (x) 4x4 -> (c, A, B, C)."""
    R = K.reshape(4, 4, 4, 4)
    c = np.trace(K).real / 16
    A = np.einsum("ijkj->ik", R) / 4 - c * I4
    B = np.einsum("ijil->jl", R) / 4 - c * I4
    C = K - c * I16 - kron(A, I4) - kron(I4, B)
    return c, A, B, C


def error_hamiltonian(H_ct):
    U_ideal = kron(CNOT, CNOT)
    U_ct = expm(-1j * H_ct)
    U_err = U_ideal.conj().T @ U_ct @ U_ideal
    K = (1j) * logm(U_err)
    return (K + K.conj().T) / 2


def _op4(q, P):
    ops = [I2, I2, I2, I2]
    ops[q] = P
    return kron(*ops)


def H_stray(theta):
    return theta * (_op4(0, Z) + _op4(2, Z))

def H_exchange(theta):
    return theta * _op4(0, Z) @ _op4(2, Z)

def H_mixed(theta, ratio=0.1):
    return H_stray(theta) + H_exchange(theta * ratio)


# ── Weight-audit primitives ──────────────────────────────────────────

def _cnot_n(ctrl, tgt, n=4):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        c_bit = (s >> (n - 1 - ctrl)) & 1
        new = s ^ (1 << (n - 1 - tgt)) if c_bit else s
        U[new, s] = 1
    return U


def _pauli_n(q, P, n=4):
    ops = [I2] * n
    ops[q] = P
    return kron(*ops)


def build_round(theta, coupling):
    n = 4
    U = _cnot_n(2, 3, n)
    if coupling == "stray":
        Hc = theta * (_pauli_n(1, Z, n) + _pauli_n(2, Z, n))
    elif coupling == "exchange":
        Hc = theta * _pauli_n(1, Z, n) @ _pauli_n(2, Z, n)
    else:
        Hc = (theta * (_pauli_n(1, Z, n) + _pauli_n(2, Z, n))
              + 0.1 * theta * _pauli_n(1, Z, n) @ _pauli_n(2, Z, n))
    U = expm(-1j * Hc) @ U
    U = U @ _cnot_n(1, 3, n) @ _cnot_n(0, 3, n)
    return U


def weight(idx):
    return sum(1 for i in idx if i > 0)


def all_paulis_3():
    for a in range(4):
        for b in range(4):
            for c in range(4):
                yield (a, b, c)


def pauli_mat_3(idx):
    return kron(PAULIS[idx[0]], PAULIS[idx[1]], PAULIS[idx[2]])


def weight_spectrum(U_round):
    """Kraus extraction + twirl -> M_0 .. M_3."""
    dim_d, dim_a = 8, 2
    kraus = []
    for j in range(dim_a):
        K = np.zeros((dim_d, dim_d), dtype=complex)
        for do in range(dim_d):
            for di in range(dim_d):
                K[do, di] = U_round[do * dim_a + j, di * dim_a + 0]
        kraus.append(K)
    M = [0.0] * 4
    for idx in all_paulis_3():
        P = pauli_mat_3(idx)
        p = sum(abs(np.trace(P @ K))**2 for K in kraus) / 64
        M[weight(idx)] += p
    return M


# ── Figure 1: two-block decomposition norms ─────────────────────────

def fig_two_block(outpath):
    thetas = np.linspace(0.001, 0.30, 200)

    norms = {
        "stray":  {"A": [], "B": [], "C": []},
        "mixed":  {"A": [], "B": [], "C": []},
    }
    for th in thetas:
        for name, Hct in [("stray", H_stray(th)),
                          ("mixed", H_mixed(th, 0.1))]:
            K = error_hamiltonian(Hct)
            _, A, B, C = decompose(K)
            norms[name]["A"].append(la_norm(A, "fro"))
            norms[name]["B"].append(la_norm(B, "fro"))
            norms[name]["C"].append(la_norm(C, "fro"))

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Local norms (same for stray and mixed — plot once)
    ax.plot(thetas, norms["stray"]["A"], color=_BLUE,
            label=r"$\|\hat A_e\|_F = \|\hat B_{e'}\|_F$ (local)")

    # Inter-block: stray (zero) and mixed (nonzero)
    ax.plot(thetas, norms["stray"]["C"], color=_RED, linewidth=2.0,
            label=r"$\|\hat C_{ee'}\|_F$, stray ($J_2\!=\!0$)")
    ax.plot(thetas, norms["mixed"]["C"], color=_ORANGE, linewidth=1.6,
            linestyle="--",
            label=r"$\|\hat C_{ee'}\|_F$, mixed ($J_2/J_1\!=\!0.1$)")

    ax.axvspan(0, 0.10, alpha=0.06, color="gray", zorder=0)
    ax.annotate("paper window", xy=(0.05, 0.38), fontsize=6,
                ha="center", color="0.45")
    ax.set_xlabel(r"$\theta=\tau J_0\kappa(d)$")
    ax.set_ylabel(r"Frobenius norm")
    ax.legend(fontsize=6.2, loc="upper left")
    ax.grid(True, alpha=0.18, linestyle="--")

    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close(fig)


# ── Figure 2: weight audit ──────────────────────────────────────────

def fig_weight_audit(outpath):
    thetas = np.linspace(0.001, 0.25, 150)

    spectra = {ct: [] for ct in ("stray", "exchange", "mixed")}
    M0_baseline = weight_spectrum(build_round(0.0, "stray"))

    for th in thetas:
        for ct in ("stray", "exchange", "mixed"):
            spectra[ct].append(weight_spectrum(build_round(th, ct)))

    ct_color = {"stray": _BLUE, "exchange": _RED, "mixed": _GREEN}
    ct_label = {
        "stray":    "Stray drive",
        "exchange": "ZZ exchange",
        "mixed":    r"Mixed ($J_2/J_1\!=\!0.1$)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    # Stray and mixed nearly overlap (J2/J1=0.1 is small).
    # Use a wide dashed green line beneath a thinner solid blue line
    # so both are visible simultaneously.
    ct_ls    = {"stray": "-", "exchange": "-", "mixed": (0, (4, 2))}
    ct_lw    = {"stray": 1.2, "exchange": 1.4, "mixed": 2.4}
    ct_order = ["mixed", "exchange", "stray"]  # stray last = on top

    # (a) Weight-1 mass vs theta, with sin^2(theta) reference
    for ct in ct_order:
        M1 = [s[1] for s in spectra[ct]]
        ax1.plot(thetas, M1, color=ct_color[ct], linestyle=ct_ls[ct],
                 linewidth=ct_lw[ct], label=ct_label[ct])

    ax1.plot(thetas, np.sin(thetas)**2, color="0.5", linestyle=":",
             linewidth=1.0, label=r"$\sin^2\theta$")
    ax1.axvspan(0, 0.10, alpha=0.06, color="gray", zorder=0)
    ax1.set_title(r"(a) Crosstalk-induced weight-1 mass $M_1$", fontsize=8.5)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$M_1(\theta)$")
    ax1.legend(fontsize=6.5, loc="upper left")
    ax1.grid(True, alpha=0.18, linestyle="--")


    # (b) Weight->=3 mass vs theta (absolute, showing decrease from 0.5)
    for ct in ct_order:
        Mge3 = [s[3] for s in spectra[ct]]  # only weight-3 for n=3
        ax2.plot(thetas, Mge3, color=ct_color[ct], linestyle=ct_ls[ct],
                 linewidth=ct_lw[ct], label=ct_label[ct])

    ax2.axhline(y=M0_baseline[3], color="0.5", linestyle=":", linewidth=1.0,
                label=r"Baseline ($\theta\!=\!0$)")
    ax2.axvspan(0, 0.10, alpha=0.06, color="gray", zorder=0)
    ax2.set_title(r"(b) Weight-$\geq 3$ mass $M_{\geq 3}$", fontsize=8.5)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$M_{\geq 3}(\theta)$")
    ax2.legend(fontsize=6.5, loc="lower left")
    ax2.grid(True, alpha=0.18, linestyle="--")

    fig.tight_layout(w_pad=1.0)
    fig.savefig(outpath)
    print(f"Saved {outpath}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_two_block(OUTDIR / "microscopic_r_int.pdf")
    fig_weight_audit(OUTDIR / "microscopic_weight_audit.pdf")


if __name__ == "__main__":
    main()
