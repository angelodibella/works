#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""
Microscopic two-block Hamiltonian extraction (advice Simulation 1).

Validates the additive-local surrogate H_x(e,e') = J0 kappa(d)(P_e + P_{e'})
by decomposing the exact error Hamiltonian into local and inter-block
components via Proposition 1 (general two-block decomposition) and computing
the interaction ratio r_int = ||C||_F / (||A||_F + ||B||_F).

Three coupling mechanisms are compared:
  (i)   Pure stray drive  — favours the additive-local surrogate (J2 = 0)
  (ii)  Pure direct ZZ    — violates it (J2 dominates)
  (iii) Mixed regime      — interpolates between the two

Output:
  - Detailed decomposition at the paper's operating point (theta = 0.04)
  - Figure: r_int vs theta for all coupling types
  - CSV of sweep data
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm, norm

# Pauli algebra

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS_1Q = [I2, X, Y, Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]

I4 = np.eye(4, dtype=complex)
I16 = np.eye(16, dtype=complex)

# CNOT (control qubit 0, target qubit 1)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# Two-block decomposition (Proposition 1)

def decompose_two_block(K, D_e=4, D_ep=4):
    """
    K on H_e (x) H_{e'} --> c I + A_e (x) I + I (x) B_{e'} + C_{ee'}.

    Returns (c, A_e, B_{e'}, C_{ee'}).
    """
    D = D_e * D_ep
    assert K.shape == (D, D), f"Expected ({D},{D}), got {K.shape}"

    R = K.reshape(D_e, D_ep, D_e, D_ep)

    c = np.trace(K).real / D
    A_e = np.einsum("ijkj->ik", R) / D_ep - c * np.eye(D_e, dtype=complex)
    B_ep = np.einsum("ijil->jl", R) / D_e - c * np.eye(D_ep, dtype=complex)
    C = K - c * I16 - kron(A_e, I4) - kron(I4, B_ep)

    return c, A_e, B_ep, C


def interaction_ratio(K, D_e=4, D_ep=4):
    """r_int = ||C||_F / (||A||_F + ||B||_F)."""
    _, A, B, C = decompose_two_block(K, D_e, D_ep)
    denom = norm(A, "fro") + norm(B, "fro")
    if denom < 1e-15:
        return 0.0 if norm(C, "fro") < 1e-15 else float("inf")
    return norm(C, "fro") / denom


def pauli_decompose_2q(M):
    """Decompose a 4x4 operator into the 16 Pauli coefficients."""
    out = {}
    for i, pi in enumerate(PAULIS_1Q):
        for j, pj in enumerate(PAULIS_1Q):
            P = np.kron(pi, pj)
            label = PAULI_LABELS[i] + PAULI_LABELS[j]
            out[label] = (np.trace(M @ P) / 4).real
    return out


# Crosstalk Hamiltonians
# 4 physical qubits: q0,q1 in block e; q2,q3 in block e'.
# Each block undergoes a CNOT (ctrl=q0/q2, tgt=q1/q3).

def _op4(paulis_per_qubit):
    """Build a 4-qubit operator from single-qubit Pauli assignments."""
    return kron(*paulis_per_qubit)


def H_stray(theta):
    """Stray drive: theta (Z_0 I_1 I_2 I_3 + I_0 I_1 Z_2 I_3)."""
    return theta * (_op4([Z, I2, I2, I2]) + _op4([I2, I2, Z, I2]))


def H_exchange(theta):
    """Direct ZZ exchange: theta Z_0 I_1 Z_2 I_3."""
    return theta * _op4([Z, I2, Z, I2])


def H_mixed(theta, ratio):
    """J1(P_e + P_{e'}) + J2 P_e (x) P_{e'} with J2/J1 = ratio."""
    return H_stray(theta) + H_exchange(theta * ratio)


# Error-Hamiltonian extraction

def error_hamiltonian(H_ct, tau=1.0):
    """
    U_ideal = CNOT_e (x) CNOT_{e'}
    U_ct    = exp(-i H_ct tau)
    U_total = U_ct  U_ideal
    U_err   = U_ideal^dag  U_total
    K_eff   = (i / tau) log(U_err)
    """
    U_ideal = kron(CNOT, CNOT)
    U_ct = expm(-1j * H_ct * tau)
    U_err = U_ideal.conj().T @ U_ct @ U_ideal
    K = (1j / tau) * logm(U_err)
    return (K + K.conj().T) / 2  # enforce Hermiticity


# Sweeps

def sweep_r_int(thetas):
    """Compute r_int vs theta for four coupling types."""
    results = {k: [] for k in ("stray", "exchange", "mixed_01", "mixed_05")}

    for th in thetas:
        results["stray"].append(interaction_ratio(error_hamiltonian(H_stray(th))))
        results["exchange"].append(interaction_ratio(error_hamiltonian(H_exchange(th))))
        results["mixed_01"].append(interaction_ratio(error_hamiltonian(H_mixed(th, 0.1))))
        results["mixed_05"].append(interaction_ratio(error_hamiltonian(H_mixed(th, 0.5))))

    return results


def detailed_report(theta=0.04):
    """Print a decomposition table at a single operating point."""
    print(f"\n{'='*70}")
    print(f"  Two-block decomposition at theta = {theta}")
    print(f"{'='*70}")

    cases = [
        ("Pure stray drive", H_stray(theta)),
        ("Pure ZZ exchange", H_exchange(theta)),
        ("Mixed J2/J1=0.1", H_mixed(theta, 0.1)),
        ("Mixed J2/J1=0.5", H_mixed(theta, 0.5)),
    ]

    for name, Hct in cases:
        K = error_hamiltonian(Hct)
        c, A, B, C = decompose_two_block(K)
        r = interaction_ratio(K)

        pa = pauli_decompose_2q(A)
        pb = pauli_decompose_2q(B)
        dom_a = max(pa.items(), key=lambda x: abs(x[1]))
        dom_b = max(pb.items(), key=lambda x: abs(x[1]))

        print(f"\n  --- {name} ---")
        print(f"  ||K||_F    = {norm(K, 'fro'):.6e}")
        print(f"  |c|        = {abs(c):.6e}")
        print(f"  ||A_e||_F  = {norm(A, 'fro'):.6e}")
        print(f"  ||B_e'||_F = {norm(B, 'fro'):.6e}")
        print(f"  ||C||_F    = {norm(C, 'fro'):.6e}")
        print(f"  r_int      = {r:.6e}")
        print(f"  Dominant A Pauli: {dom_a[0]}  ({dom_a[1]:+.6e})")
        print(f"  Dominant B Pauli: {dom_b[0]}  ({dom_b[1]:+.6e})")


# Plotting

def plot_r_int(thetas, results, outdir: Path):
    """Produce r_int vs theta figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    ax.semilogy(thetas, results["stray"], label="Pure stray drive ($J_2=0$)",
                color="#4C72B0", linewidth=1.6)
    ax.semilogy(thetas, results["exchange"], label=r"Pure ZZ exchange ($J_1=0$)",
                color="#C44E52", linewidth=1.6)
    ax.semilogy(thetas, results["mixed_01"], label=r"Mixed $J_2/J_1=0.1$",
                color="#55A868", linewidth=1.6, linestyle="--")
    ax.semilogy(thetas, results["mixed_05"], label=r"Mixed $J_2/J_1=0.5$",
                color="#E17C05", linewidth=1.6, linestyle="--")

    # Mark the paper's operating window
    ax.axvspan(0, 0.10, alpha=0.08, color="gray")
    ax.annotate(r"paper window ($\theta\leq 0.1$)", xy=(0.05, 0.5),
                fontsize=7, ha="center", color="0.4",
                xycoords=("data", "axes fraction"))

    ax.set_xlabel(r"Dimensionless coupling $\theta = \tau J_0 \kappa(d)$")
    ax.set_ylabel(r"Interaction ratio $r_{\mathrm{int}} = \|C\|/(\|A\|+\|B\|)$")
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)
    ax.set_xlim(0, thetas[-1])
    ax.grid(True, which="both", alpha=0.18, linestyle="--")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "two_block_r_int.pdf", bbox_inches="tight", dpi=600)
    print(f"Saved: {outdir / 'two_block_r_int.pdf'}")
    plt.close(fig)


def save_csv(thetas, results, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta", "r_stray", "r_exchange", "r_mixed_01", "r_mixed_05"])
        for i, th in enumerate(thetas):
            w.writerow([
                f"{th:.6f}",
                f"{results['stray'][i]:.8e}",
                f"{results['exchange'][i]:.8e}",
                f"{results['mixed_01'][i]:.8e}",
                f"{results['mixed_05'][i]:.8e}",
            ])
    print(f"Saved: {outpath}")


# Main

def main():
    outdir = Path(__file__).resolve().parent.parent / "results" / "microscopic"

    # Detailed report at the paper's operating point
    detailed_report(theta=0.04)
    detailed_report(theta=0.10)

    # Sweep
    thetas = np.linspace(0.001, 0.5, 500)
    print("\nRunning r_int sweep over 500 theta values ...")
    results = sweep_r_int(thetas)
    print("  done.")

    # Save
    save_csv(thetas, results, outdir / "two_block_r_int.csv")
    plot_r_int(thetas, results, outdir)

    # Summary for the paper
    # At theta = 0.04 (typical operating point)
    idx04 = np.argmin(np.abs(thetas - 0.04))
    print(f"\nAt theta = 0.04:")
    print(f"  r_int (stray)    = {results['stray'][idx04]:.4e}")
    print(f"  r_int (exchange) = {results['exchange'][idx04]:.4e}")
    print(f"  r_int (mixed 0.1)= {results['mixed_01'][idx04]:.4e}")


if __name__ == "__main__":
    main()
