#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""
One-round propagated Pauli-weight audit (advice Simulation 2).

Takes the exact two-block unitary error channel, propagates it through
a minimal one-round syndrome-extraction subcircuit, eliminates ancillas
via Kraus decomposition, and Pauli-twirls the resulting data channel.

The output is the body-weight spectrum M_k = sum_{wt(P)=k} p_P and
M_{>=3}, which bounds the truncation error of the retained single-and-pair
model (Proposition: controlled low-body truncation).

Subcircuit:  3 data qubits d0, d1, d2 in q(L) connected to 1 ancilla
a0 in q(Z) by three consecutive CNOT rounds (B1, B2, B3).  The B3
round has crosstalk between the d1-a0 and d2-a0 gate edges.

Pauli twirl via Kraus formula: p_P = (1/4^n) sum_i |Tr(P K_i)|^2
where K_i are the data-level Kraus operators obtained from the
unitary + ancilla elimination.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

# Pauli algebra

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS_1Q = [I2, X, Y, Z]


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_weight(indices):
    return sum(1 for i in indices if i > 0)


def all_paulis(n):
    if n == 0:
        yield ()
        return
    for rest in all_paulis(n - 1):
        for i in range(4):
            yield (i,) + rest


def pauli_matrix(indices):
    return kron(*(PAULIS_1Q[i] for i in indices))


# Circuit primitives on n qubits

def _cnot(ctrl, tgt, n):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        c_bit = (s >> (n - 1 - ctrl)) & 1
        new_s = s ^ (1 << (n - 1 - tgt)) if c_bit else s
        U[new_s, s] = 1
    return U


def _pauli_on(pauli, qubit, n):
    ops = [I2] * n
    ops[qubit] = pauli
    return kron(*ops)


# Kraus extraction and Pauli twirl

def extract_kraus(U, n_data, n_ancilla):
    """
    Extract Kraus operators from a unitary on data+ancilla.
    Ancilla initialised in |0>.

    K_j[d_out, d_in] = <d_out, j| U |d_in, 0>

    Qubit ordering: data qubits first, ancilla qubits last.
    """
    dim_d = 2**n_data
    dim_a = 2**n_ancilla
    kraus = []
    for j in range(dim_a):
        K = np.zeros((dim_d, dim_d), dtype=complex)
        for d_out in range(dim_d):
            for d_in in range(dim_d):
                row = d_out * dim_a + j
                col = d_in * dim_a + 0
                K[d_out, d_in] = U[row, col]
        kraus.append(K)
    return kraus


def pauli_twirl_probs(kraus, n):
    """
    p_P = (1/4^n) sum_i |Tr(P K_i)|^2

    Returns dict {pauli_index_tuple: probability}.
    """
    probs = {}
    four_n = 4**n
    for idx in all_paulis(n):
        P = pauli_matrix(idx)
        p = sum(abs(np.trace(P @ K))**2 for K in kraus) / four_n
        probs[idx] = p
    return probs


def weight_spectrum(probs, n):
    """M_k = sum_{wt(P)=k} p_P."""
    M = [0.0] * (n + 1)
    for idx, p in probs.items():
        M[pauli_weight(idx)] += p
    return M


# Subcircuit model
# 4 qubits: d0=q0, d1=q1, d2=q2, a0=q3.
# One syndrome round:
#   B1: CNOT(d0 -> a0)   [no crosstalk]
#   B2: CNOT(d1 -> a0)   [no crosstalk]
#   B3: CNOT(d2 -> a0)   [crosstalk between d1,d2 gate edges]

def build_round_unitary(theta, coupling_type="stray"):
    n = 4

    U_B1 = _cnot(0, 3, n)
    U_B2 = _cnot(1, 3, n)
    U_B3_ideal = _cnot(2, 3, n)

    if coupling_type == "stray":
        H_ct = theta * (_pauli_on(Z, 1, n) + _pauli_on(Z, 2, n))
    elif coupling_type == "exchange":
        H_ct = theta * (_pauli_on(Z, 1, n) @ _pauli_on(Z, 2, n))
    elif coupling_type == "mixed":
        H_ct = (theta * (_pauli_on(Z, 1, n) + _pauli_on(Z, 2, n))
                + 0.1 * theta * _pauli_on(Z, 1, n) @ _pauli_on(Z, 2, n))
    else:
        raise ValueError(coupling_type)

    U_ct = expm(-1j * H_ct)
    U_B3 = U_ct @ U_B3_ideal

    return U_B3 @ U_B2 @ U_B1


# Sweep

def sweep_weight_spectrum(thetas):
    """
    Compute differential weight spectrum: delta_M_k = M_k(theta) - M_k(0).

    At theta=0 the channel is E(rho) = (rho + ZZZ rho ZZZ)/2, so
    M_0^baseline = 0.5, M_3^baseline = 0.5, M_1 = M_2 = 0. The
    crosstalk contribution is entirely captured by the differential.
    """
    # Baseline (theta=0)
    U0 = build_round_unitary(0.0, "stray")  # coupling type irrelevant at theta=0
    kraus0 = extract_kraus(U0, n_data=3, n_ancilla=1)
    probs0 = pauli_twirl_probs(kraus0, n=3)
    M0 = weight_spectrum(probs0, n=3)

    results = {}
    for ct in ("stray", "exchange", "mixed"):
        M_all = []
        dM_all = []
        for th in thetas:
            U = build_round_unitary(th, ct)
            kraus = extract_kraus(U, n_data=3, n_ancilla=1)
            probs = pauli_twirl_probs(kraus, n=3)
            M = weight_spectrum(probs, n=3)
            dM = [M[k] - M0[k] for k in range(len(M))]
            M_all.append(M)
            dM_all.append(dM)
        results[ct] = {"M": M_all, "dM": dM_all}
    results["baseline"] = M0
    return results


# Output

def print_table(thetas, results, sample_thetas=(0.01, 0.04, 0.10, 0.20)):
    M0 = results["baseline"]
    print(f"\n{'='*80}")
    print("  Differential Pauli-weight spectrum (relative to theta=0 baseline)")
    print(f"  Baseline: M_0={M0[0]:.4f}, M_1={M0[1]:.4e}, M_2={M0[2]:.4e}, "
          f"M_3={M0[3]:.4e}")
    print(f"{'='*80}")

    for ct in ("stray", "exchange", "mixed"):
        print(f"\n  --- Coupling: {ct} ---")
        hdr = (f"  {'theta':>8s}  {'dM_1':>10s}  {'dM_2':>10s}  "
               f"{'dM_>=3':>10s}  {'dM_>=3/(dM_1+dM_2)':>20s}")
        print(hdr)
        print(f"  {'-'*58}")
        for st in sample_thetas:
            idx = np.argmin(np.abs(thetas - st))
            dM = results[ct]["dM"][idx]
            dm1 = dM[1]
            dm2 = dM[2]
            dm_ge3 = sum(dM[3:])
            denom = dm1 + dm2
            ratio = dm_ge3 / denom if denom > 1e-15 else 0.0
            print(f"  {thetas[idx]:8.4f}  {dm1:10.4e}  {dm2:10.4e}  "
                  f"{dm_ge3:10.4e}  {ratio:20.6f}")


def plot_weight_spectrum(thetas, results, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    titles = {"stray": "Pure stray drive ($J_2=0$)",
              "exchange": "Pure ZZ exchange ($J_1=0$)",
              "mixed": r"Mixed ($J_2/J_1=0.1$)"}
    colors = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}

    for ax, ct in zip(axes, ("stray", "exchange", "mixed")):
        dM_arr = np.array(results[ct]["dM"])  # shape (N, 4)
        for k in (1, 2):
            vals = np.abs(dM_arr[:, k])
            ax.semilogy(thetas, np.maximum(vals, 1e-20),
                        label=rf"$\Delta M_{k}$", color=colors[k], linewidth=1.4)
        dm_ge3 = np.abs(dM_arr[:, 3:].sum(axis=1))
        ax.semilogy(thetas, np.maximum(dm_ge3, 1e-20),
                    label=r"$\Delta M_{\geq 3}$", color=colors[3],
                    linewidth=1.4, linestyle="--")

        ax.set_title(titles[ct], fontsize=9)
        ax.set_xlabel(r"$\theta = \tau J_0 \kappa(d)$", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.18, linestyle="--")
        ax.axvspan(0, 0.10, alpha=0.06, color="gray")
        ax.set_xlim(0, thetas[-1])
        ax.set_ylim(1e-8, 1)

    axes[0].set_ylabel("Pauli-weight mass", fontsize=9)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "weight_audit.pdf", bbox_inches="tight", dpi=600)
    print(f"\nSaved: {outdir / 'weight_audit.pdf'}")
    plt.close(fig)


def save_csv(thetas, results, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta", "coupling", "dM_0", "dM_1", "dM_2", "dM_ge3"])
        for ct in ("stray", "exchange", "mixed"):
            for i, th in enumerate(thetas):
                dM = results[ct]["dM"][i]
                dm_ge3 = sum(dM[3:])
                w.writerow([f"{th:.6f}", ct,
                            f"{dM[0]:.10e}", f"{dM[1]:.10e}",
                            f"{dM[2]:.10e}", f"{dm_ge3:.10e}"])
    print(f"Saved: {outpath}")


# Main

def main():
    outdir = Path(__file__).resolve().parent.parent / "results" / "microscopic"

    thetas = np.linspace(0.001, 0.30, 200)
    print("Running weight-spectrum sweep (200 theta values, 3 coupling types) ...")
    results = sweep_weight_spectrum(thetas)
    print("  done.")

    # Verify normalisation
    M_test = results["stray"]["M"][100]
    print(f"\nNormalisation check (stray, theta={thetas[100]:.4f}): sum M_k = {sum(M_test):.8f}")

    print_table(thetas, results)

    # Flatten for CSV
    save_csv(thetas, results, outdir / "weight_audit.csv")
    plot_weight_spectrum(thetas, results, outdir)

    # Key numbers for the paper
    idx04 = np.argmin(np.abs(thetas - 0.04))
    for ct in ("stray", "exchange", "mixed"):
        dM = results[ct]["dM"][idx04]
        dm1, dm2 = dM[1], dM[2]
        dm_ge3 = sum(dM[3:])
        ratio = dm_ge3 / (dm1 + dm2) if (dm1 + dm2) > 1e-15 else 0.0
        print(f"\nAt theta=0.04, {ct}:")
        print(f"  dM_1={dm1:.4e}, dM_2={dm2:.4e}, dM_ge3={dm_ge3:.4e}")
        print(f"  dM_ge3 / (dM_1 + dM_2) = {ratio:.4f}")


if __name__ == "__main__":
    main()
