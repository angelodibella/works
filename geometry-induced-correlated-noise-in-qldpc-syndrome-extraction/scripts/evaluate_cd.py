#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Estimate the decoder-aware first-order coefficient C_D(phi).

From Theorem 4:  p_L(phi, lambda) = p_{L,0}(phi) + lambda C_D(phi) + O(lambda^2)

We estimate C_D by finite difference at weak coupling:
  C_D(phi) ~ [p_L(phi, J0=delta) - p_L(phi, J0=0)] / delta

Run at the reference power-law kernel (alpha=3, r0=1) with p=1e-3,
6 cycles, for three embeddings. Two J0 values per embedding:
  J0=0 (baseline, no geometry noise)
  J0=0.01 (weak geometry, well within the linear regime)

10,000 shots per point for statistical precision.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bbstim.experiments import Experiment, run_experiment

EMBS = [
    ("monomial_column", "mono"),
    ("logical_aware_fixed:configs/logical_aware_bb72_truefamily.json", "la"),
    ("ibm_biplanar", "bi"),
]
SHOTS = 10000
WORKERS = 10
DELTA = 0.01  # weak coupling for finite difference
OUTDIR = Path(__file__).resolve().parent.parent / "results-out"


def run_point(emb_name, tag, j0):
    exp = Experiment(
        experiment_id=f"cd_{tag}_J{j0:.3f}",
        code="BB72",
        embedding=emb_name,
        experiment_kind="z_memory",
        cycles=6,
        shots=SHOTS,
        p_cnot=1e-3,
        p_idle=1e-3,
        p_prep=1e-3,
        p_meas=1e-3,
        kernel="powerlaw",
        kernel_params={"alpha": 3.0, "r0": 1.0},
        J0=j0,
        tau=1.0,
        use_weak_limit=False,
        primary_decoder="bposd",
        geometry_scope="theory_reduced",
    )
    t0 = time.time()
    row = run_experiment(exp, num_workers=WORKERS, show_progress=True)
    elapsed = time.time() - t0
    ler = row["primary_ler_total"]
    lo = row["primary_ler_total_lo"]
    hi = row["primary_ler_total_hi"]
    print(f"  J0={j0:.3f}: LER={ler:.4f} [{lo:.4f}, {hi:.4f}] ({elapsed:.0f}s)")
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for emb_name, tag in EMBS:
        print(f"\n=== {tag} ===")

        # Baseline (J0=0)
        row0 = run_point(emb_name, tag, 0.0)
        all_rows.append(row0)

        # Weak coupling (J0=delta)
        row1 = run_point(emb_name, tag, DELTA)
        all_rows.append(row1)

        # Finite-difference estimate
        p0 = row0["primary_ler_total"]
        p1 = row1["primary_ler_total"]
        cd = (p1 - p0) / DELTA
        print(f"  C_D({tag}) ~ (p_L({DELTA}) - p_L(0)) / {DELTA}")
        print(f"            = ({p1:.4f} - {p0:.4f}) / {DELTA}")
        print(f"            = {cd:.2f}")

    out = OUTDIR / "results_cd_evaluation.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    print("\n=== C_D Summary ===")
    for emb_name, tag in EMBS:
        rows = [r for r in all_rows if tag in r["experiment_id"]]
        p0 = [r for r in rows if r["J0"] == 0.0][0]["primary_ler_total"]
        p1 = [r for r in rows if r["J0"] == DELTA][0]["primary_ler_total"]
        print(f"  {tag:5s}: p_L(0)={p0:.4f}, p_L({DELTA})={p1:.4f}, "
              f"C_D={((p1 - p0) / DELTA):.2f}")


if __name__ == "__main__":
    main()
