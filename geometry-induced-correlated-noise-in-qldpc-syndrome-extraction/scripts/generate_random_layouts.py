#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""
Generate random four-column BB72 layouts for the many-layout experiment (advice Item 8).

For each layout:
  - draw random sigma_L, sigma_Z
  - compute J_kappa (max exposure over the 36-support family)
  - save the config JSON for use with FixedPermutationColumnEmbedding

Also includes the monomial and logical-aware layouts for comparison.
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bbstim.bbcode import BBCodeSpec, build_bb72
from bbstim.algebra import enumerate_pure_L_minwt_logicals
from bbstim.embeddings import _J_exposure, _precompute_term_maps

OUTDIR = Path(__file__).resolve().parent.parent / "configs" / "random_layouts"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_RANDOM = 20
SEED = 42
ALPHA = 3.0
R0 = 1.0
TAU = 1.0
J0 = 0.04

def main():
    spec = build_bb72()
    half = spec.half  # 36

    family = [set(s) for s in enumerate_pure_L_minwt_logicals(spec)]
    print(f"BB72: {len(family)} pure-q(L) supports, half={half}")

    B_terms = spec.B_terms
    term_maps = _precompute_term_maps(spec, B_terms)

    results = []

    # Monomial (identity permutation)
    sigma_L_mono = list(range(half))
    sigma_Z_mono = list(range(half))
    j_mono = _J_exposure(spec, sigma_L_mono, sigma_Z_mono, family, B_terms,
                         1.0, 1.0, 3.0, TAU, J0, ALPHA, R0, term_maps)
    results.append(("monomial", sigma_L_mono, sigma_Z_mono, j_mono))
    print(f"  monomial: J_kappa = {j_mono:.6f}")

    # Logical-aware (load from existing config)
    la_config = Path(__file__).resolve().parent.parent / "configs" / "logical_aware_bb72_truefamily.json"
    if la_config.exists():
        payload = json.loads(la_config.read_text())
        sigma_L_la = payload["sigma_L"]
        sigma_Z_la = payload["sigma_Z"]
        j_la = _J_exposure(spec, sigma_L_la, sigma_Z_la, family, B_terms,
                           1.0, 1.0, 3.0, TAU, J0, ALPHA, R0, term_maps)
        results.append(("logical_aware", sigma_L_la, sigma_Z_la, j_la))
        print(f"  logical_aware: J_kappa = {j_la:.6f}")

    # Random layouts
    rng = random.Random(SEED)
    for i in range(N_RANDOM):
        sigma_L = list(range(half))
        sigma_Z = list(range(half))
        rng.shuffle(sigma_L)
        rng.shuffle(sigma_Z)
        j = _J_exposure(spec, sigma_L, sigma_Z, family, B_terms,
                        1.0, 1.0, 3.0, TAU, J0, ALPHA, R0, term_maps)
        results.append((f"random_{i:02d}", sigma_L, sigma_Z, j))
        print(f"  random_{i:02d}: J_kappa = {j:.6f}")

    # Save configs
    for name, sL, sZ, j in results:
        config = {
            "sigma_L": sL,
            "sigma_Z": sZ,
            "x_positions": {"X": 0.0, "L": 1.0, "R": 2.0, "Z": 3.0},
            "scale_y": 1.0,
            "J_kappa": j,
            "note": f"BB72 layout {name} for many-layout experiment"
        }
        out = OUTDIR / f"{name}.json"
        out.write_text(json.dumps(config, indent=2) + "\n")

    # Summary CSV
    import csv
    summary = OUTDIR / "layout_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "J_kappa"])
        for name, _, _, j in sorted(results, key=lambda r: r[3]):
            w.writerow([name, f"{j:.8f}"])

    print(f"\nSaved {len(results)} configs to {OUTDIR}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
