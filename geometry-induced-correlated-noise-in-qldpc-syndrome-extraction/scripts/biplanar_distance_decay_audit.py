#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Deterministic distance-decay audit for the IBM-inspired bounded-thickness embedding.

Explores a small admissible family of geometry parameters (layer height h,
lane epsilon) and reports the true-family max weighted exposure for BB72.

Uses the algebraically-enumerated pure-L family (36 supports).

Outputs:
  results/biplanar_distance_decay_audit.csv
  results/biplanar_distance_decay_audit.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code, get_kernel
from bbstim.embeddings import (
    IBMToricBiplanarEmbedding,
    MonomialColumnEmbedding,
    enumerate_pure_L_minwt_logicals,
)
from bbstim.geometry import (
    pairwise_round_coefficients,
    weighted_exposure_on_support,
)

J0 = 0.04
TAU = 1.0
ALPHA = 3.0

# Parameter grid for bounded-thickness exploration
H_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
LANE_EPS_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0]  # relative to default


def get_b_rounds(spec, emb):
    rounds = []
    for term, tname in zip(spec.B_terms, ['B1', 'B2', 'B3']):
        g = emb.routing_geometry(
            control_reg='L', target_reg='Z',
            term_name=tname, term=term, transpose=True, name=tname,
        )
        rounds.append(g)
    return rounds


def compute_family_exposure(spec, emb, kernel_fn, family):
    rounds = get_b_rounds(spec, emb)
    coeffs = [
        pairwise_round_coefficients(r.edge_polylines, tau=TAU, J0=J0,
                                     kernel=kernel_fn, use_weak_limit=False)
        for r in rounds
    ]
    exposures = [weighted_exposure_on_support(sorted(s), coeffs) for s in family]
    return max(exposures), np.mean(exposures)


def main():
    spec = get_code('BB72')
    kernel_fn = get_kernel('powerlaw', {'alpha': ALPHA, 'r0': 1.0})
    family = enumerate_pure_L_minwt_logicals(spec)
    print(f'BB72 pure-L family: {len(family)} supports')

    # Monomial baseline
    mono = MonomialColumnEmbedding(spec)
    mono_max, mono_mean = compute_family_exposure(spec, mono, kernel_fn, family)
    print(f'Monomial baseline: max_exp={mono_max:.6f}, mean_exp={mono_mean:.6f}')

    # Default biplanar
    bi_default = IBMToricBiplanarEmbedding(spec)
    bi_default_max, bi_default_mean = compute_family_exposure(
        spec, bi_default, kernel_fn, family)
    print(f'Biplanar default (h=1.0): max_exp={bi_default_max:.6f}, mean_exp={bi_default_mean:.6f}')

    # Sweep over h and lane_eps
    rows = []
    default_lane_eps = 1.0 * 1e-3 / max(1, spec.half)

    for h in H_VALUES:
        for le_mult in LANE_EPS_MULTIPLIERS:
            lane_eps = h * 1e-3 / max(1, spec.half) * le_mult
            emb = IBMToricBiplanarEmbedding(spec, h=h, lane_eps=lane_eps)
            max_exp, mean_exp = compute_family_exposure(spec, emb, kernel_fn, family)
            improvement = (1 - max_exp / bi_default_max) * 100
            rows.append({
                'h': h,
                'lane_eps_mult': le_mult,
                'lane_eps': lane_eps,
                'max_exposure': max_exp,
                'mean_exposure': mean_exp,
                'improvement_pct': improvement,
            })
            if abs(le_mult - 1.0) < 0.01:
                print(f'  h={h:.2f}: max_exp={max_exp:.6f} ({improvement:+.1f}%)')

    df = pd.DataFrame(rows)
    best = df.loc[df['max_exposure'].idxmin()]

    outdir = Path('results')
    df.to_csv(outdir / 'biplanar_distance_decay_audit.csv', index=False)

    lines = [
        '# Biplanar Distance-Decay Audit (BB72)',
        '',
        f'Kernel: powerlaw α={ALPHA}, J₀={J0}, τ={TAU}',
        f'Family: {len(family)} pure-L weight-6 X-logical supports',
        '',
        '## Baselines',
        '',
        f'- Monomial max exposure: {mono_max:.6f}',
        f'- Biplanar default (h=1.0) max exposure: {bi_default_max:.6f}',
        f'- Biplanar advantage over monomial: {(1 - bi_default_max/mono_max)*100:.1f}%',
        '',
        '## Best bounded-thickness candidate',
        '',
        f'- h = {best["h"]:.2f}',
        f'- lane_eps_mult = {best["lane_eps_mult"]:.1f}',
        f'- Max exposure: {best["max_exposure"]:.6f}',
        f'- Improvement over default: {best["improvement_pct"]:.1f}%',
        '',
        '## Sweep results (lane_eps_mult=1.0)',
        '',
        '| h | Max exposure | vs default | vs monomial |',
        '|---|-------------|-----------|-------------|',
    ]
    for _, r in df[df['lane_eps_mult'] == 1.0].sort_values('h').iterrows():
        vs_mono = (1 - r['max_exposure'] / mono_max) * 100
        lines.append(
            f'| {r["h"]:.2f} | {r["max_exposure"]:.6f} '
            f'| {r["improvement_pct"]:+.1f}% | {vs_mono:+.1f}% |'
        )

    gain = best['improvement_pct']
    lines.extend([
        '',
        '## Decision',
        '',
    ])
    if gain >= 10:
        lines.append(f'**Gain ≥ 10% ({gain:.1f}%)**: bounded-thickness optimization is worth pursuing.')
    else:
        lines.append(f'**Gain < 10% ({gain:.1f}%)**: bounded-thickness optimization is NOT worth pursuing for this submission.')
        lines.append('The default h=1.0 embedding is sufficient.')

    report = '\n'.join(lines)
    (outdir / 'biplanar_distance_decay_audit.md').write_text(report)
    print(f'\n{report}')


if __name__ == '__main__':
    main()
