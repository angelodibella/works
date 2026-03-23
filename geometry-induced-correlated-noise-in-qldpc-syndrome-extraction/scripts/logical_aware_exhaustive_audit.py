#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Exhaustive logical-aware audit using ALL pure-L weight-6 BB72 X-logical supports.

Compares monomial, logical-aware, and ibm_biplanar on:
  - max weighted exposure over full family
  - mean weighted exposure over full family
  - total crossings

Outputs:
  results/logical_aware_exhaustive_audit.csv
  results/logical_aware_exhaustive_audit.md
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code, get_embedding, get_kernel
from bbstim.geometry import (
    count_zero_distance_pairs,
    pairwise_round_coefficients,
    weighted_exposure_on_support,
)
from bbstim.embeddings import _reference_family

J0 = 0.04
TAU = 1.0
ALPHA = 3.0

def load_exhaustive_family():
    """Load the full algebraically-enumerated pure-L family."""
    from bbstim.algebra import enumerate_pure_L_minwt_logicals
    spec = get_code('BB72')
    family = enumerate_pure_L_minwt_logicals(spec)
    workbook = {3, 12, 21, 24, 27, 33}
    wb_in_family = workbook in family
    return [sorted(s) for s in family], wb_in_family


def get_b_rounds(spec, emb, emb_name):
    rounds = []
    for term, tname in zip(spec.B_terms, ['B1', 'B2', 'B3']):
        try:
            g = emb.routing_geometry(
                control_reg='L', target_reg='Z',
                term_name=tname, term=term, transpose=True, name=tname,
            )
        except TypeError:
            g = emb.routing_geometry(
                control_reg='L', target_reg='Z',
                term=term, transpose=True, name=tname,
            )
        rounds.append(g)
    return rounds


def audit_embedding(spec, emb, emb_name, kernel_fn, family):
    rounds = get_b_rounds(spec, emb, emb_name)
    crossings = sum(count_zero_distance_pairs(r.edge_polylines) for r in rounds)

    coeffs = [
        pairwise_round_coefficients(r.edge_polylines, tau=TAU, J0=J0,
                                     kernel=kernel_fn, use_weak_limit=False)
        for r in rounds
    ]

    exposures = []
    for support in family:
        exp = weighted_exposure_on_support(support, coeffs)
        exposures.append(exp)

    return {
        'embedding': emb_name,
        'total_crossings': crossings,
        'max_exposure': max(exposures),
        'mean_exposure': np.mean(exposures),
        'min_exposure': min(exposures),
        'n_supports': len(family),
        'exposures': exposures,
    }


def main():
    spec = get_code('BB72')
    kernel_fn = get_kernel('powerlaw', {'alpha': ALPHA, 'r0': 1.0})

    print('Loading exhaustive family...')
    family, wb_in_family = load_exhaustive_family()
    print(f'  {len(family)} supports (workbook support in family: {wb_in_family})')

    results = []
    for emb_name in ['monomial_column', 'logical_aware', 'ibm_biplanar']:
        print(f'Auditing {emb_name}...')
        t0 = time.time()
        emb = get_embedding(spec, emb_name)
        dt = time.time() - t0
        r = audit_embedding(spec, emb, emb_name, kernel_fn, family)
        r['build_time'] = dt
        results.append(r)
        print(f'  crossings={r["total_crossings"]}, max_exp={r["max_exposure"]:.6f}, '
              f'mean_exp={r["mean_exposure"]:.6f}, time={dt:.1f}s')

    # Build comparison DataFrame
    rows = []
    for r in results:
        rows.append({
            'embedding': r['embedding'],
            'total_crossings': r['total_crossings'],
            'max_exposure': r['max_exposure'],
            'mean_exposure': r['mean_exposure'],
            'min_exposure': r['min_exposure'],
            'n_supports': r['n_supports'],
            'build_time_s': r['build_time'],
        })
    df = pd.DataFrame(rows)
    outdir = Path('results')
    df.to_csv(outdir / 'logical_aware_exhaustive_audit.csv', index=False)

    mono = results[0]
    la = results[1]
    bi = results[2]

    la_beats_mono = la['max_exposure'] < mono['max_exposure']
    la_vs_bi = ('below' if la['max_exposure'] < bi['max_exposure']
                else 'comparable' if abs(la['max_exposure'] - bi['max_exposure']) / bi['max_exposure'] < 0.05
                else 'above')

    lines = [
        '# Exhaustive Logical-Aware Audit',
        '',
        f'BB72, z_memory, powerlaw α={ALPHA}, J₀={J0}, τ={TAU}',
        f'Full family: {len(family)} weight-6 X-logical supports '
        f'(algebraic pure-L enumeration)',
        '',
        '## Comparison Table',
        '',
        '| Metric | Monomial | Logical-aware | Biplanar |',
        '|--------|----------|--------------|----------|',
        f'| Total crossings | {mono["total_crossings"]} | {la["total_crossings"]} | {bi["total_crossings"]} |',
        f'| Max exposure | {mono["max_exposure"]:.6f} | {la["max_exposure"]:.6f} | {bi["max_exposure"]:.6f} |',
        f'| Mean exposure | {mono["mean_exposure"]:.6f} | {la["mean_exposure"]:.6f} | {bi["mean_exposure"]:.6f} |',
        f'| Min exposure | {mono["min_exposure"]:.6f} | {la["min_exposure"]:.6f} | {bi["min_exposure"]:.6f} |',
        f'| Build time | <0.1s | {la["build_time"]:.1f}s | <0.1s |',
        '',
        '## Gate Decision',
        '',
        f'**LA max exposure < mono max exposure: {"PASS" if la_beats_mono else "FAIL"}**',
        '',
    ]

    if la_beats_mono:
        reduction = (1 - la['max_exposure'] / mono['max_exposure']) * 100
        lines.append(f'Reduction: {reduction:.1f}% on max-exposure objective over full family.')
    else:
        lines.append(f'LA max exposure ({la["max_exposure"]:.6f}) >= mono ({mono["max_exposure"]:.6f}).')
        lines.append('Logical-aware novelty should NOT be promoted to main text.')

    lines.extend([
        '',
        f'**LA vs biplanar**: LA is **{la_vs_bi}** biplanar on max-exposure.',
        f'  LA max = {la["max_exposure"]:.6f}, bi max = {bi["max_exposure"]:.6f}',
        '',
        '## Per-Support Exposure (worst 5 for each embedding)',
        '',
    ])

    # Worst 5 per embedding
    for r in results:
        exp_sorted = sorted(enumerate(r['exposures']), key=lambda x: -x[1])[:5]
        lines.append(f'### {r["embedding"]}')
        for idx, val in exp_sorted:
            lines.append(f'  Support {idx} ({family[idx]}): {val:.6f}')
        lines.append('')

    report = '\n'.join(lines)
    (outdir / 'logical_aware_exhaustive_audit.md').write_text(report)
    print('\n' + report)
    print(f'Written: {outdir / "logical_aware_exhaustive_audit.csv"}')
    print(f'Written: {outdir / "logical_aware_exhaustive_audit.md"}')


if __name__ == '__main__':
    main()
