#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Deterministic geometry audit (G0, G1, G2).

Outputs:
  results/geometry_audit.csv  — per-embedding aggregate metrics
  results/geometry_audit.md   — human-readable report
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.geometry import (
    aggregate_pair_amplitude,
    aggregate_pair_probability,
    aggregate_location_strength,
    count_zero_distance_pairs,
    pairwise_round_amplitudes,
    pairwise_round_coefficients,
    pairwise_round_location_strengths,
    weighted_exposure_on_support,
)
from bbstim.experiments import BB72_X_LOGICAL_SUPPORT_L, get_code, get_embedding, get_kernel

# ── Configuration ──
CODES = ['BB72', 'BB144']
EMBEDDINGS = ['monomial_column', 'ibm_biplanar']
KERNELS = [
    ('crossing', {}, 'crossing'),
    ('powerlaw', {'alpha': 3.0, 'r0': 1.0}, 'powerlaw_a3'),
    ('powerlaw', {'alpha': 1.5, 'r0': 1.0}, 'powerlaw_a1.5'),
]
J0 = 0.04
TAU = 1.0


def _get_b_rounds(spec, embedding, emb_name):
    """Build the 3 theory-facing B-round routing geometries."""
    rounds = []
    for i, (term, tname) in enumerate(zip(spec.B_terms, ['B1', 'B2', 'B3'])):
        try:
            g = embedding.routing_geometry(
                control_reg='L', target_reg='Z',
                term_name=tname, term=term, transpose=True, name=tname,
            )
        except TypeError:
            g = embedding.routing_geometry(
                control_reg='L', target_reg='Z',
                term=term, transpose=True, name=tname,
            )
        rounds.append(g)
    return rounds


def audit_one(code_name, emb_name, kernel_name, kernel_params, kernel_label):
    spec = get_code(code_name)
    emb = get_embedding(spec, emb_name)
    kernel_fn = get_kernel(kernel_name, kernel_params)
    rounds = _get_b_rounds(spec, emb, emb_name)

    # Per-round crossing counts
    total_crossings = 0
    for r in rounds:
        total_crossings += count_zero_distance_pairs(r.edge_polylines)

    # Pairwise coefficients
    coeffs_prob = [
        pairwise_round_coefficients(r.edge_polylines, tau=TAU, J0=J0,
                                     kernel=kernel_fn, use_weak_limit=False)
        for r in rounds
    ]
    coeffs_amp = [
        pairwise_round_amplitudes(r.edge_polylines, J0=J0, kernel=kernel_fn)
        for r in rounds
    ]
    coeffs_loc = [
        pairwise_round_location_strengths(r.edge_polylines, tau=TAU, J0=J0,
                                           kernel=kernel_fn)
        for r in rounds
    ]

    # Aggregate metrics
    agg_prob = max([max(aggregate_pair_probability(c).values()) if c else 0.0
                    for c in coeffs_prob], default=0.0)
    agg_amp = max([max(aggregate_pair_amplitude(c).values()) if c else 0.0
                   for c in coeffs_amp], default=0.0)
    agg_loc = max([max(aggregate_location_strength(c).values()) if c else 0.0
                   for c in coeffs_loc], default=0.0)

    # Pair channel count
    n_pairs = sum(len(c) for c in coeffs_prob)

    # Weighted exposure on support (BB72 only)
    if code_name == 'BB72':
        exposure = weighted_exposure_on_support(BB72_X_LOGICAL_SUPPORT_L, coeffs_prob)
        # Support-induced crossing pairs
        S = set(BB72_X_LOGICAL_SUPPORT_L)
        support_crossings = 0
        for r in rounds:
            items = list(r.edge_polylines.items())
            from itertools import combinations
            for (e1, p1), (e2, p2) in combinations(items, 2):
                if e1[0] != 'L' or e2[0] != 'L':
                    continue
                if e1[1] not in S or e2[1] not in S:
                    continue
                verts = {(e1[0], e1[1]), (e1[2], e1[3])} & {(e2[0], e2[1]), (e2[2], e2[3])}
                if verts:
                    continue
                from bbstim.geometry import polyline_distance
                if polyline_distance(p1, p2) < 1e-12:
                    support_crossings += 1
        # Matching number: for crossing kernel on monomial, workbook proves ν=3
        # For biplanar crossing, ν=0
        if kernel_name == 'crossing' and emb_name == 'monomial_column':
            matching_number = 3
        elif kernel_name == 'crossing' and 'biplanar' in emb_name:
            matching_number = 0
        else:
            matching_number = None  # not defined for distance-decay
    else:
        exposure = math.nan
        support_crossings = None
        matching_number = None

    return {
        'code': code_name,
        'embedding': emb_name,
        'kernel': kernel_label,
        'J0': J0,
        'total_crossings': total_crossings,
        'support_crossings': support_crossings,
        'matching_number': matching_number,
        'weighted_exposure': exposure,
        'agg_pair_prob_max': agg_prob,
        'agg_amplitude_max': agg_amp,
        'agg_location_strength_max': agg_loc,
        'num_pair_channels': n_pairs,
    }


def exposure_vs_ler(results_csv):
    """G2: Spearman correlation between weighted exposure and LER."""
    df = pd.read_csv(results_csv)
    sub = df.dropna(subset=['reference_weighted_exposure', 'primary_ler_total'])
    sub = sub[(sub['reference_weighted_exposure'] > 0) & (sub['primary_failures'] >= 10)]
    if len(sub) < 5:
        return None, None, 0
    rho, pval = spearmanr(sub['reference_weighted_exposure'], sub['primary_ler_total'])
    return rho, pval, len(sub)


def main():
    outdir = Path('results')
    outdir.mkdir(exist_ok=True)

    # G0 + G1: deterministic audit
    rows = []
    for code in CODES:
        for emb in EMBEDDINGS:
            for kname, kparams, klabel in KERNELS:
                print(f'  Auditing {code} / {emb} / {klabel}...', flush=True)
                row = audit_one(code, emb, kname, kparams, klabel)
                rows.append(row)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(outdir / 'geometry_audit.csv', index=False)

    # G2: exposure vs LER
    results_csv = outdir / 'results.csv'
    rho, pval, n_pts = exposure_vs_ler(results_csv)

    # Write markdown report
    lines = ['# Geometry Audit Report', '']
    lines.append(f'Generated from `scripts/geometry_audit.py` at J₀ = {J0}, τ = {TAU}.')
    lines.append('')

    for code in CODES:
        lines.append(f'## {code}')
        lines.append('')
        cdf = audit_df[audit_df['code'] == code]

        # Crossing kernel table
        ck = cdf[cdf['kernel'] == 'crossing']
        if not ck.empty:
            lines.append('### Crossing kernel')
            lines.append('')
            lines.append('| Embedding | Total crossings | Support crossings | Matching ν | Exposure | Pair channels |')
            lines.append('|-----------|----------------|-------------------|-----------|----------|---------------|')
            for _, r in ck.iterrows():
                emb_short = 'mono' if 'mono' in r['embedding'] else 'bi'
                sc = r['support_crossings'] if r['support_crossings'] is not None else '—'
                mn = r['matching_number'] if r['matching_number'] is not None else '—'
                exp = f'{r["weighted_exposure"]:.6f}' if not math.isnan(r['weighted_exposure']) else '—'
                lines.append(f'| {emb_short} | {r["total_crossings"]} | {sc} | {mn} | {exp} | {r["num_pair_channels"]} |')
            lines.append('')

        # Distance-decay table
        dd = cdf[cdf['kernel'] != 'crossing']
        if not dd.empty:
            lines.append('### Distance-decay kernels')
            lines.append('')
            lines.append('| Embedding | Kernel | Exposure | Agg prob | Agg amp | Agg loc | Pair ch. |')
            lines.append('|-----------|--------|----------|----------|---------|---------|----------|')
            for _, r in dd.iterrows():
                emb_short = 'mono' if 'mono' in r['embedding'] else 'bi'
                exp = f'{r["weighted_exposure"]:.6f}' if not math.isnan(r['weighted_exposure']) else '—'
                lines.append(
                    f'| {emb_short} | {r["kernel"]} | {exp} '
                    f'| {r["agg_pair_prob_max"]:.6f} | {r["agg_amplitude_max"]:.6f} '
                    f'| {r["agg_location_strength_max"]:.6f} | {r["num_pair_channels"]} |'
                )
            lines.append('')

    # G2 summary
    lines.append('## G2: Exposure vs LER correlation')
    lines.append('')
    if rho is not None:
        lines.append(f'- Spearman ρ = {rho:.3f} (p = {pval:.2e}, n = {n_pts} points)')
        lines.append(f'- Points with ≥10 failures and exposure > 0')
    else:
        lines.append('- Insufficient data for correlation analysis.')
    lines.append('')

    # d_eff summary for BB72
    lines.append('## Effective distance summary (BB72, crossing kernel)')
    lines.append('')
    lines.append('| Embedding | d | ν | d_eff ≤ d − ν |')
    lines.append('|-----------|---|---|---------------|')
    lines.append('| mono | 6 | 3 | 3 |')
    lines.append('| bi | 6 | 0 | 6 |')
    lines.append('')
    lines.append('(Theorem 3.1 / Corollary 5.5 of workbook)')

    report = '\n'.join(lines)
    (outdir / 'geometry_audit.md').write_text(report)
    print(report)
    print(f'\nWritten: {outdir / "geometry_audit.csv"}, {outdir / "geometry_audit.md"}')


if __name__ == '__main__':
    main()
