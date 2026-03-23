#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Deterministic geometry audit for BB72, BB90, BB108, BB144.

For each code and embedding (monomial, biplanar), computes:
  - code parameters (n, k, d, half)
  - crossing counts on theory-facing B-rounds
  - aggregate pair-probability and location-strength maxima
  - pure-q(L) family size and max/mean exposure (where feasible)

Outputs:
  results/intermediate_bb_geometry_audit.csv
  results/intermediate_bb_geometry_audit.md
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code, get_embedding, get_kernel
from bbstim.algebra import enumerate_pure_L_minwt_logicals, pure_L_quotient_dimension
from bbstim.geometry import (
    count_zero_distance_pairs,
    pairwise_round_coefficients,
    aggregate_pair_probability,
    aggregate_location_strength,
    weighted_exposure_on_support,
)

J0 = 0.04
TAU = 1.0
ALPHA = 3.0

CODES = ['BB72', 'BB90', 'BB108', 'BB144']
EMBEDDINGS = ['monomial_column', 'ibm_biplanar']


def get_b_rounds(spec, emb):
    rounds = []
    for term, tname in zip(spec.B_terms, ['B1', 'B2', 'B3']):
        try:
            g = emb.routing_geometry(control_reg='L', target_reg='Z',
                                      term_name=tname, term=term, transpose=True, name=tname)
        except TypeError:
            g = emb.routing_geometry(control_reg='L', target_reg='Z',
                                      term=term, transpose=True, name=tname)
        rounds.append(g)
    return rounds


def audit_one(code_name, emb_name, kernel_fn):
    spec = get_code(code_name)
    emb = get_embedding(spec, emb_name)
    x_logs, z_logs = spec.logical_bases()
    k = int(getattr(spec, 'known_k', None) or x_logs.shape[0])
    d = int(getattr(spec, 'known_d', None) or min(
        *(int(r.sum()) for r in x_logs),
        *(int(r.sum()) for r in z_logs),
    ))

    rounds = get_b_rounds(spec, emb)
    crossings = sum(count_zero_distance_pairs(r.edge_polylines) for r in rounds)

    coeffs = [
        pairwise_round_coefficients(r.edge_polylines, tau=TAU, J0=J0,
                                     kernel=kernel_fn, use_weak_limit=False)
        for r in rounds
    ]
    agg_prob = max([max(aggregate_pair_probability(c).values()) if c else 0.0 for c in coeffs], default=0.0)
    agg_loc = max([max(aggregate_location_strength(c).values()) if c else 0.0 for c in coeffs], default=0.0)

    # Pure-q(L) family and exposure (skip for large codes if too slow)
    dk, dt, dq = pure_L_quotient_dimension(spec)
    family_size = None
    pureL_min_weight = None
    max_exp = None
    mean_exp = None
    if 2**dk <= 8192:  # feasible enumeration
        family = enumerate_pure_L_minwt_logicals(spec)
        family_size = len(family)
        pureL_min_weight = len(next(iter(family))) if family else None
        exposures = [weighted_exposure_on_support(sorted(s), coeffs) for s in family]
        max_exp = max(exposures) if exposures else None
        mean_exp = float(np.mean(exposures)) if exposures else None

    return {
        'code': code_name,
        'embedding': emb_name,
        'n': spec.n_data,
        'k': k,
        'd': d,
        'half': spec.half,
        'dim_ker_BT': dk,
        'dim_TL': dt,
        'quotient_dim': dq,
        'crossings': crossings,
        'agg_prob_max': agg_prob,
        'agg_loc_max': agg_loc,
        'family_size': family_size,
        'pureL_min_weight': pureL_min_weight,
        'max_exposure': max_exp,
        'mean_exposure': mean_exp,
    }


def main():
    kernel_fn = get_kernel('powerlaw', {'alpha': ALPHA, 'r0': 1.0})
    rows = []
    for code in CODES:
        for emb in EMBEDDINGS:
            print(f'  {code} / {emb}...', flush=True)
            t0 = time.time()
            row = audit_one(code, emb, kernel_fn)
            row['time_s'] = time.time() - t0
            rows.append(row)
            print(f'    crossings={row["crossings"]}, agg_prob={row["agg_prob_max"]:.4f}, '
                  f'family={row["family_size"]}, max_exp={row["max_exposure"]}'
                  f' ({row["time_s"]:.1f}s)')

    df = pd.DataFrame(rows)
    outdir = Path('results')
    df.to_csv(outdir / 'intermediate_bb_geometry_audit.csv', index=False)

    # Markdown
    lines = [
        '# Intermediate BB Geometry Audit',
        '',
        f'Kernel: powerlaw α={ALPHA}, J₀={J0}, τ={TAU}',
        '',
        '## Code parameters',
        '',
        '| Code | n | k | d | pure-L min wt | half |',
        '|------|---|---|---|----------------|------|',
    ]
    for code in CODES:
        r = [x for x in rows if x['code'] == code][0]
        pure_w = r['pureL_min_weight'] if r['pureL_min_weight'] is not None else '—'
        lines.append(f'| {code} | {r["n"]} | {r["k"]} | {r["d"]} | {pure_w} | {r["half"]} |')

    lines.extend(['', '## Geometry metrics', '',
                  '| Code | Embedding | Crossings | Agg prob | Agg loc | Family | Max exp | Mean exp |',
                  '|------|-----------|-----------|----------|---------|--------|---------|----------|'])
    for r in rows:
        fam = r['family_size'] if r['family_size'] is not None else '—'
        mxe = f'{r["max_exposure"]:.6f}' if r['max_exposure'] is not None else '—'
        mne = f'{r["mean_exposure"]:.6f}' if r['mean_exposure'] is not None else '—'
        emb_short = 'mono' if 'mono' in r['embedding'] else 'bi'
        lines.append(
            f'| {r["code"]} | {emb_short} | {r["crossings"]} | {r["agg_prob_max"]:.4f} '
            f'| {r["agg_loc_max"]:.4f} | {fam} | {mxe} | {mne} |'
        )

    lines.extend(['', '## Pure-q(L) quotient dimensions', '',
                  '| Code | dim ker(B^T) | dim T_L | Quotient |',
                  '|------|-------------|---------|----------|'])
    seen = set()
    for r in rows:
        if r['code'] not in seen:
            seen.add(r['code'])
            lines.append(f'| {r["code"]} | {r["dim_ker_BT"]} | {r["dim_TL"]} | {r["quotient_dim"]} |')

    lines.extend(['', '## Note on code distance vs pure-q(L) family', '',
                  'For BB108, the algebraically enumerated pure-q(L) minimum-weight family has weight 12, while the full code distance of the canonical BB108 benchmark is 10 in the BB literature. These are different notions: the theorem-facing pure-q(L) family is a restricted sector and need not attain the full code distance.'])
    (outdir / 'intermediate_bb_geometry_audit.md').write_text('\n'.join(lines))
    print('\nReports written.')


if __name__ == '__main__':
    main()
