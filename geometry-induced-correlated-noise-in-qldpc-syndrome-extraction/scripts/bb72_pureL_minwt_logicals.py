#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Enumerate ALL pure-q(L) minimum-weight X logicals for BB72.

Uses the exact quotient ker(B^T) / T_L where
  T_L = { λA : λB = 0 }
is the L-projection of stabilizers with vanishing R-component.

Outputs:
  results/bb72_pureL_minwt_logicals.csv
  results/bb72_pureL_minwt_logicals.md
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code
from bbstim.algebra import (
    enumerate_pure_L_minwt_logicals,
    pure_L_quotient_dimension,
    reference_family_hash,
)


def main():
    spec = get_code('BB72')
    half = spec.half

    dk, dt, dq = pure_L_quotient_dimension(spec)
    print(f'BB72: half={half}')
    print(f'  dim ker(B^T) = {dk}')
    print(f'  dim T_L = {dt}')
    print(f'  pure-q(L) quotient dimension = {dq}')

    family = enumerate_pure_L_minwt_logicals(spec)
    min_wt = len(next(iter(family))) if family else 0
    fhash = reference_family_hash(family)

    print(f'  Minimum weight: d_X = {min_wt}')
    print(f'  Total unique weight-{min_wt} supports: {len(family)}')
    print(f'  Family hash: {fhash}')

    workbook_support = {3, 12, 21, 24, 27, 33}
    wb_in_family = workbook_support in family
    print(f'  Workbook support {{3,12,21,24,27,33}} in family: {wb_in_family}')

    # Build CSV
    rows = []
    for i, support in enumerate(family):
        s = sorted(support)
        monomials = []
        for idx in s:
            a, b = spec.ab(idx)
            parts = []
            if a > 0:
                parts.append(f'x^{a}' if a > 1 else 'x')
            if b > 0:
                parts.append(f'y^{b}' if b > 1 else 'y')
            monomials.append(''.join(parts) if parts else '1')
        rows.append({
            'index': i,
            'support_indices': str(s),
            'support_monomials': ', '.join(monomials),
        })

    df = pd.DataFrame(rows)
    outdir = Path('results')
    df.to_csv(outdir / 'bb72_pureL_minwt_logicals.csv', index=False)

    lines = [
        '# BB72 Pure-q(L) Minimum-Weight X-Logical Family',
        '',
        'Exact quotient: ker(B^T) / T_L where T_L = { λA : λB = 0 }.',
        '',
        f'- Code: [[72,12,6]], half={half}',
        f'- dim ker(B^T) = {dk}',
        f'- dim T_L = {dt}',
        f'- Pure-q(L) quotient dimension = {dq}',
        f'- Minimum weight: d_X = {min_wt}',
        f'- Total unique weight-{min_wt} supports: **{len(family)}**',
        f'- Family hash: {fhash}',
        f'- Workbook support {{3,12,21,24,27,33}} in family: **{"Yes" if wb_in_family else "No"}**',
        '',
        '## All supports',
        '',
        '| # | Indices | Monomials |',
        '|---|---------|-----------|',
    ]
    for _, r in df.iterrows():
        lines.append(f'| {r["index"]} | {r["support_indices"]} | {r["support_monomials"]} |')

    (outdir / 'bb72_pureL_minwt_logicals.md').write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
