#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Build results_semantic.csv by merging semantically identical rows.

Sums failures, shots, discards; recomputes CIs from totals.

Outputs:
  results/results_semantic.csv
  results/results_semantic.md
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def clopper_pearson(k, n, alpha=0.3173):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(_beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def derived_per_cycle_rate(p_total, cycles):
    if p_total <= 0 or p_total >= 1 or cycles <= 0:
        return p_total
    return 1 - (1 - p_total) ** (1.0 / cycles)


def main():
    df = pd.read_csv('results/results.csv')

    def parse_alpha(kp):
        try:
            return ast.literal_eval(kp).get('alpha')
        except:
            return None

    def parse_xi(kp):
        try:
            return ast.literal_eval(kp).get('xi')
        except:
            return None

    df['_alpha'] = df['kernel_params'].apply(parse_alpha)
    df['_xi'] = df['kernel_params'].apply(parse_xi)

    # Semantic key: the physical parameters that define a unique experiment
    sem_cols = ['code', 'embedding', 'experiment_kind', 'geometry_scope',
                'kernel', '_alpha', '_xi', 'J0', 'p_cnot', 'cycles']

    # Group and aggregate
    rows = []
    merge_log = []
    for key, grp in df.groupby([c for c in sem_cols if c in df.columns], dropna=False):
        key_dict = dict(zip(sem_cols, key))
        ids = sorted(grp['experiment_id'].tolist())

        total_fail = int(grp['primary_failures'].sum())
        total_shots = int(grp['primary_shots'].sum())
        total_disc = int(grp['primary_discards'].sum()) if 'primary_discards' in grp else 0

        p = total_fail / total_shots if total_shots > 0 else 0.0
        lo, hi = clopper_pearson(total_fail, total_shots)
        cycles = int(grp['cycles'].iloc[0])
        p_cyc = derived_per_cycle_rate(p, cycles)

        # Take metadata from the row with most shots
        best = grp.sort_values('primary_shots', ascending=False).iloc[0]

        row = {
            'semantic_id': f"{key_dict['code']}_{key_dict['embedding'][:4]}_{key_dict['kernel'][:4]}_J{key_dict['J0']}_p{key_dict['p_cnot']}",
            'merged_ids': '|'.join(ids),
            'n_merged': len(ids),
        }
        for c in ['code', 'embedding', 'experiment_kind', 'geometry_scope',
                   'kernel', 'kernel_params', 'J0', 'p_cnot', 'cycles',
                   'decoded_sector', 'd', 'n_data', 'k',
                   'reference_weighted_exposure']:
            if c in best.index:
                row[c] = best[c]

        row.update({
            'primary_failures': total_fail,
            'primary_shots': total_shots,
            'primary_discards': total_disc,
            'primary_ler_total': p,
            'primary_ler_total_lo': lo,
            'primary_ler_total_hi': hi,
            'primary_ler_per_cycle_derived': p_cyc,
        })
        rows.append(row)

        if len(ids) > 1:
            merge_log.append((row['semantic_id'], ids, total_shots, total_fail))

    sem_df = pd.DataFrame(rows)
    sem_df.to_csv('results/results_semantic.csv', index=False)

    # Write markdown
    lines = [
        '# Semantic Aggregation Report',
        '',
        f'Raw rows: {len(df)} → Semantic points: {len(sem_df)}',
        f'Points with merged rows: {len(merge_log)}',
        '',
        '## Merged points',
        '',
    ]
    if merge_log:
        lines.append('| Semantic ID | Merged experiment IDs | Total shots | Total failures |')
        lines.append('|------------|----------------------|-------------|----------------|')
        for sid, ids, shots, fail in sorted(merge_log):
            lines.append(f'| {sid} | {", ".join(ids)} | {shots} | {fail} |')
    else:
        lines.append('No semantic duplicates found.')

    lines.extend([
        '',
        '## Per-code summary',
        '',
        '| Code | Semantic points | Main-text | Appendix |',
        '|------|----------------|-----------|----------|',
    ])
    for code in ['BB72', 'BB90', 'BB108', 'BB144']:
        n = len(sem_df[sem_df['code'] == code])
        mt = 'Yes' if code in ('BB72', 'BB144') else 'No'
        ap = 'Yes' if code in ('BB90', 'BB108') else '—'
        lines.append(f'| {code} | {n} | {mt} | {ap} |')

    Path('results/results_semantic.md').write_text('\n'.join(lines))
    print(f'Written: results/results_semantic.csv ({len(sem_df)} rows)')
    print(f'Written: results/results_semantic.md')


if __name__ == '__main__':
    main()
