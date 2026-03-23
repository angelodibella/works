#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Optimize BB72 logical-aware embedding on the full pure-q(L) family.

Uses the exact ker(B^T) / T_L quotient family (36 weight-6 supports),
multi-restart SA + deterministic swap descent, and exports a frozen
config for publication-facing experiments.

Outputs:
  configs/logical_aware_bb72_truefamily.json
  results/logical_aware_truefamily_optimization.csv
  results/logical_aware_truefamily_optimization.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code, get_embedding
from bbstim.algebra import (
    enumerate_pure_L_minwt_logicals,
    reference_family_hash,
    pure_L_quotient_dimension,
)
from bbstim.embeddings import optimize_row_order, _J_exposure
from bbstim.geometry import (
    count_zero_distance_pairs,
    pairwise_round_coefficients,
    weighted_exposure_on_support,
    regularized_power_law_kernel,
)

# Working point (paper-facing candidate)
J0 = 0.04
TAU = 1.0
ALPHA = 3.0
R0 = 1.0
SEED = 42
# SA parameters: the corrected 36-support family is substantially more
# expensive per iteration than the old 7-support proxy.  The present
# 20k-iteration / 10-restart setting already reproduces the deterministic
# gate-passing solution in about 40-45 minutes on a strong single core.
N_SA_ITER = 20_000
N_RESTARTS = 10


def get_b_rounds(spec, emb, emb_name):
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


def compute_family_stats(spec, emb, emb_name, kernel_fn, family):
    rounds = get_b_rounds(spec, emb, emb_name)
    crossings = sum(count_zero_distance_pairs(r.edge_polylines) for r in rounds)
    coeffs = [
        pairwise_round_coefficients(r.edge_polylines, tau=TAU, J0=J0,
                                     kernel=kernel_fn, use_weak_limit=False)
        for r in rounds
    ]
    exposures = [weighted_exposure_on_support(sorted(s), coeffs) for s in family]
    return crossings, max(exposures), np.mean(exposures), min(exposures)


def main():
    spec = get_code('BB72')
    kernel_fn = regularized_power_law_kernel(ALPHA, R0)

    # Build full family
    family = enumerate_pure_L_minwt_logicals(spec)
    fhash = reference_family_hash(family)
    dk, dt, dq = pure_L_quotient_dimension(spec)
    print(f'BB72 pure-q(L) family: {len(family)} supports')
    print(f'  dim ker(B^T)={dk}, dim T_L={dt}, quotient={dq}')
    print(f'  family hash: {fhash}')

    # Monomial baseline
    mono_cross, mono_max, mono_mean, mono_min = compute_family_stats(
        spec, get_embedding(spec, 'monomial_column'), 'monomial_column', kernel_fn, family)
    print(f'\nMonomial: crossings={mono_cross}, max_exp={mono_max:.6f}, mean={mono_mean:.6f}')

    # Biplanar baseline
    bi_cross, bi_max, bi_mean, bi_min = compute_family_stats(
        spec, get_embedding(spec, 'ibm_biplanar'), 'ibm_biplanar', kernel_fn, family)
    print(f'Biplanar: crossings={bi_cross}, max_exp={bi_max:.6f}, mean={bi_mean:.6f}')

    # Optimize on full family
    print(f'\nOptimizing on full {len(family)}-support family...')
    t0 = time.time()
    sigma_L, sigma_Z, info = optimize_row_order(
        spec,
        objective='exposure',
        reference_family=family,
        n_sa_iter=N_SA_ITER,
        n_restarts=N_RESTARTS,
        seed=SEED,
        tau=TAU, J0=J0, alpha=ALPHA, r0=R0,
    )
    opt_time = time.time() - t0
    print(f'  Done in {opt_time:.1f}s')
    print(f'  initial={info["initial_cost"]:.6f}, sa={info["sa_cost"]:.6f}, final={info["final_cost"]:.6f}')

    # Evaluate LA embedding
    from bbstim.embeddings import FixedPermutationColumnEmbedding
    config = {
        'code': 'BB72',
        'objective': 'exposure',
        'sigma_L': sigma_L,
        'sigma_Z': sigma_Z,
        'x_positions': {'X': 0.0, 'L': 1.0, 'R': 2.0, 'Z': 3.0},
        'scale_y': 1.0,
        'tau': TAU,
        'J0': J0,
        'alpha': ALPHA,
        'r0': R0,
        'seed': SEED,
        'n_sa_iter': N_SA_ITER,
        'n_restarts': N_RESTARTS,
        'family_hash': fhash,
        'family_size': len(family),
        'opt_info': info,
    }
    config_path = Path('configs/logical_aware_bb72_truefamily.json')
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(f'\nFrozen config: {config_path}')

    la_emb = FixedPermutationColumnEmbedding(spec, config_path)
    la_cross, la_max, la_mean, la_min = compute_family_stats(
        spec, la_emb, 'logical_aware_fixed', kernel_fn, family)
    print(f'LA (full family): crossings={la_cross}, max_exp={la_max:.6f}, mean={la_mean:.6f}')

    # Improvement
    improvement = (1 - la_max / mono_max) * 100
    print(f'\nMax exposure improvement over monomial: {improvement:.1f}%')
    print(f'LA vs biplanar: {"below" if la_max < bi_max else "above"} ({la_max:.6f} vs {bi_max:.6f})')

    # Write CSV
    import pandas as pd
    rows = [
        {'embedding': 'monomial', 'crossings': mono_cross, 'max_exp': mono_max,
         'mean_exp': mono_mean, 'min_exp': mono_min},
        {'embedding': 'logical_aware', 'crossings': la_cross, 'max_exp': la_max,
         'mean_exp': la_mean, 'min_exp': la_min},
        {'embedding': 'biplanar', 'crossings': bi_cross, 'max_exp': bi_max,
         'mean_exp': bi_mean, 'min_exp': bi_min},
    ]
    pd.DataFrame(rows).to_csv('results/logical_aware_truefamily_optimization.csv', index=False)

    # Write report
    lines = [
        '# BB72 Logical-Aware Optimization (True Family)',
        '',
        f'Kernel: powerlaw α={ALPHA}, J₀={J0}, τ={TAU}, r₀={R0}',
        f'Family: {len(family)} pure-q(L) weight-6 X-logical supports',
        f'Family hash: {fhash}',
        f'Quotient: dim(ker(B^T))={dk}, dim(T_L)={dt}, dim(quotient)={dq}',
        '',
        '## Optimizer parameters',
        '',
        f'- SA iterations: {N_SA_ITER}',
        f'- SA restarts: {N_RESTARTS}',
        f'- Seed: {SEED}',
        f'- Runtime: {opt_time:.1f}s',
        f'- Family source: {info["reference_family_source"]}',
        '',
        '## Objective values',
        '',
        f'- Initial (monomial): {info["initial_cost"]:.6f}',
        f'- Best SA: {info["sa_cost"]:.6f}',
        f'- Final (swap descent): {info["final_cost"]:.6f}',
        '',
        '## Comparison',
        '',
        '| Metric | Monomial | Logical-aware | Biplanar |',
        '|--------|----------|--------------|----------|',
        f'| Total crossings | {mono_cross} | {la_cross} | {bi_cross} |',
        f'| Max exposure | {mono_max:.6f} | {la_max:.6f} | {bi_max:.6f} |',
        f'| Mean exposure | {mono_mean:.6f} | {la_mean:.6f} | {bi_mean:.6f} |',
        f'| Min exposure | {mono_min:.6f} | {la_min:.6f} | {bi_min:.6f} |',
        '',
        f'## Improvement: {improvement:.1f}% reduction in max exposure vs monomial',
    ]
    Path('results/logical_aware_truefamily_optimization.md').write_text('\n'.join(lines))
    print('\nReports written.')


if __name__ == '__main__':
    main()
