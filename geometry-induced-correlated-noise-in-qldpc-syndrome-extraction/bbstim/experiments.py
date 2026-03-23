# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from dataclasses import dataclass, asdict
from typing import Any, Literal
import datetime
import math
import time

import pandas as pd
from scipy.stats import beta
import sinter
from tqdm.auto import tqdm

from .bbcode import BBCodeSpec, build_bb72, build_bb90, build_bb108, build_bb144
from .embeddings import MonomialColumnEmbedding, IBMToricBiplanarEmbedding, IBMBiplanarSurrogateEmbedding, LogicalAwareColumnEmbedding, FixedPermutationColumnEmbedding
from .geometry import (
    crossing_kernel,
    regularized_power_law_kernel,
    exponential_kernel,
    weighted_exposure_on_support,
    pairwise_round_coefficients,
    pairwise_round_amplitudes,
    pairwise_round_location_strengths,
    aggregate_pair_probability,
    aggregate_pair_amplitude,
    aggregate_location_strength,
)
from .circuit import LocalNoiseConfig, GeometryNoiseConfig, build_bb_memory_experiment, ExperimentKind, decoded_sector
from .decoders import SinterBPOSDDecoder, decode_with_bposd, decode_with_mwpm, collect_pair_edges, build_cbposd_dem, CompiledCBPOSDDecoder
from .gf2 import hamming_weight

# Workbook Chapter 5 worked-example support: X(f,0) with
# f = y^3 + x^2 + x^3y^3 + x^4 + x^4y^3 + x^5y^3.
# This is one of 36 minimum-weight pure-L X-logicals for BB72
# (verified by algebraic enumeration in scripts/bb72_pureL_minwt_logicals.py).
BB72_X_LOGICAL_SUPPORT_L = [3, 12, 21, 24, 27, 33]


def _code_distance(spec: BBCodeSpec) -> int:
    """Minimum Hamming weight of all X and Z logical operators."""
    x_logs, z_logs = spec.logical_bases()
    d = min(
        *(hamming_weight(row) for row in x_logs),
        *(hamming_weight(row) for row in z_logs),
    )
    return d

CodeName = Literal['BB72', 'BB90', 'BB108', 'BB144']
EmbeddingName = Literal['monomial_column', 'ibm_biplanar', 'ibm_biplanar_surrogate']
KernelName = Literal['crossing', 'powerlaw', 'exponential']

DEFAULT_NUM_WORKERS = 12

# OOM-safe worker caps.  BB144 biplanar has ~100k DEM error mechanisms;
# each BPOSD worker holds the full decoder state (~8-15 GB).
# On 249600 MB nodes: 12 workers ≈ 20 GB/worker (safe).
_MAX_WORKERS = {
    ('BB90', 'ibm_biplanar'): 8,
    ('BB108', 'ibm_biplanar'): 10,
    ('BB108', 'monomial_column'): 10,
    ('BB144', 'ibm_biplanar'): 12,
    ('BB144', 'ibm_biplanar_surrogate'): 12,
}


def safe_num_workers(code: str, embedding: str, requested: int) -> int:
    """Clamp worker count to OOM-safe maximum for the given code/embedding."""
    cap = _MAX_WORKERS.get((code, embedding))
    if cap is not None and requested > cap:
        return cap
    return requested


def get_code(name: str) -> BBCodeSpec:
    normalized = name.upper()
    if normalized == 'BB72':
        return build_bb72()
    if normalized == 'BB90':
        return build_bb90()
    if normalized == 'BB108':
        return build_bb108()
    if normalized == 'BB144':
        return build_bb144()
    raise ValueError(f'Unknown code {name!r}. Expected one of: BB72, BB90, BB108, BB144.')


def get_embedding(spec: BBCodeSpec, name: str):
    if name == 'monomial_column':
        return MonomialColumnEmbedding(spec)
    if name == 'ibm_biplanar':
        return IBMToricBiplanarEmbedding(spec)
    if name == 'ibm_biplanar_surrogate':
        return IBMBiplanarSurrogateEmbedding(spec)
    if name == 'logical_aware':
        raise ValueError(
            "The dynamic 'logical_aware' embedding is non-production and should not be used in Monte Carlo runs. "
            "First run scripts/optimize_bb72_logical_aware_truefamily.py and then use "
            "'logical_aware_fixed:configs/logical_aware_bb72_truefamily.json'."
        )
    if name.startswith('logical_aware_fixed:'):
        config_path = name.split(':', 1)[1]
        return FixedPermutationColumnEmbedding(spec, config_path)
    raise ValueError(f'Unknown embedding {name!r}. Expected monomial_column, ibm_biplanar, logical_aware, logical_aware_fixed:<path>, or ibm_biplanar_surrogate.')


def get_kernel(name: str, params: dict[str, float]):
    if name == 'crossing':
        return crossing_kernel
    if name == 'powerlaw':
        return regularized_power_law_kernel(alpha=params['alpha'], r0=params['r0'])
    if name == 'exponential':
        return exponential_kernel(xi=params['xi'])
    raise ValueError(name)


def clopper_pearson(k: int, n: int, alpha: float = 0.3173) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial CI.  Default *alpha* gives a 1-sigma
    (68.27 %) interval; use alpha=0.05 for a 95 % interval."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def derived_per_cycle_rate(p_total: float, cycles: int) -> float:
    return 1 - (1 - p_total) ** (1 / cycles)


def _bb72_exposure(spec, embedding_name, kernel_name, kernel_params, tau, J0, use_weak_limit):
    if spec.name != 'BB72':
        return math.nan, math.nan, math.nan, math.nan
    embedding = get_embedding(spec, embedding_name)
    kernel_fn = get_kernel(kernel_name, kernel_params)
    if embedding_name == 'monomial_column':
        rounds = [
            embedding.routing_geometry(control_reg='L', target_reg='Z', term=spec.B_terms[0], transpose=True, name='B1'),
            embedding.routing_geometry(control_reg='L', target_reg='Z', term=spec.B_terms[1], transpose=True, name='B2'),
            embedding.routing_geometry(control_reg='L', target_reg='Z', term=spec.B_terms[2], transpose=True, name='B3'),
        ]
    else:
        rounds = [
            embedding.routing_geometry(control_reg='L', target_reg='Z', term_name='B1', term=spec.B_terms[0], transpose=True, name='B1'),
            embedding.routing_geometry(control_reg='L', target_reg='Z', term_name='B2', term=spec.B_terms[1], transpose=True, name='B2'),
            embedding.routing_geometry(control_reg='L', target_reg='Z', term_name='B3', term=spec.B_terms[2], transpose=True, name='B3'),
        ]
    coeffs_prob = [
        pairwise_round_coefficients(r.edge_polylines, tau=tau, J0=J0, kernel=kernel_fn, use_weak_limit=use_weak_limit)
        for r in rounds
    ]
    coeffs_amp = [
        pairwise_round_amplitudes(r.edge_polylines, J0=J0, kernel=kernel_fn)
        for r in rounds
    ]
    coeffs_loc = [
        pairwise_round_location_strengths(r.edge_polylines, tau=tau, J0=J0, kernel=kernel_fn)
        for r in rounds
    ]

    exposure = weighted_exposure_on_support(BB72_X_LOGICAL_SUPPORT_L, coeffs_prob)
    agg_prob = max([max(aggregate_pair_probability(c).values()) if c else 0.0 for c in coeffs_prob], default=0.0)
    agg_amp = max([max(aggregate_pair_amplitude(c).values()) if c else 0.0 for c in coeffs_amp], default=0.0)
    agg_loc = max([max(aggregate_location_strength(c).values()) if c else 0.0 for c in coeffs_loc], default=0.0)
    return exposure, agg_prob, agg_amp, agg_loc


def experiment_decoded_sector(exp) -> str:
    """Return the decoded error sector for an experiment."""
    return decoded_sector(exp.experiment_kind)


def reference_exposure_metric(exp, spec):
    # Workbook-aligned metric: BB72, decoded X-error sector, theory-reduced geometry.
    if exp.code != 'BB72' or experiment_decoded_sector(exp) != 'X' or exp.geometry_scope != 'theory_reduced':
        return math.nan, math.nan, math.nan, math.nan
    return _bb72_exposure(spec, exp.embedding, exp.kernel, exp.kernel_params, exp.tau, exp.J0, exp.use_weak_limit)


def _full_cycle_aggregates(spec, embedding_name, kernel_name, kernel_params, tau, J0, use_weak_limit, experiment_kind):
    """Compute aggregate metrics over all sector-relevant rounds (full_cycle scope)."""
    from .circuit import decoded_sector as _ds, ibm_round_specs
    sector = _ds(experiment_kind)
    embedding = get_embedding(spec, embedding_name)
    kernel_fn = get_kernel(kernel_name, kernel_params)

    all_prob = []
    all_amp = []
    all_loc = []

    for round_name, ops, idles in ibm_round_specs(spec):
        for op in ops:
            control_reg, target_reg, term_name, term, transpose = op
            if sector == 'X' and target_reg != 'Z':
                continue
            if sector == 'Z' and control_reg != 'X':
                continue
            try:
                geom = embedding.routing_geometry(
                    control_reg=control_reg, target_reg=target_reg,
                    term_name=term_name, term=term, transpose=transpose,
                    name=f'{control_reg}{target_reg}_{term_name}',
                )
            except TypeError:
                geom = embedding.routing_geometry(
                    control_reg=control_reg, target_reg=target_reg,
                    term=term, transpose=transpose, name=term_name,
                )

            c_prob = pairwise_round_coefficients(
                geom.edge_polylines, tau=tau, J0=J0, kernel=kernel_fn, use_weak_limit=use_weak_limit,
            )
            c_amp = pairwise_round_amplitudes(
                geom.edge_polylines, J0=J0, kernel=kernel_fn,
            )
            c_loc = pairwise_round_location_strengths(
                geom.edge_polylines, tau=tau, J0=J0, kernel=kernel_fn,
            )

            if c_prob:
                all_prob.append(c_prob)
            if c_amp:
                all_amp.append(c_amp)
            if c_loc:
                all_loc.append(c_loc)

    if not all_prob:
        return math.nan, math.nan, math.nan

    full_prob = max(max(aggregate_pair_probability(c).values()) for c in all_prob)
    full_amp  = max(max(aggregate_pair_amplitude(c).values()) for c in all_amp)
    full_loc  = max(max(aggregate_location_strength(c).values()) for c in all_loc)
    return full_prob, full_amp, full_loc


@dataclass(slots=True)
class Experiment:
    experiment_id: str
    code: CodeName
    embedding: EmbeddingName
    experiment_kind: ExperimentKind
    cycles: int
    shots: int
    p_cnot: float
    p_idle: float
    p_prep: float
    p_meas: float
    kernel: KernelName
    kernel_params: dict[str, float]
    J0: float
    tau: float
    use_weak_limit: bool = False
    primary_decoder: str = 'bposd'
    secondary_decoder: str | None = None
    notes: str = ''
    geometry_scope: str = 'theory_reduced'

    def __post_init__(self):
        if self.cycles <= 0:
            raise ValueError('cycles must be positive.')
        if self.shots <= 0:
            raise ValueError('shots must be positive.')
        if self.primary_decoder not in {'bposd', 'cbposd'}:
            raise ValueError("primary_decoder must be 'bposd' or 'cbposd'.")
        if self.secondary_decoder not in {None, 'mwpm', 'cbposd'}:
            raise ValueError("secondary_decoder must be None, 'mwpm', or 'cbposd'.")
        if self.kernel == 'powerlaw':
            if not {'alpha', 'r0'} <= self.kernel_params.keys():
                raise ValueError("powerlaw kernel requires 'alpha' and 'r0'.")
        if self.kernel == 'exponential' and 'xi' not in self.kernel_params:
            raise ValueError("exponential kernel requires 'xi'.")
        for name, value in (
            ('p_cnot', self.p_cnot),
            ('p_idle', self.p_idle),
            ('p_prep', self.p_prep),
            ('p_meas', self.p_meas),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1].')
        if self.J0 < 0 or self.tau <= 0:
            raise ValueError('J0 must be non-negative and tau must be positive.')
        if self.geometry_scope not in {'theory_reduced', 'full_cycle'}:
            raise ValueError("geometry_scope must be 'theory_reduced' or 'full_cycle'.")


def _task_metadata(exp: Experiment) -> dict[str, Any]:
    return {
        'experiment_id': exp.experiment_id,
        'code': exp.code,
        'embedding': exp.embedding,
        'experiment_kind': exp.experiment_kind,
        'cycles': exp.cycles,
        'shots_requested': exp.shots,
        'kernel': exp.kernel,
        'kernel_params': exp.kernel_params,
        'J0': exp.J0,
        'tau': exp.tau,
        'use_weak_limit': exp.use_weak_limit,
    }


def _collect_task_stats(
    task: sinter.Task,
    *,
    include_mwpm: bool,
    num_workers: int,
    show_progress: bool,
    experiment_id: str,
) -> list[sinter.TaskStats]:
    decoders = ['bposd']
    custom_decoders: dict[str, sinter.Decoder] = {'bposd': SinterBPOSDDecoder()}
    if include_mwpm:
        decoders.append('pymatching')

    progress_bar: tqdm | None = None

    def _on_progress(p: sinter.Progress) -> None:
        if progress_bar is None:
            return
        current_total = sum(s.shots for s in p.new_stats)
        if current_total > 0:
            progress_bar.update(current_total)

    if show_progress:
        shots_requested = task.collection_options.max_shots or 0
        total = shots_requested * len(decoders)
        progress_bar = tqdm(
            total=total,
            desc=experiment_id,
            unit='shot',
            leave=False,
        )

    try:
        return sinter.collect(
            num_workers=num_workers,
            tasks=[task],
            decoders=decoders,
            custom_decoders=custom_decoders,
            progress_callback=_on_progress if show_progress else None,
            print_progress=False,
            max_batch_seconds=10,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _run_experiment_single_process(
    exp: Experiment,
    *,
    circuit,
    dem,
    row: dict[str, Any],
    spec: BBCodeSpec | None = None,
    emb=None,
    kernel_fn=None,
) -> dict[str, Any]:
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(exp.shots, separate_observables=True)
    t1 = time.time()
    if not dets.any() and not obs.any():
        row.update({
            'primary_decoder': 'bposd',
            'primary_failures': 0,
            'primary_shots': exp.shots,
            'primary_discards': 0,
            'primary_ler_total': 0.0,
            'primary_ler_total_lo': 0.0,
            'primary_ler_total_hi': clopper_pearson(0, exp.shots)[1],
            'primary_ler_per_cycle_derived': 0.0,
            'primary_decoder_seconds': time.time() - t1,
            'primary_cpu_seconds': time.time() - t1,
        })
        return row
    pred = decode_with_bposd(dem, dets)
    fails = int((pred != obs).any(axis=1).sum())
    p = fails / exp.shots
    lo, hi = clopper_pearson(fails, exp.shots)
    row.update({
        'primary_decoder': 'bposd',
        'primary_failures': fails,
        'primary_shots': exp.shots,
        'primary_discards': 0,
        'primary_ler_total': p,
        'primary_ler_total_lo': lo,
        'primary_ler_total_hi': hi,
        'primary_ler_per_cycle_derived': derived_per_cycle_rate(p, exp.cycles),
        'primary_decoder_seconds': time.time() - t1,
        'primary_cpu_seconds': time.time() - t1,
    })
    if exp.secondary_decoder == 'mwpm':
        t2 = time.time()
        pred2 = decode_with_mwpm(dem, dets)
        fails2 = int((pred2 != obs).any(axis=1).sum())
        p2 = fails2 / exp.shots
        lo2, hi2 = clopper_pearson(fails2, exp.shots)
        row.update({
            'secondary_decoder': 'mwpm',
            'secondary_failures': fails2,
            'secondary_shots': exp.shots,
            'secondary_discards': 0,
            'secondary_ler_total': p2,
            'secondary_ler_total_lo': lo2,
            'secondary_ler_total_hi': hi2,
            'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, exp.cycles),
            'secondary_cpu_seconds': time.time() - t2,
        })
    if exp.secondary_decoder == 'cbposd':
        t2 = time.time()
        sector = decoded_sector(exp.experiment_kind)
        pair_edges = collect_pair_edges(
            spec, emb, sector,
            J0=exp.J0, tau=exp.tau, kernel_fn=kernel_fn,
            use_weak_limit=exp.use_weak_limit,
            geometry_scope=exp.geometry_scope,
        )
        synthetic_dem, F_dense, logical_matrix = build_cbposd_dem(
            spec, sector, pair_edges, u_single=exp.p_cnot,
        )
        cbposd_dec = CompiledCBPOSDDecoder(
            synthetic_dem,
            num_circuit_detectors=dem.num_detectors,
            spec_half=spec.half,
        )
        fails2, shots2 = cbposd_dec.decode_shots(dets, obs)
        p2 = fails2 / shots2
        lo2, hi2 = clopper_pearson(fails2, shots2)
        row.update({
            'secondary_decoder': 'cbposd',
            'secondary_failures': fails2,
            'secondary_shots': shots2,
            'secondary_discards': 0,
            'secondary_ler_total': p2,
            'secondary_ler_total_lo': lo2,
            'secondary_ler_total_hi': hi2,
            'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, exp.cycles),
            'secondary_cpu_seconds': time.time() - t2,
            'secondary_note': f'cbposd: {len(pair_edges)} pair edges',
        })
    return row


def _make_phase_diagram_suite(
    code: CodeName,
    cycles: int,
    shots: int,
    experiment_kind: ExperimentKind = 'z_memory',
    embeddings: tuple[EmbeddingName, ...] = ('monomial_column', 'ibm_biplanar'),
) -> list[Experiment]:
    """Phase diagram: 5x5 grid in (eta0, alpha) for selected embeddings."""
    alphas = [1.5, 2.0, 3.0, 4.0, 5.0]
    eta0s = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3]
    p = 1e-3
    exps = []
    for emb in embeddings:
        emb_tag = 'mono' if emb == 'monomial_column' else 'bi'
        for alpha in alphas:
            for eta0 in eta0s:
                tag = f'{code.lower()}_pd_{emb_tag}_a{alpha}_e{eta0:.0e}'
                exps.append(Experiment(
                    tag, code, emb, experiment_kind, cycles, shots,
                    p, p, p, p,
                    'powerlaw', {'alpha': alpha, 'r0': 1.0},
                    eta0, 1.0, False, 'bposd', None,
                    f'phase diagram {emb_tag} a={alpha} eta0={eta0}',
                ))
    return exps


def _make_per_sweep_suite(code: CodeName, emb: EmbeddingName, cycles: int, shots: int) -> list[Experiment]:
    """PER sweep: 6 physical error rates, local-only + crossing + powerlaw a=3."""
    pers = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    emb_tag = 'mono' if emb == 'monomial_column' else 'bi'
    exps = []
    for p in pers:
        ptag = f'{p:.0e}'
        # Baseline: local noise only
        exps.append(Experiment(
            f'{code.lower()}_per_{emb_tag}_local_p{ptag}', code, emb, 'z_memory', cycles, shots,
            p, p, p, p, 'crossing', {}, 0.0, 1.0, False, 'bposd', None,
            f'PER sweep local p={p}',
        ))
        # Crossing noise
        exps.append(Experiment(
            f'{code.lower()}_per_{emb_tag}_cross_p{ptag}', code, emb, 'z_memory', cycles, shots,
            p, p, p, p, 'crossing', {}, 0.08, 1.0, False, 'bposd', None,
            f'PER sweep crossing p={p}',
        ))
        # Power-law a=3
        exps.append(Experiment(
            f'{code.lower()}_per_{emb_tag}_powa3_p{ptag}', code, emb, 'z_memory', cycles, shots,
            p, p, p, p, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None,
            f'PER sweep powerlaw a=3 p={p}',
        ))
    return exps


SUITES: dict[str, list[Experiment]] = {
    # ── Smoke tests ──
    'bb72_smoke': [
        Experiment('bb72_smoke_mono', 'BB72', 'monomial_column', 'z_memory', 3, 200, 5e-4, 5e-4, 5e-4, 5e-4, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'smoke local only'),
        Experiment('bb72_smoke_bi', 'BB72', 'ibm_biplanar', 'z_memory', 3, 200, 5e-4, 5e-4, 5e-4, 5e-4, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'smoke local only'),
    ],

    # ── BB72 crossing comparison (Fig 5 panel) — both sectors ──
    'bb72_crossing_compare': [
        # z_memory (X-error sector, theory-facing)
        Experiment('bb72_z_crossing_mono', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'z_memory crossing mono'),
        Experiment('bb72_z_crossing_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'z_memory crossing bi'),
        Experiment('bb72_z_baseline_mono', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'z_memory baseline mono'),
        Experiment('bb72_z_baseline_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'z_memory baseline bi'),
        # x_memory (Z-error sector, for X/Z symmetry comparison)
        Experiment('bb72_x_crossing_mono', 'BB72', 'monomial_column', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'x_memory crossing mono'),
        Experiment('bb72_x_crossing_bi', 'BB72', 'ibm_biplanar', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'x_memory crossing bi'),
        Experiment('bb72_x_baseline_mono', 'BB72', 'monomial_column', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'x_memory baseline mono'),
        Experiment('bb72_x_baseline_bi', 'BB72', 'ibm_biplanar', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'x_memory baseline bi'),
    ],

    # ── BB72 distance-decay kernels (Fig 5 panel) — both sectors ──
    'bb72_distance_decay': [
        # z_memory (X-error sector, theory-facing)
        Experiment('bb72_z_pow_mono_a3', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw mono a3'),
        Experiment('bb72_z_pow_bi_a3', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw bi a3'),
        Experiment('bb72_z_pow_mono_a5', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 5.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw mono a5'),
        Experiment('bb72_z_pow_bi_a5', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 5.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw bi a5'),
        Experiment('bb72_z_pow_mono_a1p5', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 1.5, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw mono a1.5'),
        Experiment('bb72_z_pow_bi_a1p5', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 1.5, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw bi a1.5'),
        Experiment('bb72_z_pow_mono_a2', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 2.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw mono a2'),
        Experiment('bb72_z_pow_bi_a2', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 2.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw bi a2'),
        Experiment('bb72_z_pow_mono_a4', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 4.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw mono a4'),
        Experiment('bb72_z_pow_bi_a4', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 4.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory powerlaw bi a4'),
        Experiment('bb72_z_exp_mono', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory exp mono'),
        Experiment('bb72_z_exp_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': 1.0}, 0.08, 1.0, False, 'bposd', None, 'z_memory exp bi'),
        # x_memory (Z-error sector, for X/Z symmetry comparison)
        Experiment('bb72_x_pow_mono_a3', 'BB72', 'monomial_column', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory powerlaw mono a3'),
        Experiment('bb72_x_pow_bi_a3', 'BB72', 'ibm_biplanar', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory powerlaw bi a3'),
        Experiment('bb72_x_pow_mono_a5', 'BB72', 'monomial_column', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 5.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory powerlaw mono a5'),
        Experiment('bb72_x_pow_bi_a5', 'BB72', 'ibm_biplanar', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 5.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory powerlaw bi a5'),
        Experiment('bb72_x_exp_mono', 'BB72', 'monomial_column', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory exp mono'),
        Experiment('bb72_x_exp_bi', 'BB72', 'ibm_biplanar', 'x_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': 1.0}, 0.08, 1.0, False, 'bposd', None, 'x_memory exp bi'),
    ],

    # ── Phase diagram: BB72 5x5 grid in (eta0, alpha) — z_memory / X-error sector (Fig 7) ──
    'bb72_phase_diagram': _make_phase_diagram_suite('BB72', 6, 5000, experiment_kind='z_memory'),

    # ── Phase diagram: BB144 5x5 grid in (eta0, alpha) — z_memory / X-error sector (Fig 7 scaling) ──
    # Split by embedding to fit within 6h SLURM wall-time limits.
    'bb144_phase_diagram': _make_phase_diagram_suite('BB144', 12, 5000, experiment_kind='z_memory'),
    'bb144_phase_diagram_mono': _make_phase_diagram_suite('BB144', 12, 5000, experiment_kind='z_memory', embeddings=('monomial_column',)),
    'bb144_phase_diagram_bi': _make_phase_diagram_suite('BB144', 12, 5000, experiment_kind='z_memory', embeddings=('ibm_biplanar',)),

    # ── PER sweep: BB72 monomial column (embedding comparison) ──
    'bb72_per_sweep_mono': _make_per_sweep_suite('BB72', 'monomial_column', 6, 5000),

    # ── PER sweep: BB72 biplanar (Fig 5 curves) ──
    'bb72_per_sweep_bi': _make_per_sweep_suite('BB72', 'ibm_biplanar', 6, 5000),

    # ── PER sweep: BB144 biplanar (scaling comparison) ──
    'bb144_per_sweep_bi': _make_per_sweep_suite('BB144', 'ibm_biplanar', 12, 5000),

    # ── BB144 benchmark (Fig 5, scaling) ──
    'bb144_scaling': [
        Experiment('bb144_baseline_bi', 'BB144', 'ibm_biplanar', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'bb144 baseline'),
        Experiment('bb144_crossing_bi', 'BB144', 'ibm_biplanar', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'bb144 crossing'),
        Experiment('bb144_pow_bi_a3', 'BB144', 'ibm_biplanar', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'bb144 powerlaw a3'),
        Experiment('bb144_pow_bi_a5', 'BB144', 'ibm_biplanar', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 5.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'bb144 powerlaw a5'),
        Experiment('bb144_exp_bi', 'BB144', 'ibm_biplanar', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': 1.0}, 0.08, 1.0, False, 'bposd', None, 'bb144 exp'),
        # Monomial-column for comparison
        Experiment('bb144_baseline_mono', 'BB144', 'monomial_column', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'bb144 baseline mono'),
        Experiment('bb144_crossing_mono', 'BB144', 'monomial_column', 'z_memory', 12, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'bb144 crossing mono'),
        # bb144_pow_mono_a3 dropped: BP-OSD decode time ~5 min/shot makes it
        # infeasible within the 6h SLURM wall-time limit.  The biplanar vs
        # monomial comparison is already established by the crossing-kernel
        # pair above plus the full BB72 kernel-family data.
    ],

    # ── J0/eta0 sweep at fixed alpha=3 — z_memory / X-error sector (coupling strength) ──
    'bb72_j0_sweep': [
        Experiment(f'bb72_z_j0_{j0}_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, j0, 1.0, False, 'bposd', None, f'z_memory J0 sweep bi J0={j0}')
        for j0 in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.16, 0.32]
    ] + [
        Experiment(f'bb72_z_j0_{j0}_mono', 'BB72', 'monomial_column', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, j0, 1.0, False, 'bposd', None, f'z_memory J0 sweep mono J0={j0}')
        for j0 in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.16, 0.32]
    ],

    # ── J0 sweep with crossing kernel — geometry scale check (Ch.3 theorem) ──
    'bb72_j0_sweep_crossing': [
        Experiment(f'bb72_z_j0cross_{j0}_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, j0, 1.0, False, 'bposd', None, f'z_memory J0 crossing bi J0={j0}')
        for j0 in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.16, 0.32]
    ] + [
        Experiment(f'bb72_z_j0cross_{j0}_mono', 'BB72', 'monomial_column', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, j0, 1.0, False, 'bposd', None, f'z_memory J0 crossing mono J0={j0}')
        for j0 in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.16, 0.32]
    ],

    # ── Exponential kernel xi sweep at fixed J0=0.08 — z_memory ──
    'bb72_xi_sweep': [
        Experiment(f'bb72_z_xi_{xi}_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': xi}, 0.08, 1.0, False, 'bposd', None, f'z_memory xi sweep bi xi={xi}')
        for xi in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    ] + [
        Experiment(f'bb72_z_xi_{xi}_mono', 'BB72', 'monomial_column', 'z_memory', 6, 5000, 1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': xi}, 0.08, 1.0, False, 'bposd', None, f'z_memory xi sweep mono xi={xi}')
        for xi in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    ],

    # ── Decoder control: BP-OSD on local-only and crossing-local ──
    # Note: MWPM (pymatching) is incompatible with BB code DEMs — they produce
    # non-graphlike 3-detector error mechanisms that cannot be decomposed into
    # a matching graph.  BP-OSD is the correct decoder for these quasi-cyclic
    # CSS codes, consistent with the IBM BB paper.
    'bb72_decoder_control': [
        Experiment('bb72_dc_local_mono', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'decoder ctrl local mono'),
        Experiment('bb72_dc_local_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.0, 1.0, False, 'bposd', None, 'decoder ctrl local bi'),
        Experiment('bb72_dc_cross_mono', 'BB72', 'monomial_column', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'decoder ctrl crossing mono'),
        Experiment('bb72_dc_cross_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 4000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'decoder ctrl crossing bi'),
    ],

    # ── Full-sector robustness control: same physical cycle, broader geometry insertion ──
    'bb72_full_cycle_control': [
        Experiment('bb72_fc_cross_mono', 'BB72', 'monomial_column', 'z_memory', 6, 3000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'full-cycle crossing mono', 'full_cycle'),
        Experiment('bb72_fc_cross_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 3000, 1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {}, 0.08, 1.0, False, 'bposd', None, 'full-cycle crossing bi', 'full_cycle'),
        Experiment('bb72_fc_pow3_mono', 'BB72', 'monomial_column', 'z_memory', 6, 3000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'full-cycle powerlaw mono', 'full_cycle'),
        Experiment('bb72_fc_pow3_bi', 'BB72', 'ibm_biplanar', 'z_memory', 6, 3000, 1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0}, 0.08, 1.0, False, 'bposd', None, 'full-cycle powerlaw bi', 'full_cycle'),
    ],

    # ══════════════════════════════════════════════════════════════════════
    #  V3 SUITE: Revised experiment program (GPT Pro advice + workbook Ch.9)
    # ══════════════════════════════════════════════════════════════════════
    #
    #  Common J0 grid (GPT Pro):
    #    J_72 = {0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08}
    #  Common alpha grid: {1.5, 2, 3, 4, 5}
    #  Common xi grid: {0.25, 0.5, 1, 2, 4, 8}
    #  Common PER grid: {3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3}
    #  Standard coupling for alpha/xi/PER sweeps: J0 = 0.04 (not 0.08)
    #  Phase diagram: p = 3e-3 (not 1e-3)

    # ── S1: Crossing-kernel J0 sweep (theorem validation, Ch.3-5) ──
    'bb72_v3_cross_j0': [
        Experiment(f'bb72_v3_cj0_{j0}_{tag}', 'BB72', emb, 'z_memory', 6, 5000,
                   1e-3, 1e-3, 1e-3, 1e-3, 'crossing', {},
                   j0, 1.0, False, 'bposd', None,
                   f'v3 crossing J0={j0} {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for j0 in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
    ],

    # ── S2: Powerlaw J0 sweep (main distance-decay figure) ──
    'bb72_v3_pow_j0': [
        Experiment(f'bb72_v3_pj0_{j0}_{tag}', 'BB72', emb, 'z_memory', 6, 5000,
                   1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0},
                   j0, 1.0, False, 'bposd', None,
                   f'v3 powerlaw a=3 J0={j0} {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for j0 in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
    ],

    # ── S3: Alpha sweep at J0=0.04 (summability/AKP story) ──
    'bb72_v3_alpha': [
        Experiment(f'bb72_v3_a{alpha}_{tag}', 'BB72', emb, 'z_memory', 6, 5000,
                   1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': alpha, 'r0': 1.0},
                   0.04, 1.0, False, 'bposd', None,
                   f'v3 alpha={alpha} J0=0.04 {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for alpha in [1.5, 2.0, 3.0, 4.0, 5.0]
    ],

    # ── S4: Xi sweep at J0=0.04 (exponential kernel) ──
    'bb72_v3_xi': [
        Experiment(f'bb72_v3_xi{xi}_{tag}', 'BB72', emb, 'z_memory', 6, 5000,
                   1e-3, 1e-3, 1e-3, 1e-3, 'exponential', {'xi': xi},
                   0.04, 1.0, False, 'bposd', None,
                   f'v3 xi={xi} J0=0.04 {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for xi in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    ],

    # ── S5: PER sweep at J0=0.04, alpha=3 (transition regime) ──
    'bb72_v3_per': [
        Experiment(f'bb72_v3_per_{p:.0e}_{tag}', 'BB72', emb, 'z_memory', 6, 5000,
                   p, p, p, p, 'powerlaw', {'alpha': 3.0, 'r0': 1.0},
                   0.04, 1.0, False, 'bposd', None,
                   f'v3 PER p={p} J0=0.04 {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for p in [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    ],

    # ── S6: Phase diagram at p=3e-3 (workbook Ch.9 / GPT Pro) ──
    'bb72_v3_phase': [
        Experiment(f'bb72_v3_ph_{tag}_a{alpha}_J{j0}', 'BB72', emb, 'z_memory', 6, 5000,
                   3e-3, 3e-3, 3e-3, 3e-3, 'powerlaw', {'alpha': alpha, 'r0': 1.0},
                   j0, 1.0, False, 'bposd', None,
                   f'v3 phase a={alpha} J0={j0} p=3e-3 {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for alpha in [1.5, 2.0, 3.0, 4.0, 5.0]
        for j0 in [0.005, 0.01, 0.02, 0.04, 0.08]
    ],

    # ── T1: BB144 J0 sweep (scaling check) ──
    'bb144_v3_j0': [
        Experiment(f'bb144_v3_pj0_{j0}_{tag}', 'BB144', emb, 'z_memory', 12, 5000,
                   1e-3, 1e-3, 1e-3, 1e-3, 'powerlaw', {'alpha': 3.0, 'r0': 1.0},
                   j0, 1.0, False, 'bposd', None,
                   f'v3 BB144 powerlaw a=3 J0={j0} {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for j0 in [0.01, 0.02, 0.04, 0.08]
    ],

    # ── T2: BB144 PER sweep at J0=0.04 (scaling check) ──
    'bb144_v3_per': [
        Experiment(f'bb144_v3_per_{p:.0e}_{tag}', 'BB144', emb, 'z_memory', 12, 5000,
                   p, p, p, p, 'powerlaw', {'alpha': 3.0, 'r0': 1.0},
                   0.04, 1.0, False, 'bposd', None,
                   f'v3 BB144 PER p={p} J0=0.04 {tag}')
        for emb, tag in [('monomial_column', 'mono'), ('ibm_biplanar', 'bi')]
        for p in [5e-4, 1e-3, 2e-3, 3e-3]
    ],
}


def run_experiment(
    exp: Experiment,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = False,
) -> dict[str, Any]:
    spec = get_code(exp.code)
    num_workers = safe_num_workers(exp.code, exp.embedding, num_workers)
    emb = get_embedding(spec, exp.embedding)
    local = LocalNoiseConfig(exp.p_cnot, exp.p_idle, exp.p_prep, exp.p_meas)
    geom = GeometryNoiseConfig(
        enabled=(exp.J0 > 0),
        J0=exp.J0,
        tau=exp.tau,
        kernel_name=exp.kernel,
        use_weak_limit=exp.use_weak_limit,
        geometry_scope=exp.geometry_scope,
    )
    kernel_fn = get_kernel(exp.kernel, exp.kernel_params)
    reference_exposure, ref_prob, ref_amp, ref_loc = reference_exposure_metric(exp, spec)
    full_prob, full_amp, full_loc = _full_cycle_aggregates(
        spec, exp.embedding, exp.kernel, exp.kernel_params,
        exp.tau, exp.J0, exp.use_weak_limit, exp.experiment_kind,
    ) if exp.J0 > 0 else (math.nan, math.nan, math.nan)
    t0 = time.time()
    circuit, aux = build_bb_memory_experiment(
        spec,
        embedding=emb,
        rounds=exp.cycles,
        experiment_kind=exp.experiment_kind,
        local_noise=local,
        geometry_noise=geom,
        kernel_fn=kernel_fn,
    )
    build_seconds = time.time() - t0
    dem = circuit.detector_error_model()
    num_pair_channels = sum(
        1 for inst in circuit.flattened() if inst.name == 'E'
    )
    circ_str = str(circuit)
    circuit_depth = circ_str.count('TICK')
    num_dem_errors = dem.num_errors
    emb_params = emb.params_dict() if hasattr(emb, 'params_dict') else {}
    emb_hash = emb.params_hash() if hasattr(emb, 'params_hash') else ''
    row: dict[str, Any] = asdict(exp)
    row.update({
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'embedding_params': str(emb_params) if emb_params else '',
        'embedding_hash': emb_hash,
        'decoded_sector': decoded_sector(exp.experiment_kind),
        'd': _code_distance(spec),
        'n_data': spec.n_data,
        'n_total': spec.n_total,
        'k': aux['x_logicals'].shape[0],
        'build_seconds': build_seconds,
        'num_detectors': dem.num_detectors,
        'num_observables': dem.num_observables,
        'num_dem_errors': num_dem_errors,
        'circuit_depth': circuit_depth,
        'reference_weighted_exposure': reference_exposure,
        'reference_aggregate_pair_probability_max': ref_prob,
        'reference_aggregate_amplitude_max': ref_amp,
        'reference_aggregate_location_strength_max': ref_loc,
        'full_cycle_aggregate_pair_probability_max': full_prob,
        'full_cycle_aggregate_amplitude_max': full_amp,
        'full_cycle_aggregate_location_strength_max': full_loc,
        'num_pair_channels': num_pair_channels,
        'num_workers': num_workers,
    })
    if num_workers <= 1:
        return _run_experiment_single_process(exp, circuit=circuit, dem=dem, row=row, spec=spec, emb=emb, kernel_fn=kernel_fn)

    task = sinter.Task(
        circuit=circuit,
        detector_error_model=dem,
        json_metadata=_task_metadata(exp),
        collection_options=sinter.CollectionOptions(max_shots=exp.shots),
    )
    t1 = time.time()
    try:
        stats_list = _collect_task_stats(
            task,
            include_mwpm=(exp.secondary_decoder == 'mwpm'),
            num_workers=num_workers,
            show_progress=show_progress,
            experiment_id=exp.experiment_id,
        )
    except PermissionError:
        return _run_experiment_single_process(exp, circuit=circuit, dem=dem, row=row)

    stats_by_decoder = {stat.decoder: stat for stat in stats_list}

    primary_stats = stats_by_decoder.get('bposd')
    if primary_stats is not None and primary_stats.shots > 0:
        p = primary_stats.errors / primary_stats.shots
        lo, hi = clopper_pearson(primary_stats.errors, primary_stats.shots)
        row.update({
            'primary_decoder': 'bposd',
            'primary_failures': primary_stats.errors,
            'primary_shots': primary_stats.shots,
            'primary_discards': primary_stats.discards,
            'primary_ler_total': p,
            'primary_ler_total_lo': lo,
            'primary_ler_total_hi': hi,
            'primary_ler_per_cycle_derived': derived_per_cycle_rate(p, exp.cycles),
            'primary_decoder_seconds': time.time() - t1,
            'primary_cpu_seconds': primary_stats.seconds,
        })
    else:
        row.update({
            'primary_decoder': 'bposd',
            'primary_failures': 0,
            'primary_shots': 0,
            'primary_discards': 0,
            'primary_ler_total': math.nan,
            'primary_decoder_seconds': time.time() - t1,
            'primary_cpu_seconds': 0.0,
        })

    if exp.secondary_decoder == 'mwpm':
        secondary_stats = stats_by_decoder.get('pymatching')
        if secondary_stats is not None and secondary_stats.shots > 0:
            p2 = secondary_stats.errors / secondary_stats.shots
            lo2, hi2 = clopper_pearson(secondary_stats.errors, secondary_stats.shots)
            row.update({
                'secondary_decoder': 'mwpm',
                'secondary_failures': secondary_stats.errors,
                'secondary_shots': secondary_stats.shots,
                'secondary_discards': secondary_stats.discards,
                'secondary_ler_total': p2,
                'secondary_ler_total_lo': lo2,
                'secondary_ler_total_hi': hi2,
                'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, exp.cycles),
                'secondary_cpu_seconds': secondary_stats.seconds,
            })

    return row


def _prepare_experiment(exp: Experiment, num_workers: int) -> tuple[dict[str, Any], sinter.Task]:
    """Build circuit, compute metrics, and create a sinter Task for one experiment."""
    spec = get_code(exp.code)
    emb = get_embedding(spec, exp.embedding)
    local = LocalNoiseConfig(exp.p_cnot, exp.p_idle, exp.p_prep, exp.p_meas)
    geom = GeometryNoiseConfig(
        enabled=(exp.J0 > 0),
        J0=exp.J0,
        tau=exp.tau,
        kernel_name=exp.kernel,
        use_weak_limit=exp.use_weak_limit,
        geometry_scope=exp.geometry_scope,
    )
    kernel_fn = get_kernel(exp.kernel, exp.kernel_params)
    reference_exposure, ref_prob, ref_amp, ref_loc = reference_exposure_metric(exp, spec)
    full_prob, full_amp, full_loc = _full_cycle_aggregates(
        spec, exp.embedding, exp.kernel, exp.kernel_params,
        exp.tau, exp.J0, exp.use_weak_limit, exp.experiment_kind,
    ) if exp.J0 > 0 else (math.nan, math.nan, math.nan)
    t0 = time.time()
    circuit, aux = build_bb_memory_experiment(
        spec,
        embedding=emb,
        rounds=exp.cycles,
        experiment_kind=exp.experiment_kind,
        local_noise=local,
        geometry_noise=geom,
        kernel_fn=kernel_fn,
    )
    build_seconds = time.time() - t0
    dem = circuit.detector_error_model()
    num_pair_channels = sum(
        1 for inst in circuit.flattened() if inst.name == 'E'
    )
    circ_str = str(circuit)
    circuit_depth = circ_str.count('TICK')
    num_dem_errors = dem.num_errors
    emb_params = emb.params_dict() if hasattr(emb, 'params_dict') else {}
    emb_hash = emb.params_hash() if hasattr(emb, 'params_hash') else ''
    row: dict[str, Any] = asdict(exp)
    row.update({
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'embedding_params': str(emb_params) if emb_params else '',
        'embedding_hash': emb_hash,
        'decoded_sector': decoded_sector(exp.experiment_kind),
        'd': _code_distance(spec),
        'n_data': spec.n_data,
        'n_total': spec.n_total,
        'k': aux['x_logicals'].shape[0],
        'build_seconds': build_seconds,
        'num_detectors': dem.num_detectors,
        'num_observables': dem.num_observables,
        'num_dem_errors': num_dem_errors,
        'circuit_depth': circuit_depth,
        'reference_weighted_exposure': reference_exposure,
        'reference_aggregate_pair_probability_max': ref_prob,
        'reference_aggregate_amplitude_max': ref_amp,
        'reference_aggregate_location_strength_max': ref_loc,
        'full_cycle_aggregate_pair_probability_max': full_prob,
        'full_cycle_aggregate_amplitude_max': full_amp,
        'full_cycle_aggregate_location_strength_max': full_loc,
        'num_pair_channels': num_pair_channels,
        'num_workers': num_workers,
    })
    task = sinter.Task(
        circuit=circuit,
        detector_error_model=dem,
        json_metadata=_task_metadata(exp),
        collection_options=sinter.CollectionOptions(max_shots=exp.shots),
    )
    return row, task


def _fill_row_from_stats(
    row: dict[str, Any],
    exp: Experiment,
    stats_map: dict[str, sinter.TaskStats],
    wall_seconds: float,
) -> None:
    """Populate a result row from sinter TaskStats."""
    primary = stats_map.get('bposd')
    if primary is not None and primary.shots > 0:
        p = primary.errors / primary.shots
        lo, hi = clopper_pearson(primary.errors, primary.shots)
        row.update({
            'primary_decoder': 'bposd',
            'primary_failures': primary.errors,
            'primary_shots': primary.shots,
            'primary_discards': primary.discards,
            'primary_ler_total': p,
            'primary_ler_total_lo': lo,
            'primary_ler_total_hi': hi,
            'primary_ler_per_cycle_derived': derived_per_cycle_rate(p, exp.cycles),
            'primary_decoder_seconds': wall_seconds,
            'primary_cpu_seconds': primary.seconds,
        })
    else:
        row.update({
            'primary_decoder': 'bposd',
            'primary_failures': 0,
            'primary_shots': 0,
            'primary_discards': 0,
            'primary_ler_total': math.nan,
            'primary_decoder_seconds': wall_seconds,
            'primary_cpu_seconds': 0.0,
        })
    if exp.secondary_decoder == 'mwpm':
        secondary = stats_map.get('pymatching')
        if secondary is not None and secondary.shots > 0:
            p2 = secondary.errors / secondary.shots
            lo2, hi2 = clopper_pearson(secondary.errors, secondary.shots)
            row.update({
                'secondary_decoder': 'mwpm',
                'secondary_failures': secondary.errors,
                'secondary_shots': secondary.shots,
                'secondary_discards': secondary.discards,
                'secondary_ler_total': p2,
                'secondary_ler_total_lo': lo2,
                'secondary_ler_total_hi': hi2,
                'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, exp.cycles),
                'secondary_cpu_seconds': secondary.seconds,
            })


def _batch_collect(
    items: list[tuple[Experiment, dict[str, Any], sinter.Task]],
    decoders: list[str],
    num_workers: int,
    show_progress: bool,
    suite_name: str,
) -> list[dict[str, Any]]:
    """Run a batch of experiments in a single sinter.collect call."""
    tasks = [t for _, _, t in items]
    total_shots = sum(e.shots for e, _, _ in items) * len(decoders)

    progress_bar: tqdm | None = None

    def _on_progress(p: sinter.Progress) -> None:
        if progress_bar is None:
            return
        progress_bar.update(sum(s.shots for s in p.new_stats))

    if show_progress:
        dec_label = '+'.join(decoders)
        progress_bar = tqdm(
            total=total_shots,
            desc=f'{suite_name} [{dec_label}]',
            unit='shot',
            leave=False,
        )

    t1 = time.time()
    try:
        stats_list = sinter.collect(
            num_workers=num_workers,
            tasks=tasks,
            decoders=decoders,
            custom_decoders={'bposd': SinterBPOSDDecoder()},
            progress_callback=_on_progress if show_progress else None,
            print_progress=False,
            max_batch_seconds=10,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    wall_seconds = time.time() - t1

    # Map stats back to experiments by experiment_id
    stats_by_exp: dict[str, dict[str, sinter.TaskStats]] = {}
    for stat in stats_list:
        exp_id = stat.json_metadata['experiment_id']
        stats_by_exp.setdefault(exp_id, {})[stat.decoder] = stat

    rows: list[dict[str, Any]] = []
    for exp, row, task in items:
        exp_stats = stats_by_exp.get(exp.experiment_id, {})
        _fill_row_from_stats(row, exp, exp_stats, wall_seconds)
        rows.append(row)
    return rows


def _parse_slice(s: str, length: int) -> slice:
    """Parse a Python-style slice string like '0:4' or '4:' into a slice object."""
    parts = s.split(':')
    if len(parts) == 1:
        i = int(parts[0])
        return slice(i, i + 1)
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return slice(start, stop, step)


def run_suite(
    suite_name: str,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = False,
    experiment_slice: str | None = None,
    shots_override: int | None = None,
) -> pd.DataFrame:
    suite = SUITES[suite_name]
    if experiment_slice is not None:
        sl = _parse_slice(experiment_slice, len(suite))
        suite = suite[sl]
    if shots_override is not None:
        from copy import copy
        suite = [copy(e) for e in suite]
        for e in suite:
            e.shots = shots_override

    # Phase 1: Build all circuits and prepare sinter Tasks
    prepared: list[tuple[Experiment, dict[str, Any], sinter.Task]] = []
    build_iter = (
        tqdm(suite, desc=f'build:{suite_name}', unit='exp', leave=False)
        if show_progress else suite
    )
    for exp in build_iter:
        row, task = _prepare_experiment(exp, num_workers)
        prepared.append((exp, row, task))

    # Single-process fallback
    if num_workers <= 1:
        rows: list[dict[str, Any]] = []
        for exp, row, task in prepared:
            rows.append(_run_experiment_single_process(
                exp, circuit=task.circuit, dem=task.detector_error_model, row=row,
            ))
        return pd.DataFrame(rows)

    # Phase 2: Batch-collect — separate by decoder config to avoid
    # unnecessary MWPM decoding on experiments that don't need it.
    bposd_batch = [(e, r, t) for e, r, t in prepared if e.secondary_decoder != 'mwpm']
    mwpm_items = [(e, r, t) for e, r, t in prepared if e.secondary_decoder == 'mwpm']

    rows = []

    # Batch-collect all BP-OSD-only experiments in one sinter.collect call.
    if bposd_batch:
        try:
            rows.extend(_batch_collect(bposd_batch, ['bposd'], num_workers, show_progress, suite_name))
        except PermissionError:
            for exp, row, task in bposd_batch:
                rows.append(_run_experiment_single_process(
                    exp, circuit=task.circuit, dem=task.detector_error_model, row=row,
                ))

    # MWPM experiments: BP-OSD is batched, MWPM attempted separately per
    # experiment because BB code DEMs are non-graphlike and pymatching may
    # fail on individual circuits.
    if mwpm_items:
        mwpm_bposd = [(e, r, t) for e, r, t in mwpm_items]
        try:
            bposd_rows = _batch_collect(mwpm_bposd, ['bposd'], num_workers, show_progress, suite_name)
        except PermissionError:
            bposd_rows = []
            for exp, row, task in mwpm_bposd:
                bposd_rows.append(_run_experiment_single_process(
                    exp, circuit=task.circuit, dem=task.detector_error_model, row=row,
                ))
        # Attempt MWPM decoding per experiment (best-effort)
        for (exp, _, task), row in zip(mwpm_items, bposd_rows):
            try:
                decomposed_dem = task.circuit.detector_error_model(
                    decompose_errors=True, ignore_decomposition_failures=True,
                )
                sampler = task.circuit.compile_detector_sampler()
                dets, obs = sampler.sample(exp.shots, separate_observables=True)
                pred = decode_with_mwpm(decomposed_dem, dets)
                fails = int((pred != obs).any(axis=1).sum())
                p2 = fails / exp.shots
                lo2, hi2 = clopper_pearson(fails, exp.shots)
                row.update({
                    'secondary_decoder': 'mwpm',
                    'secondary_failures': fails,
                    'secondary_shots': exp.shots,
                    'secondary_discards': 0,
                    'secondary_ler_total': p2,
                    'secondary_ler_total_lo': lo2,
                    'secondary_ler_total_hi': hi2,
                    'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, exp.cycles),
                    'secondary_cpu_seconds': 0.0,
                })
            except Exception:
                row.update({
                    'secondary_decoder': 'mwpm',
                    'secondary_failures': 0,
                    'secondary_shots': 0,
                    'secondary_ler_total': math.nan,
                    'secondary_note': 'pymatching failed: non-graphlike DEM',
                })
        rows.extend(bposd_rows)
    return pd.DataFrame(rows)
