# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim

if TYPE_CHECKING:
    from .bbcode import BBCodeSpec
    from .geometry import Kernel

try:
    from stimbposd import BPOSD  # type: ignore
except ImportError:
    BPOSD = None  # type: ignore

try:
    import pymatching  # type: ignore
except ImportError:
    pymatching = None  # type: ignore

try:
    import sinter  # type: ignore
except ImportError:
    sinter = None  # type: ignore


class DecoderError(RuntimeError):
    pass


def decode_with_bposd(
    dem,
    det_samples: np.ndarray,
    *,
    max_bp_iters: int = 200,
    bp_method: str = 'ps',
    osd_order: int = 10,
) -> np.ndarray:
    if BPOSD is None:
        raise DecoderError('stimbposd is not installed.')
    dec = BPOSD(dem, max_bp_iters=max_bp_iters, bp_method=bp_method, osd_order=osd_order)
    return np.asarray(dec.decode_batch(det_samples), dtype=np.uint8)


def decode_with_mwpm(dem, det_samples: np.ndarray) -> np.ndarray:
    if pymatching is None:
        raise DecoderError('PyMatching is not installed.')
    try:
        matching = pymatching.Matching.from_detector_error_model(dem)
    except Exception as ex:
        raise DecoderError(f'PyMatching could not load detector error model: {ex}') from ex
    if hasattr(matching, 'decode_batch'):
        return np.asarray(matching.decode_batch(det_samples), dtype=np.uint8)
    return np.asarray([matching.decode(sample) for sample in det_samples], dtype=np.uint8)


def require_sinter():
    if sinter is None:
        raise DecoderError('sinter is not installed.')


_CompiledDecoderBase = sinter.CompiledDecoder if sinter is not None else object
_DecoderBase = sinter.Decoder if sinter is not None else object


class CompiledBPOSDDecoder(_CompiledDecoderBase):
    def __init__(self, dem, *, max_bp_iters: int, bp_method: str, osd_order: int):
        self.dem = dem
        self.num_dets = dem.num_detectors
        self.num_obs = dem.num_observables
        self.trivial = self.num_dets == 0 or self.num_obs == 0 or dem.num_errors == 0
        self.decoder = None if self.trivial else BPOSD(
            dem,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_order=osd_order,
        )

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]
        num_obs_bytes = (self.num_obs + 7) // 8
        if self.trivial:
            return np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)
        dets = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self.num_dets,
            bitorder='little',
        )
        predictions = np.asarray(self.decoder.decode_batch(dets), dtype=np.uint8)
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        return np.packbits(predictions, axis=1, bitorder='little')


class SinterBPOSDDecoder(_DecoderBase):
    def __init__(self, *, max_bp_iters: int = 200, bp_method: str = 'ps', osd_order: int = 10):
        require_sinter()
        if BPOSD is None:
            raise DecoderError('stimbposd is not installed.')
        self.max_bp_iters = max_bp_iters
        self.bp_method = bp_method
        self.osd_order = osd_order

    def compile_decoder_for_dem(self, *, dem):
        return CompiledBPOSDDecoder(
            dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            osd_order=self.osd_order,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Correlation-Aware BP+OSD  (Chapter 7 of the workbook)
# ═══════════════════════════════════════════════════════════════════════
#
#  Implements the exact augmented fault-location decoder of Theorem 7.3
#  and Definition 7.5.  Operates at code-capacity level on the augmented
#  check matrix  H̃ = H_σ [I_n | B]  where B encodes pair-fault locations
#  from the sector-resolved correlation graph.
#
#  A synthetic stim DEM is built from H̃ so that the existing BPOSD
#  engine can decode it.  The circuit's final-round detectors provide
#  the code-capacity syndrome.


def collect_pair_edges(
    spec: BBCodeSpec,
    embedding,
    sector: str,
    *,
    J0: float,
    tau: float,
    kernel_fn: Kernel,
    use_weak_limit: bool = False,
    geometry_scope: str = 'theory_reduced',
    epsilon: float = 1e-6,
) -> list[tuple[int, int, float]]:
    """Gather sector-resolved pair-fault edges and probabilities.

    Reuses the same geometry pipeline as the circuit builder to ensure
    physical consistency.  Returns (i, j, v_a) triples where i, j are
    data-qubit indices into the n_data-length space and v_a is the
    aggregated pair-fault probability across all relevant rounds.

    Truncation (Definition 7.8): edges with v_a < epsilon are dropped.
    """
    from .circuit import ibm_round_specs, _is_geometry_relevant, _round_geom
    from .circuit import GeometryNoiseConfig

    geom_cfg = GeometryNoiseConfig(
        enabled=True, J0=J0, tau=tau,
        kernel_name='', use_weak_limit=use_weak_limit,
        geometry_scope=geometry_scope,
    )

    # Accumulate per-pair probabilities across rounds: v = 1 - ∏(1 - p_r)
    pair_accum: dict[tuple[int, int], float] = {}

    for _round_name, ops, _meas in ibm_round_specs(spec):
        for op in ops:
            if not _is_geometry_relevant(sector, op, geometry_scope):
                continue
            coeffs = _round_geom(embedding, spec, op, geom_cfg, kernel_fn)
            control_reg = op[0]
            for (e1, e2), prob in coeffs.items():
                if prob <= 0:
                    continue
                # Map edge keys to data-qubit indices.
                # X-sector: pair faults on control register (L), indices 0..half-1.
                # Z-sector: pair faults on target register (R), indices half..2*half-1.
                if sector == 'X':
                    i, j = e1[1], e2[1]  # L-register indices
                else:
                    i, j = spec.half + e1[3], spec.half + e2[3]  # R-register indices
                key = (min(i, j), max(i, j))
                # Independent union bound: 1 - (1 - v_old)(1 - p_r)
                v_old = pair_accum.get(key, 0.0)
                pair_accum[key] = 1.0 - (1.0 - v_old) * (1.0 - prob)

    # Truncate (Definition 7.8)
    return [(i, j, v) for (i, j), v in sorted(pair_accum.items()) if v >= epsilon]


def build_cbposd_dem(
    spec: BBCodeSpec,
    sector: str,
    pair_edges: list[tuple[int, int, float]],
    u_single: float | np.ndarray,
) -> tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray]:
    """Build a synthetic DEM from the augmented fault-location matrix.

    Implements Theorem 7.3 / Definition 7.5 of the workbook:
        H̃ = H_σ [I_n | B],  F = [I_n | B]
    where B has columns e_{i_a} + e_{j_a} for each pair edge.

    Args:
        spec: BB code specification.
        sector: 'X' (for z_memory) or 'Z' (for x_memory).
        pair_edges: List of (i, j, v_a) from collect_pair_edges().
        u_single: Per-qubit single-fault prior (scalar or length-n array).

    Returns:
        (synthetic_dem, F_dense, logical_matrix) where:
        - synthetic_dem: stim DEM with spec.half detectors and n+m error mechanisms.
        - F_dense: (n, n+m) uint8 fault-location matrix for projection.
        - logical_matrix: (k, n) uint8 logical operator matrix.
    """
    # ── H_sigma: parity-check matrix for the decoded sector ──
    # z_memory decodes X-errors via Z-checks; x_memory decodes Z-errors via X-checks.
    H_sigma = spec.hz() if sector == 'X' else spec.hx()
    r, n = H_sigma.shape
    assert n == spec.n_data, f"H_sigma has {n} columns, expected {spec.n_data}"
    assert r == spec.half, f"H_sigma has {r} rows, expected {spec.half}"

    m = len(pair_edges)

    # ── Fault-location matrix F = [I_n | B] ──
    F = np.zeros((n, n + m), dtype=np.uint8)
    F[:n, :n] = np.eye(n, dtype=np.uint8)
    for a, (i, j, _v) in enumerate(pair_edges):
        F[i, n + a] = 1
        F[j, n + a] = 1

    # ── Augmented check matrix H̃ = H_sigma @ F (mod 2) ──
    H_tilde = (H_sigma @ F) % 2

    # ── Logical operator matrix ──
    x_logs, z_logs = spec.logical_bases()
    # z_memory (sector X, basis Z) → z_logs detect X-error logical flips
    # x_memory (sector Z, basis X) → x_logs detect Z-error logical flips
    logical_matrix = z_logs if sector == 'X' else x_logs
    k = logical_matrix.shape[0]

    # Observable assignment through F: for column j, observable k flips iff
    # (L_k)^T f_j ≡ 1 (mod 2).  This is (logical_matrix @ F)[:,j] mod 2.
    obs_through_F = (logical_matrix @ F) % 2  # shape (k, n+m)

    # ── Prior probabilities ──
    if isinstance(u_single, (int, float)):
        u_arr = np.full(n, float(u_single), dtype=np.float64)
    else:
        u_arr = np.asarray(u_single, dtype=np.float64)
    v_arr = np.array([v for _, _, v in pair_edges], dtype=np.float64)
    # Clamp to (0, 0.5) to avoid degenerate LLRs
    priors = np.concatenate([u_arr, v_arr])
    priors = np.clip(priors, 1e-12, 0.5 - 1e-12)

    # ── Build synthetic stim DEM ──
    dem = stim.DetectorErrorModel()
    for col in range(n + m):
        p = float(priors[col])
        targets: list[stim.DemTarget] = []
        # Detector targets: nonzero rows of H̃ column
        for row in range(r):
            if H_tilde[row, col]:
                targets.append(stim.target_relative_detector_id(row))
        # Observable targets: which logicals this fault flips
        for obs_idx in range(k):
            if obs_through_F[obs_idx, col]:
                targets.append(stim.target_logical_observable_id(obs_idx))
        if targets:  # skip trivial (no-detector, no-observable) mechanisms
            dem.append('error', p, targets)

    return dem, F.astype(np.uint8), logical_matrix.astype(np.uint8)


class CompiledCBPOSDDecoder:
    """Pair-aware BP+OSD decoder on the augmented fault-location matrix.

    Implements Definition 7.5 of the workbook.  Extracts the final-round
    code-capacity syndrome from the circuit's detection events, decodes
    on the synthetic DEM built from H̃, and projects the decoded
    fault-location vector back to a data-level error estimate via F.

    This is a code-capacity decoder: it uses only the final-round syndrome,
    not the full multi-round detection event history.  It serves as a
    secondary decoder for comparing pair-fault awareness against the
    circuit-level bposd decoder.
    """

    def __init__(
        self,
        synthetic_dem: stim.DetectorErrorModel,
        *,
        num_circuit_detectors: int,
        spec_half: int,
        max_bp_iters: int = 200,
        bp_method: str = 'ps',
        osd_order: int = 10,
    ):
        if BPOSD is None:
            raise DecoderError('stimbposd is not installed.')
        self.num_circuit_detectors = num_circuit_detectors
        self.spec_half = spec_half
        # Final-round detectors are the last spec_half entries (circuit.py:337-341)
        self.final_det_start = num_circuit_detectors - spec_half
        self.bposd = BPOSD(
            synthetic_dem,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_order=osd_order,
        )

    def decode_shots(
        self,
        det_events: np.ndarray,
        obs_actual: np.ndarray,
    ) -> tuple[int, int]:
        """Decode all shots and return (failures, total_shots).

        Args:
            det_events: (num_shots, num_circuit_detectors) uint8 binary.
            obs_actual: (num_shots, num_observables) uint8 binary.

        Returns:
            (num_failures, num_shots).
        """
        num_shots = det_events.shape[0]
        # Extract final-round syndrome (code-capacity syndrome)
        syndrome = det_events[:, self.final_det_start:]
        assert syndrome.shape[1] == self.spec_half, (
            f"Expected {self.spec_half} final-round detectors, got {syndrome.shape[1]}"
        )
        # Decode with BPOSD on synthetic DEM
        # BPOSD.decode_batch returns observable predictions directly
        obs_pred = np.asarray(self.bposd.decode_batch(syndrome), dtype=np.uint8)
        if obs_pred.ndim == 1:
            obs_pred = obs_pred[:, np.newaxis]
        # Count failures: any observable mismatch in a shot is a failure
        failures = int((obs_pred != obs_actual).any(axis=1).sum())
        return failures, num_shots
