# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from dataclasses import dataclass
from typing import Literal
from .bbcode import BBCodeSpec
from .gf2 import row_basis_mod2
from .geometry import Kernel, pairwise_round_coefficients

try:
    import stim  # type: ignore
except ImportError:
    stim = None  # type: ignore

ExperimentKind = Literal['x_memory', 'z_memory']
DecodedSector = Literal['X', 'Z']


def decoded_sector(experiment_kind: ExperimentKind) -> DecodedSector:
    """z_memory decodes X-errors (via q(Z) syndrome); x_memory decodes Z-errors."""
    return 'Z' if experiment_kind == 'x_memory' else 'X'


def data_measure_basis(experiment_kind: ExperimentKind) -> Literal['X', 'Z']:
    """Final data readout basis for the logical memory experiment."""
    return 'X' if experiment_kind == 'x_memory' else 'Z'


def check_register_for_sector(sector: DecodedSector) -> Literal['X', 'Z']:
    """X-errors are detected through q(Z); Z-errors through q(X)."""
    return 'Z' if sector == 'X' else 'X'

@dataclass
class LocalNoiseConfig:
    p_cnot: float = 0.0
    p_idle: float = 0.0
    p_prep: float = 0.0
    p_meas: float = 0.0

GeometryScope = Literal['theory_reduced', 'full_cycle']


@dataclass
class GeometryNoiseConfig:
    enabled: bool = False
    J0: float = 0.0
    tau: float = 1.0
    kernel_name: str = 'crossing'
    use_weak_limit: bool = False
    geometry_scope: GeometryScope = 'full_cycle'

class RegisterMap:
    def __init__(self, half: int):
        self.half = half
        self.offsets = {'L': 0, 'R': half, 'X': 2 * half, 'Z': 3 * half}

    def q(self, reg: str, i: int) -> int:
        return self.offsets[reg] + i

    @property
    def data_qubits(self) -> list[int]:
        return list(range(0, 2 * self.half))

    @property
    def x_checks(self) -> list[int]:
        return list(range(2 * self.half, 3 * self.half))

    @property
    def z_checks(self) -> list[int]:
        return list(range(3 * self.half, 4 * self.half))


def require_stim():
    if stim is None:
        raise RuntimeError('Stim is not installed; run `pip install -e .`')


def pauli_string_from_binary(vec, basis: str):
    require_stim()
    return stim.PauliString(''.join(basis if int(b) else '_' for b in vec))


def prepare_logical_memory_circuit(spec: BBCodeSpec, experiment_kind: ExperimentKind):
    require_stim()
    hx_basis = row_basis_mod2(spec.hx())
    hz_basis = row_basis_mod2(spec.hz())
    x_logs, z_logs = spec.logical_bases()
    stabilizers = [pauli_string_from_binary(row, 'X') for row in hx_basis]
    stabilizers += [pauli_string_from_binary(row, 'Z') for row in hz_basis]
    basis = data_measure_basis(experiment_kind)
    if basis == 'Z':
        stabilizers += [pauli_string_from_binary(row, 'Z') for row in z_logs]
    else:
        stabilizers += [pauli_string_from_binary(row, 'X') for row in x_logs]
    tableau = stim.Tableau.from_stabilizers(stabilizers, allow_redundant=False, allow_underconstrained=False)
    return tableau.to_circuit(method='elimination'), x_logs, z_logs


def _append_reset_basis(c, qubits, basis, p_prep):
    if not qubits:
        return
    if basis == 'Z':
        c.append('R', qubits)
        if p_prep > 0:
            c.append('X_ERROR', qubits, p_prep)
    elif basis == 'X':
        c.append('RX', qubits)
        if p_prep > 0:
            c.append('Z_ERROR', qubits, p_prep)
    else:
        raise ValueError(basis)


def _append_measure_basis(c, qubits, basis, p_meas):
    if p_meas > 0:
        c.append('X_ERROR' if basis == 'Z' else 'Z_ERROR', qubits, p_meas)
    c.append('M' if basis == 'Z' else 'MX', qubits)


def _append_idle_noise(c, qubits, p_idle):
    if qubits and p_idle > 0:
        c.append('DEPOLARIZE1', qubits, p_idle)


def _append_cnot_layer(c, pairs, p_cnot):
    if not pairs:
        return
    flat = [q for pair in pairs for q in pair]
    c.append('CX', flat)
    if p_cnot > 0:
        for a, b in pairs:
            c.append('DEPOLARIZE2', [a, b], p_cnot)


def _append_correlated_sector_noise(c, coeffs, regmap: RegisterMap, pauli: str, control_reg_filter=None, target_reg_filter=None):
    require_stim()
    for (e1, e2), p in coeffs.items():
        if p <= 0:
            continue
        if control_reg_filter is not None:
            if not (e1[0] == control_reg_filter and e2[0] == control_reg_filter):
                continue
            q1 = regmap.q(e1[0], e1[1])
            q2 = regmap.q(e2[0], e2[1])
        elif target_reg_filter is not None:
            if not (e1[2] == target_reg_filter and e2[2] == target_reg_filter):
                continue
            q1 = regmap.q(e1[2], e1[3])
            q2 = regmap.q(e2[2], e2[3])
        else:
            raise ValueError('Need control_reg_filter or target_reg_filter.')
        t1 = stim.target_x(q1) if pauli == 'X' else stim.target_z(q1)
        t2 = stim.target_x(q2) if pauli == 'X' else stim.target_z(q2)
        c.append('CORRELATED_ERROR', [t1, t2], p)


def ibm_round_specs(spec: BBCodeSpec):
    A1, A2, A3 = spec.A_terms
    B1, B2, B3 = spec.B_terms
    return [
        ('R1', [('R', 'Z', 'A1', A1, True)], {'L': True}),
        ('R2', [('X', 'L', 'A2', A2, False), ('R', 'Z', 'A3', A3, True)], {}),
        ('R3', [('X', 'R', 'B2', B2, False), ('L', 'Z', 'B1', B1, True)], {}),
        ('R4', [('X', 'R', 'B1', B1, False), ('L', 'Z', 'B2', B2, True)], {}),
        ('R5', [('X', 'R', 'B3', B3, False), ('L', 'Z', 'B3', B3, True)], {}),
        ('R6', [('X', 'L', 'A1', A1, False), ('R', 'Z', 'A2', A2, True)], {}),
        ('R7', [('X', 'L', 'A3', A3, False)], {'R': True}),
        ('R8', [], {'L': True, 'R': True}),
    ]


def _round_pairs(spec, regmap, op):
    control_reg, target_reg, _term_name, term, transpose = op
    pairs = []
    for i in range(spec.half):
        tgt = spec.mapped_target_index(i, term, transpose, target_reg)
        pairs.append((regmap.q(control_reg, i), regmap.q(target_reg, tgt)))
    return pairs


def _round_geom(embedding, spec, op, geom_cfg, kernel_fn):
    control_reg, target_reg, term_name, term, transpose = op
    geom = embedding.routing_geometry(
        control_reg=control_reg,
        target_reg=target_reg,
        term_name=term_name,
        term=term,
        transpose=transpose,
        name=f'{control_reg}{target_reg}_{term_name}',
    ) if hasattr(embedding, 'routing_geometry') else None
    if geom is None:
        return {}
    return pairwise_round_coefficients(
        geom.edge_polylines,
        tau=geom_cfg.tau,
        J0=geom_cfg.J0,
        kernel=kernel_fn,
        use_weak_limit=geom_cfg.use_weak_limit,
    )


def _is_geometry_relevant(sector: DecodedSector, op, scope: GeometryScope) -> bool:
    """Decide whether geometry-induced pair channels should be inserted for this op."""
    control_reg, target_reg, term_name, _, _ = op
    if sector == 'X':
        # z_memory: geometry on data→q(Z) ops (target_reg='Z')
        if target_reg != 'Z':
            return False
        if scope == 'theory_reduced':
            return term_name in ('B1', 'B2', 'B3')
        return True
    else:
        # x_memory: geometry on q(X)→data ops (control_reg='X')
        if control_reg != 'X':
            return False
        if scope == 'theory_reduced':
            return term_name in ('B1', 'B2', 'B3')
        return True


def _append_cycle_detectors(circ, current_meas, previous_meas):
    for curr, prev in zip(current_meas, previous_meas or []):
        circ.append('DETECTOR', [stim.target_rec(curr), stim.target_rec(prev)])
    if previous_meas is None:
        for curr in current_meas:
            circ.append('DETECTOR', [stim.target_rec(curr)])


def build_bb_memory_experiment(
    spec: BBCodeSpec,
    *,
    embedding,
    rounds: int,
    experiment_kind: ExperimentKind,
    local_noise: LocalNoiseConfig,
    geometry_noise: GeometryNoiseConfig,
    kernel_fn: Kernel,
):
    """Build a full IBM depth-8 syndrome-extraction circuit.

    All 8 rounds of the IBM syndrome cycle are executed every cycle.
    Ancilla operations follow the IBM schedule:
      - Before first cycle: InitZ q(Z)
      - R1: InitX q(X), then CX(R→Z, A1^T)
      - R2-R6: CNOT layers
      - R7: CX(X→L, A3), then MeasZ q(Z)
      - R8: MeasX q(X), then InitZ q(Z) (for next cycle)

    Local noise (CNOT depolarization, idle noise, prep/meas errors) is
    applied on the full physical cycle. Geometry-induced pair channels
    are inserted on sector-relevant ops according to geometry_scope.

    Detectors and observables are sector-specific:
      - z_memory: detectors from q(Z) measurement stream
      - x_memory: detectors from q(X) measurement stream
    """
    require_stim()
    sector = decoded_sector(experiment_kind)
    basis = data_measure_basis(experiment_kind)
    reg = RegisterMap(spec.half)
    circ = stim.Circuit()
    prep, x_logs, z_logs = prepare_logical_memory_circuit(spec, experiment_kind)
    circ += prep

    # Initial q(Z) reset before first cycle (IBM paper convention)
    _append_reset_basis(circ, reg.z_checks, 'Z', local_noise.p_prep)

    mcount = 0
    last_sector_meas = None
    relevant_check_rows = spec.check_supports_z() if sector == 'X' else spec.check_supports_x()
    logical_basis = z_logs if basis == 'Z' else x_logs
    all_qubits = set(range(4 * spec.half))
    round_specs = ibm_round_specs(spec)

    for cycle in range(rounds):
        z_meas_indices = None
        x_meas_indices = None

        for round_name, ops, idles in round_specs:
            circ.append('TICK')
            active_qubits: set[int] = set()

            # Pre-round ancilla actions
            if round_name == 'R1':
                _append_reset_basis(circ, reg.x_checks, 'X', local_noise.p_prep)
                active_qubits.update(reg.x_checks)
            if round_name == 'R8':
                _append_measure_basis(circ, reg.x_checks, 'X', local_noise.p_meas)
                x_meas_indices = list(range(mcount, mcount + spec.half))
                mcount += spec.half
                active_qubits.update(reg.x_checks)

            # Execute ALL CNOT layers (full physical cycle)
            for op in ops:
                pairs = _round_pairs(spec, reg, op)
                _append_cnot_layer(circ, pairs, local_noise.p_cnot)
                for a, b in pairs:
                    active_qubits.add(a)
                    active_qubits.add(b)
                # Geometry noise on sector-relevant ops
                if geometry_noise.enabled and _is_geometry_relevant(sector, op, geometry_noise.geometry_scope):
                    coeffs = _round_geom(embedding, spec, op, geometry_noise, kernel_fn)
                    if sector == 'X':
                        _append_correlated_sector_noise(circ, coeffs, reg, pauli='X', control_reg_filter=op[0])
                    else:
                        _append_correlated_sector_noise(circ, coeffs, reg, pauli='Z', target_reg_filter=op[1])

            # Post-round ancilla actions
            if round_name == 'R7':
                _append_measure_basis(circ, reg.z_checks, 'Z', local_noise.p_meas)
                z_meas_indices = list(range(mcount, mcount + spec.half))
                mcount += spec.half
                active_qubits.update(reg.z_checks)
            if round_name == 'R8':
                if cycle < rounds - 1:
                    _append_reset_basis(circ, reg.z_checks, 'Z', local_noise.p_prep)
                active_qubits.update(reg.z_checks)

            # Idle noise on non-active qubits
            idle_qubits = sorted(all_qubits - active_qubits)
            _append_idle_noise(circ, idle_qubits, local_noise.p_idle)

        # Build detectors for the decoded sector
        curr = z_meas_indices if sector == 'X' else x_meas_indices
        current_rec_offsets = [idx - mcount for idx in curr]
        previous_rec_offsets = None
        if last_sector_meas is not None:
            previous_rec_offsets = [idx - mcount for idx in last_sector_meas]
        _append_cycle_detectors(circ, current_rec_offsets, previous_rec_offsets)
        last_sector_meas = curr

    # Final data measurement
    circ.append('TICK')
    _append_measure_basis(circ, reg.data_qubits, basis, local_noise.p_meas)
    data_meas = list(range(mcount, mcount + spec.n_data))
    mcount += spec.n_data

    if last_sector_meas is None:
        raise RuntimeError('No relevant sector measurements were recorded.')
    for i, support in enumerate(relevant_check_rows):
        targets = [stim.target_rec(last_sector_meas[i] - mcount)]
        for q in support:
            targets.append(stim.target_rec(data_meas[q] - mcount))
        circ.append('DETECTOR', targets)
    for obs_index, logical in enumerate(logical_basis):
        targets = [stim.target_rec(data_meas[q] - mcount) for q, b in enumerate(logical) if b]
        circ.append('OBSERVABLE_INCLUDE', targets, obs_index)
    return circ, {'x_logicals': x_logs, 'z_logicals': z_logs, 'relevant_check_rows': relevant_check_rows}
