# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from bbstim.bbcode import build_bb72
from bbstim.circuit import (
    LocalNoiseConfig, GeometryNoiseConfig, build_bb_memory_experiment,
    decoded_sector, data_measure_basis,
)
from bbstim.embeddings import IBMToricBiplanarEmbedding, IBMBiplanarSurrogateEmbedding, MonomialColumnEmbedding
from bbstim.experiments import get_embedding
from bbstim.geometry import crossing_kernel, regularized_power_law_kernel, count_zero_distance_pairs


def _assert_noiseless_detectors_are_zero(experiment_kind: str) -> None:
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    circuit, _ = build_bb_memory_experiment(
        spec,
        embedding=embedding,
        rounds=1,
        experiment_kind=experiment_kind,
        local_noise=LocalNoiseConfig(),
        geometry_noise=GeometryNoiseConfig(),
        kernel_fn=crossing_kernel,
    )
    shots = circuit.compile_detector_sampler().sample(8)
    assert shots.sum() == 0
    assert circuit.detector_error_model().num_detectors > 0


def test_z_memory_circuit_is_deterministic_without_noise() -> None:
    _assert_noiseless_detectors_are_zero('z_memory')


def test_x_memory_circuit_is_deterministic_without_noise() -> None:
    _assert_noiseless_detectors_are_zero('x_memory')


def test_geometry_enabled_builds_for_both_embeddings() -> None:
    spec = build_bb72()
    local = LocalNoiseConfig()
    geom = GeometryNoiseConfig(enabled=True, J0=0.08, tau=1.0)
    kernel = regularized_power_law_kernel(alpha=3.0, r0=1.0)
    for embedding in (MonomialColumnEmbedding(spec), IBMToricBiplanarEmbedding(spec)):
        circuit, _ = build_bb_memory_experiment(
            spec,
            embedding=embedding,
            rounds=1,
            experiment_kind='z_memory',
            local_noise=local,
            geometry_noise=geom,
            kernel_fn=kernel,
        )
        circuit.detector_error_model()


def test_decoded_sector_helper() -> None:
    assert decoded_sector('z_memory') == 'X'
    assert decoded_sector('x_memory') == 'Z'


def test_data_measure_basis_helper() -> None:
    assert data_measure_basis('z_memory') == 'Z'
    assert data_measure_basis('x_memory') == 'X'


def test_get_embedding_returns_correct_classes() -> None:
    spec = build_bb72()
    assert isinstance(get_embedding(spec, 'ibm_biplanar'), IBMToricBiplanarEmbedding)
    assert isinstance(get_embedding(spec, 'ibm_biplanar_surrogate'), IBMBiplanarSurrogateEmbedding)


def test_toric_biplanar_cross_layer_no_crossings() -> None:
    """Edges routed through different z-planes (layer A vs layer B) should
    never produce zero-distance pairs, verifying the biplanar separation."""
    spec = build_bb72()
    toric = IBMToricBiplanarEmbedding(spec)
    B1, B2, B3 = spec.B_terms
    # B1 routes through layer B (z=-h), B3 through layer A (z=+h)
    geom_b1 = toric.routing_geometry(
        control_reg='L', target_reg='Z',
        term_name='B1', term=B1, transpose=True, name='B1',
    )
    geom_b3 = toric.routing_geometry(
        control_reg='L', target_reg='Z',
        term_name='B3', term=B3, transpose=True, name='B3',
    )
    # Merge edge polylines from different layers and check for crossings
    merged = {}
    merged.update(geom_b1.edge_polylines)
    # Offset keys to avoid collision
    for k, v in geom_b3.edge_polylines.items():
        merged[('B3_' + k[0], k[1], 'B3_' + k[2], k[3])] = v
    cross_layer_crossings = count_zero_distance_pairs(merged)
    # Same-layer edges can have in-plane crossings due to torus wrapping,
    # but cross-layer edges are separated by 2h vertically → zero crossings.
    # The total includes some same-layer crossings within B1 and B3 individually;
    # the key check is that no cross-layer pair has zero distance.
    from bbstim.geometry import polyline_distance
    import itertools
    b1_items = list(geom_b1.edge_polylines.items())
    b3_items = list(geom_b3.edge_polylines.items())
    for (e1, p1), (e2, p2) in itertools.product(b1_items, b3_items):
        # Skip pairs that share a vertex (they trivially have zero distance)
        verts1 = {(e1[0], e1[1]), (e1[2], e1[3])}
        verts2 = {(e2[0], e2[1]), (e2[2], e2[3])}
        if verts1 & verts2:
            continue
        d = polyline_distance(p1, p2)
        assert d > 0, f'Cross-layer edges {e1} and {e2} have zero distance'


def test_monomial_column_has_crossings() -> None:
    """Monomial column is single-layer — crossings expected on B rounds."""
    spec = build_bb72()
    mono = MonomialColumnEmbedding(spec)
    B1, B2, B3 = spec.B_terms
    b_rounds = [
        ('L', 'Z', 'B1', B1, True),
        ('L', 'Z', 'B2', B2, True),
        ('L', 'Z', 'B3', B3, True),
    ]
    mono_crossings = 0
    for control_reg, target_reg, term_name, term, transpose in b_rounds:
        mono_geom = mono.routing_geometry(
            control_reg=control_reg, target_reg=target_reg,
            term=term, transpose=transpose, name=term_name,
        )
        mono_crossings += count_zero_distance_pairs(mono_geom.edge_polylines)
    assert mono_crossings > 0


def test_full_cycle_includes_both_ancilla_streams() -> None:
    """Full IBM cycle should include both q(X) and q(Z) operations."""
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    for kind in ('z_memory', 'x_memory'):
        circuit, _ = build_bb_memory_experiment(
            spec,
            embedding=embedding,
            rounds=1,
            experiment_kind=kind,
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            kernel_fn=crossing_kernel,
        )
        circ_str = str(circuit)
        # Both R and RX resets should be present (q(Z) and q(X))
        assert 'R ' in circ_str or '\nR ' in circ_str, f'{kind}: missing Z-basis reset'
        assert 'RX' in circ_str, f'{kind}: missing X-basis reset'
        # Both M and MX measurements should be present
        assert 'MX' in circ_str, f'{kind}: missing X-basis measurement'
        # Circuit should be deterministic
        shots = circuit.compile_detector_sampler().sample(8)
        assert shots.sum() == 0
        assert circuit.detector_error_model().num_detectors > 0


def test_full_cycle_one_round_schedule() -> None:
    """For a one-cycle build, verify the IBM schedule structure."""
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    circuit, _ = build_bb_memory_experiment(
        spec,
        embedding=embedding,
        rounds=1,
        experiment_kind='z_memory',
        local_noise=LocalNoiseConfig(),
        geometry_noise=GeometryNoiseConfig(),
        kernel_fn=crossing_kernel,
    )
    circ_str = str(circuit)
    # Count TICK markers — should have 8 per cycle + 1 for final data measurement
    tick_count = circ_str.count('TICK')
    assert tick_count == 9, f'Expected 9 TICKs, got {tick_count}'


def test_geometry_scope_theory_reduced_fewer_channels() -> None:
    """theory_reduced should insert fewer pair channels than full_cycle."""
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    local = LocalNoiseConfig()
    kernel = regularized_power_law_kernel(alpha=3.0, r0=1.0)

    geom_full = GeometryNoiseConfig(enabled=True, J0=0.08, tau=1.0, geometry_scope='full_cycle')
    circ_full, _ = build_bb_memory_experiment(
        spec, embedding=embedding, rounds=1, experiment_kind='z_memory',
        local_noise=local, geometry_noise=geom_full, kernel_fn=kernel,
    )
    n_full = sum(1 for inst in circ_full.flattened() if inst.name == 'E')

    geom_reduced = GeometryNoiseConfig(enabled=True, J0=0.08, tau=1.0, geometry_scope='theory_reduced')
    circ_reduced, _ = build_bb_memory_experiment(
        spec, embedding=embedding, rounds=1, experiment_kind='z_memory',
        local_noise=local, geometry_noise=geom_reduced, kernel_fn=kernel,
    )
    n_reduced = sum(1 for inst in circ_reduced.flattened() if inst.name == 'E')

    assert n_reduced < n_full, f'theory_reduced ({n_reduced}) should have fewer channels than full_cycle ({n_full})'
    assert n_reduced > 0, 'theory_reduced should still have some channels'
    assert n_full > 0, 'full_cycle should have channels'


def test_final_data_measurement_uses_measurement_noise() -> None:
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    circuit, _ = build_bb_memory_experiment(
        spec,
        embedding=embedding,
        rounds=1,
        experiment_kind='z_memory',
        local_noise=LocalNoiseConfig(p_meas=1e-3),
        geometry_noise=GeometryNoiseConfig(),
        kernel_fn=crossing_kernel,
    )
    circ_str = str(circuit)
    # The final data measurement should be preceded by a measurement-noise insertion.
    # For Z-basis data measurement, _append_measure_basis uses X_ERROR before M.
    assert 'X_ERROR' in circ_str


def test_biplanar_theory_reduced_crossing_kernel_has_zero_pair_channels() -> None:
    spec = build_bb72()
    embedding = IBMToricBiplanarEmbedding(spec)
    local = LocalNoiseConfig()
    geom = GeometryNoiseConfig(enabled=True, J0=0.08, tau=1.0, geometry_scope='theory_reduced')
    circuit, _ = build_bb_memory_experiment(
        spec, embedding=embedding, rounds=1, experiment_kind='z_memory',
        local_noise=local, geometry_noise=geom, kernel_fn=crossing_kernel,
    )
    n_pair = sum(1 for inst in circuit.flattened() if inst.name == 'E')
    assert n_pair == 0, f'Expected zero crossing-local pair channels for biplanar embedding, got {n_pair}'


def test_monomial_theory_reduced_crossing_kernel_has_pair_channels() -> None:
    spec = build_bb72()
    embedding = MonomialColumnEmbedding(spec)
    local = LocalNoiseConfig()
    geom = GeometryNoiseConfig(enabled=True, J0=0.08, tau=1.0, geometry_scope='theory_reduced')
    circuit, _ = build_bb_memory_experiment(
        spec, embedding=embedding, rounds=1, experiment_kind='z_memory',
        local_noise=local, geometry_noise=geom, kernel_fn=crossing_kernel,
    )
    n_pair = sum(1 for inst in circuit.flattened() if inst.name == 'E')
    assert n_pair > 0, 'Expected crossing-local pair channels for monomial embedding'
