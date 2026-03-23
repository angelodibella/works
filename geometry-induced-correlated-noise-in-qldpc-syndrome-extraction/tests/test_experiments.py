# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
import math

import pytest

from bbstim.experiments import Experiment, get_code, get_embedding, run_experiment, experiment_decoded_sector


def test_invalid_code_and_embedding_raise() -> None:
    with pytest.raises(ValueError):
        get_code('invalid')
    with pytest.raises(ValueError):
        get_embedding(get_code('BB72'), 'invalid')


def test_reference_exposure_is_not_reported_for_mismatched_sector() -> None:
    # x_memory decodes the Z-error sector — NOT the theory-facing X-error sector.
    # The workbook theory is about X-errors detected by the Z-check cycle
    # (experiment_kind='z_memory'), so exposure should be NaN for x_memory.
    row = run_experiment(
        Experiment(
            experiment_id='unit_smoke',
            code='BB72',
            embedding='monomial_column',
            experiment_kind='x_memory',
            cycles=1,
            shots=16,
            p_cnot=0.0,
            p_idle=0.0,
            p_prep=0.0,
            p_meas=0.0,
            kernel='crossing',
            kernel_params={},
            J0=0.0,
            tau=1.0,
        ),
        num_workers=1,
    )
    assert math.isnan(row['reference_weighted_exposure'])
    assert math.isnan(row['reference_aggregate_pair_probability_max'])
    assert math.isnan(row['reference_aggregate_amplitude_max'])
    assert math.isnan(row['reference_aggregate_location_strength_max'])
    assert 'primary_error' not in row
    assert row['primary_failures'] == 0


def test_reference_exposure_is_reported_for_theory_facing_sector() -> None:
    # z_memory runs the Z-check cycle (X-error sector) — the theory-facing
    # sector for the workbook's q(L)->q(Z) weighted exposure.
    row = run_experiment(
        Experiment(
            experiment_id='unit_z_exposure',
            code='BB72',
            embedding='monomial_column',
            experiment_kind='z_memory',
            cycles=1,
            shots=16,
            p_cnot=0.0,
            p_idle=0.0,
            p_prep=0.0,
            p_meas=0.0,
            kernel='crossing',
            kernel_params={},
            J0=0.0,
            tau=1.0,
        ),
        num_workers=1,
    )
    assert not math.isnan(row['reference_weighted_exposure'])
    assert not math.isnan(row['reference_aggregate_pair_probability_max'])
    assert not math.isnan(row['reference_aggregate_amplitude_max'])
    assert not math.isnan(row['reference_aggregate_location_strength_max'])
    assert row['primary_failures'] == 0


def test_decoded_sector_semantics() -> None:
    """Verify that z_memory decodes X-errors and x_memory decodes Z-errors."""
    exp_z = Experiment(
        experiment_id='test', code='BB72', embedding='monomial_column',
        experiment_kind='z_memory', cycles=1, shots=1,
        p_cnot=0.0, p_idle=0.0, p_prep=0.0, p_meas=0.0,
        kernel='crossing', kernel_params={}, J0=0.0, tau=1.0,
    )
    exp_x = Experiment(
        experiment_id='test', code='BB72', embedding='monomial_column',
        experiment_kind='x_memory', cycles=1, shots=1,
        p_cnot=0.0, p_idle=0.0, p_prep=0.0, p_meas=0.0,
        kernel='crossing', kernel_params={}, J0=0.0, tau=1.0,
    )
    assert experiment_decoded_sector(exp_z) == 'X'
    assert experiment_decoded_sector(exp_x) == 'Z'


def test_theory_facing_suite_defaults_to_theory_reduced() -> None:
    exp = Experiment(
        experiment_id='test', code='BB72', embedding='monomial_column',
        experiment_kind='z_memory', cycles=1, shots=1,
        p_cnot=0.0, p_idle=0.0, p_prep=0.0, p_meas=0.0,
        kernel='crossing', kernel_params={}, J0=0.0, tau=1.0,
    )
    assert exp.geometry_scope == 'theory_reduced'


def test_amplitude_and_location_strength_metrics_agree_at_tau_one() -> None:
    row = run_experiment(
        Experiment(
            experiment_id='unit_metric_consistency',
            code='BB72',
            embedding='monomial_column',
            experiment_kind='z_memory',
            cycles=1,
            shots=16,
            p_cnot=0.0,
            p_idle=0.0,
            p_prep=0.0,
            p_meas=0.0,
            kernel='powerlaw',
            kernel_params={'alpha': 3.0, 'r0': 1.0},
            J0=0.08,
            tau=1.0,
            geometry_scope='theory_reduced',
        ),
        num_workers=1,
    )
    assert math.isclose(
        row['reference_aggregate_amplitude_max'],
        row['reference_aggregate_location_strength_max'],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_reference_exposure_not_reported_for_full_cycle_control() -> None:
    row = run_experiment(
        Experiment(
            experiment_id='unit_full_cycle',
            code='BB72',
            embedding='monomial_column',
            experiment_kind='z_memory',
            cycles=1,
            shots=16,
            p_cnot=0.0,
            p_idle=0.0,
            p_prep=0.0,
            p_meas=0.0,
            kernel='crossing',
            kernel_params={},
            J0=0.0,
            tau=1.0,
            geometry_scope='full_cycle',
        ),
        num_workers=1,
    )
    assert math.isnan(row['reference_weighted_exposure'])
