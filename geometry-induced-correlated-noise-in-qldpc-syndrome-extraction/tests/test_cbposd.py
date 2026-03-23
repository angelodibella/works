# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
"""Tests for the correlation-aware BP+OSD decoder (Chapter 7 of the workbook)."""
import numpy as np
import pytest
import stim

from bbstim.bbcode import build_bb72
from bbstim.decoders import build_cbposd_dem, collect_pair_edges, CompiledCBPOSDDecoder
from bbstim.embeddings import MonomialColumnEmbedding
from bbstim.geometry import crossing_kernel, regularized_power_law_kernel


@pytest.fixture
def bb72():
    return build_bb72()


@pytest.fixture
def mono_emb(bb72):
    return MonomialColumnEmbedding(bb72)


class TestBuildCbposdDem:
    """Tests for the synthetic DEM construction (Theorem 7.3 / Definition 7.5)."""

    def test_no_pairs_reduces_to_standard(self, bb72):
        """Proposition 7.7: with no pair edges, H̃ = H_sigma, F = I_n."""
        dem, F, logicals = build_cbposd_dem(bb72, 'X', [], u_single=0.001)
        n = bb72.n_data
        # F should be identity
        assert F.shape == (n, n)
        assert np.array_equal(F, np.eye(n, dtype=np.uint8))
        # DEM should have spec.half detectors and n error mechanisms
        assert dem.num_detectors == bb72.half
        assert dem.num_errors <= n  # some columns might be all-zero in H_z
        # Logical matrix should have k=12 logicals
        assert logicals.shape[0] == 12

    def test_augmented_shape_with_pairs(self, bb72, mono_emb):
        """H̃ has shape (half, n_data + m) and F has shape (n_data, n_data + m)."""
        pair_edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=crossing_kernel,
        )
        assert len(pair_edges) > 0, "Monomial crossing should have pair edges"
        m = len(pair_edges)
        dem, F, logicals = build_cbposd_dem(bb72, 'X', pair_edges, u_single=0.001)
        n = bb72.n_data
        assert F.shape == (n, n + m)
        assert dem.num_detectors == bb72.half

    def test_H_tilde_z_equals_H_sigma_Fz(self, bb72, mono_emb):
        """Verify H̃ z = H_σ F z (mod 2) for random z vectors."""
        pair_edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=crossing_kernel,
        )
        m = len(pair_edges)
        n = bb72.n_data
        H_sigma = bb72.hz()
        # Reconstruct H_tilde from the same construction
        F = np.zeros((n, n + m), dtype=np.uint8)
        F[:n, :n] = np.eye(n, dtype=np.uint8)
        for a, (i, j, _) in enumerate(pair_edges):
            F[i, n + a] = 1
            F[j, n + a] = 1
        H_tilde = (H_sigma @ F) % 2

        rng = np.random.default_rng(42)
        for _ in range(20):
            z = rng.integers(0, 2, size=n + m, dtype=np.uint8)
            lhs = (H_tilde @ z) % 2
            rhs = (H_sigma @ ((F @ z) % 2)) % 2
            assert np.array_equal(lhs, rhs), "H̃z ≠ H_σ(Fz) mod 2"

    def test_pair_observable_assignment(self, bb72):
        """Pair-fault column flips observable k iff L_k[i] ⊕ L_k[j] = 1."""
        x_logs, z_logs = bb72.logical_bases()
        # Use z_logs (for z_memory / X-sector)
        logicals = z_logs
        n = bb72.n_data
        # Construct a fake pair edge at (0, 1)
        pair_edges = [(0, 1, 0.01)]
        _dem, F, _log = build_cbposd_dem(bb72, 'X', pair_edges, u_single=0.001)
        # The pair-fault column is the last column of F
        pair_col = F[:, n]  # column n+0
        assert pair_col[0] == 1 and pair_col[1] == 1, "Pair column should have 1s at i,j"
        # Observable flip for this pair: L_k[0] XOR L_k[1] for each logical k
        expected_obs = (logicals[:, 0] ^ logicals[:, 1]).astype(np.uint8)
        actual_obs = ((logicals @ pair_col) % 2).astype(np.uint8)
        assert np.array_equal(actual_obs, expected_obs)


class TestCollectPairEdges:
    """Tests for pair-edge collection from geometry pipeline."""

    def test_crossing_has_pairs_for_monomial(self, bb72, mono_emb):
        """Monomial BB72 with crossing kernel should have nonzero pair edges."""
        edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=crossing_kernel,
        )
        assert len(edges) > 0
        # All edges should be on L-register (indices < half)
        for i, j, v in edges:
            assert 0 <= i < bb72.half, f"i={i} not in L-register"
            assert 0 <= j < bb72.half, f"j={j} not in L-register"
            assert 0 < v <= 1.0, f"v={v} out of range"

    def test_no_pairs_at_j0_zero(self, bb72, mono_emb):
        """J0=0 should produce no pair edges (no geometry noise)."""
        edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.0, tau=1.0, kernel_fn=crossing_kernel,
        )
        assert len(edges) == 0

    def test_powerlaw_has_more_pairs_than_crossing(self, bb72, mono_emb):
        """Power-law kernel should produce more pair edges than crossing."""
        crossing_edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=crossing_kernel,
        )
        powerlaw_fn = regularized_power_law_kernel(alpha=3.0, r0=1.0)
        powerlaw_edges = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=powerlaw_fn,
        )
        assert len(powerlaw_edges) >= len(crossing_edges)

    def test_truncation_reduces_edges(self, bb72, mono_emb):
        """Higher epsilon should produce fewer edges."""
        powerlaw_fn = regularized_power_law_kernel(alpha=3.0, r0=1.0)
        edges_loose = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=powerlaw_fn,
            epsilon=1e-10,
        )
        edges_tight = collect_pair_edges(
            bb72, mono_emb, 'X',
            J0=0.08, tau=1.0, kernel_fn=powerlaw_fn,
            epsilon=0.01,
        )
        assert len(edges_tight) <= len(edges_loose)
