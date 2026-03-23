# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
"""Core tests for BB72 logical family, biplanar embedding, and semantic plotting."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbstim.experiments import get_code, get_embedding, get_kernel, BB72_X_LOGICAL_SUPPORT_L
from bbstim.embeddings import (
    enumerate_pure_L_minwt_logicals, _reference_family, _matching_number,
    _segment_distance_2d, _J_exposure,
)
from bbstim.geometry import count_zero_distance_pairs, pairwise_round_coefficients, polyline_distance
from bbstim.plotting import _parse_kernel_params, _sem


# ═══════════════════════════════════════════════════════════════════════
#  1. BB72 pure-L family tests
# ═══════════════════════════════════════════════════════════════════════

class TestBB72PureLFamily:
    @pytest.fixture(scope="class")
    def spec(self):
        return get_code('BB72')

    @pytest.fixture(scope="class")
    def family(self, spec):
        return enumerate_pure_L_minwt_logicals(spec)

    def test_workbook_support_in_family(self, family):
        """Workbook support {3,12,21,24,27,33} must be in the true family."""
        workbook = {3, 12, 21, 24, 27, 33}
        assert workbook in family

    def test_all_supports_pure_L(self, spec, family):
        """Every support must be a subset of L-register indices [0, half)."""
        for support in family:
            assert all(0 <= i < spec.half for i in support)

    def test_all_supports_same_weight(self, family):
        """All supports must have the same minimum weight."""
        weights = {len(s) for s in family}
        assert len(weights) == 1, f"Found multiple weights: {weights}"

    def test_all_supports_nontrivial(self, spec, family):
        """Each support must correspond to a nontrivial X-logical (u ∉ T_L)."""
        B = spec.polynomial_matrix(spec.B_terms)
        BT = B.T
        from bbstim.algebra import TL_basis, rowspace_gf2

        TL = TL_basis(spec)

        for support in family:
            u = np.zeros(spec.half, dtype=np.uint8)
            for i in support:
                u[i] = 1
            # Must commute with Z-checks: B^T u = 0
            assert np.all((BT @ u) % 2 == 0), f"B^T u != 0 for support {support}"
            # Must be nontrivial: u ∉ T_L = { λA : λB = 0 }
            aug = np.vstack([TL, u.reshape(1, -1)])
            rank_aug = len(rowspace_gf2(aug))
            assert rank_aug > len(TL), f"Support {support} is in T_L (trivial)"

    def test_reference_family_matches(self, spec, family):
        """_reference_family() must return the same set as direct enumeration."""
        ref = _reference_family(spec)
        assert set(frozenset(s) for s in ref) == set(frozenset(s) for s in family)

    def test_family_count(self, family):
        """BB72 should have exactly 36 weight-6 pure-L X-logicals."""
        assert len(family) == 36


# ═══════════════════════════════════════════════════════════════════════
#  2. IBM-inspired bounded-thickness embedding tests
# ═══════════════════════════════════════════════════════════════════════

class TestBiplanarEmbedding:
    @pytest.fixture(scope="class")
    def spec(self):
        return get_code('BB72')

    @pytest.fixture(scope="class")
    def emb(self, spec):
        return get_embedding(spec, 'ibm_biplanar')

    def test_layer_partition(self, emb):
        """Layer A = {A2, A3, B3}, Layer B = {A1, B1, B2}."""
        from bbstim.embeddings import IBMToricBiplanarEmbedding
        assert IBMToricBiplanarEmbedding.LAYER_A_TERMS == frozenset({'A2', 'A3', 'B3'})
        assert IBMToricBiplanarEmbedding.LAYER_B_TERMS == frozenset({'A1', 'B1', 'B2'})

    def test_crossing_kernel_empty_support_graph(self, spec, emb):
        """Under crossing kernel, no support-graph edges for biplanar (Prop. 5.4)."""
        rounds = []
        for term, tname in zip(spec.B_terms, ['B1', 'B2', 'B3']):
            g = emb.routing_geometry(
                control_reg='L', target_reg='Z',
                term_name=tname, term=term, transpose=True, name=tname,
            )
            rounds.append(g)
        total_crossings = sum(count_zero_distance_pairs(r.edge_polylines) for r in rounds)
        assert total_crossings == 0

    def test_b_round_layer_separation(self, spec, emb):
        """B1,B2 in layer B (z<0), B3 in layer A (z>0) — Prop. 5.6."""
        for term, tname in zip(spec.B_terms, ['B1', 'B2', 'B3']):
            g = emb.routing_geometry(
                control_reg='L', target_reg='Z',
                term_name=tname, term=term, transpose=True, name=tname,
            )
            for edge_key, polyline in g.edge_polylines.items():
                # Middle points (indices 2,3) are in the routing plane
                for pt in polyline[2:-2]:
                    if tname in ('B1', 'B2'):
                        assert pt[2] < 0, f"{tname} edge should be in z<0"
                    else:  # B3
                        assert pt[2] > 0, f"{tname} edge should be in z>0"


# ═══════════════════════════════════════════════════════════════════════
#  3. Semantic plot filtering tests
# ═══════════════════════════════════════════════════════════════════════

class TestSemanticFiltering:
    @pytest.fixture
    def sample_df(self):
        """Create a minimal DataFrame with semantically duplicate rows."""
        rows = [
            # S2 J0=0.04 point from bb72_v3_pj0 suite
            {'experiment_id': 'bb72_v3_pj0_0.04_mono', 'code': 'BB72',
             'embedding': 'monomial_column', 'experiment_kind': 'z_memory',
             'kernel': 'powerlaw', 'kernel_params': "{'alpha': 3.0, 'r0': 1.0}",
             'J0': 0.04, 'p_cnot': 1e-3, 'primary_shots': 5000,
             'primary_failures': 1452, 'primary_ler_total': 0.2904,
             'primary_ler_total_lo': 0.28, 'primary_ler_total_hi': 0.30},
            # Same physics from bb72_v3_a3.0_mono (alpha sweep)
            {'experiment_id': 'bb72_v3_a3.0_mono', 'code': 'BB72',
             'embedding': 'monomial_column', 'experiment_kind': 'z_memory',
             'kernel': 'powerlaw', 'kernel_params': "{'alpha': 3.0, 'r0': 1.0}",
             'J0': 0.04, 'p_cnot': 1e-3, 'primary_shots': 5000,
             'primary_failures': 1434, 'primary_ler_total': 0.2868,
             'primary_ler_total_lo': 0.27, 'primary_ler_total_hi': 0.30},
            # Different point
            {'experiment_id': 'bb72_v3_pj0_0.08_mono', 'code': 'BB72',
             'embedding': 'monomial_column', 'experiment_kind': 'z_memory',
             'kernel': 'powerlaw', 'kernel_params': "{'alpha': 3.0, 'r0': 1.0}",
             'J0': 0.08, 'p_cnot': 1e-3, 'primary_shots': 5000,
             'primary_failures': 4643, 'primary_ler_total': 0.9286,
             'primary_ler_total_lo': 0.92, 'primary_ler_total_hi': 0.94},
        ]
        df = pd.DataFrame(rows)
        return _parse_kernel_params(df)

    def test_dedup_keeps_max_shots(self, sample_df):
        """Semantic dedup should keep the row with most shots (or first if tied)."""
        result = _sem(sample_df, code='BB72', kernel='powerlaw', p_cnot=1e-3)
        # Two distinct J0 values after dedup
        assert len(result) == 2

    def test_no_false_gaps(self, sample_df):
        """Alpha sweep at J0=0.04 should find the point regardless of experiment_id."""
        result = _sem(sample_df, code='BB72', kernel='powerlaw', J0=0.04, p_cnot=1e-3)
        assert len(result) == 1
        # Should not be empty just because experiment_id doesn't start with 'bb72_v3_a'


# ═══════════════════════════════════════════════════════════════════════
#  4. Matching number exactness (C2)
# ═══════════════════════════════════════════════════════════════════════

class TestMatchingNumber:
    def test_empty_graph(self):
        assert _matching_number({1, 2, 3}, set()) == 0

    def test_single_edge(self):
        assert _matching_number({1, 2}, {(1, 2)}) == 1

    def test_triangle(self):
        """Triangle is non-bipartite; matching = 1."""
        assert _matching_number({1, 2, 3}, {(1, 2), (2, 3), (1, 3)}) == 1

    def test_k33(self):
        """K_{3,3} bipartite complete graph; matching = 3."""
        A, B = [1, 2, 3], [4, 5, 6]
        edges = {(a, b) for a in A for b in B}
        assert _matching_number(set(A + B), edges) == 3

    def test_k33_plus_triangle(self):
        """K_{3,3} + triangle on B = the BB72 workbook support graph; matching = 3."""
        A, B = [3, 12, 21], [24, 27, 33]
        edges = {(a, b) for a in A for b in B}
        edges |= {(24, 27), (24, 33), (27, 33)}
        assert _matching_number(set(A + B), edges) == 3

    def test_odd_cycle_c5(self):
        """C_5 (5-cycle) is non-bipartite; maximum matching = 2."""
        verts = {0, 1, 2, 3, 4}
        edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)}
        assert _matching_number(verts, edges) == 2

    def test_petersen_graph(self):
        """Petersen graph: 10 vertices, non-bipartite, matching = 5 (perfect)."""
        outer = [(i, (i + 1) % 5) for i in range(5)]
        inner = [(i + 5, (i + 2) % 5 + 5) for i in range(5)]
        spokes = [(i, i + 5) for i in range(5)]
        edges = set(outer + inner + spokes)
        verts = set(range(10))
        assert _matching_number(verts, edges) == 5


# ═══════════════════════════════════════════════════════════════════════
#  5. Exposure distance exactness (C2)
# ═══════════════════════════════════════════════════════════════════════

class TestExposureDistance:
    def test_crossing_segments_distance_zero(self):
        """Two segments that cross must have distance 0."""
        # Seg A: (1,0) → (3,10), Seg B: (1,10) → (3,0) — they cross
        d = _segment_distance_2d(1, 0, 3, 10, 1, 10, 3, 0)
        assert d < 1e-10

    def test_parallel_segments(self):
        """Two parallel horizontal segments at y=0 and y=5."""
        d = _segment_distance_2d(0, 0, 10, 0, 0, 5, 10, 5)
        assert abs(d - 5.0) < 1e-10

    def test_non_crossing_non_parallel(self):
        """Non-parallel non-crossing segments: exact dist < endpoint dist."""
        # Seg A: (1,0) → (3,4), Seg B: (1,2) → (3,8)
        # Endpoints: |0-2|=2 and |4-8|=4, so old code gives min=2
        # But segments diverge, so closest approach is at the source end
        d = _segment_distance_2d(1, 0, 3, 4, 1, 2, 3, 8)
        assert d <= 2.0  # must be ≤ endpoint min

    def test_agrees_with_geometry_engine(self):
        """_segment_distance_2d must agree with geometry.polyline_distance on 3D segments at z=0."""
        cases = [
            ((1, 0, 3, 10), (1, 10, 3, 0)),     # crossing
            ((1, 0, 3, 4), (1, 2, 3, 8)),         # non-crossing
            ((1, 5, 3, 5), (1, 15, 3, 15)),        # parallel
            ((1, 0, 3, 20), (1, 3, 3, 7)),         # converging then diverging
        ]
        for (x1, y1, x2, y2), (x3, y3, x4, y4) in cases:
            d_2d = _segment_distance_2d(x1, y1, x2, y2, x3, y3, x4, y4)
            poly1 = [(x1, y1, 0.0), (x2, y2, 0.0)]
            poly2 = [(x3, y3, 0.0), (x4, y4, 0.0)]
            d_3d = polyline_distance(poly1, poly2)
            assert abs(d_2d - d_3d) < 1e-8, (
                f"Mismatch: 2d={d_2d}, 3d={d_3d} for segs "
                f"({x1},{y1})→({x2},{y2}) vs ({x3},{y3})→({x4},{y4})"
            )

    def test_exposure_deterministic(self):
        """_J_exposure must be deterministic for the identity permutation."""
        spec = get_code('BB72')
        family = enumerate_pure_L_minwt_logicals(spec)
        sigma = list(range(spec.half))
        B_terms = list(spec.B_terms)
        e1 = _J_exposure(spec, sigma, sigma, family, B_terms,
                         1.0, 1.0, 3.0, 1.0, 0.04, 3.0, 1.0)
        e2 = _J_exposure(spec, sigma, sigma, family, B_terms,
                         1.0, 1.0, 3.0, 1.0, 0.04, 3.0, 1.0)
        assert e1 == e2

    def test_j_exposure_matches_geometry_engine(self):
        """_J_exposure must agree with the full geometry engine on BB72 monomial.

        This is the critical regression test: the LA optimizer's internal
        distance computation must be exact relative to the geometry engine
        that drives the Monte Carlo simulations.
        """
        from bbstim.geometry import (
            pairwise_round_coefficients, weighted_exposure_on_support,
            regularized_power_law_kernel,
        )
        from bbstim.embeddings import MonomialColumnEmbedding

        spec = get_code('BB72')
        family = enumerate_pure_L_minwt_logicals(spec)
        sigma = list(range(spec.half))
        B_terms = list(spec.B_terms)

        # LA-internal computation
        la_max = _J_exposure(spec, sigma, sigma, family, B_terms,
                             1.0, 1.0, 3.0, 1.0, 0.04, 3.0, 1.0)

        # Geometry engine computation
        emb = MonomialColumnEmbedding(spec)
        kernel_fn = regularized_power_law_kernel(3.0, 1.0)
        rounds = [
            emb.routing_geometry(control_reg='L', target_reg='Z',
                                 term=t, transpose=True, name=n)
            for t, n in zip(spec.B_terms, ['B1', 'B2', 'B3'])
        ]
        coeffs = [
            pairwise_round_coefficients(r.edge_polylines, tau=1.0, J0=0.04,
                                         kernel=kernel_fn, use_weak_limit=False)
            for r in rounds
        ]
        geo_exps = [weighted_exposure_on_support(sorted(s), coeffs) for s in family]
        geo_max = max(geo_exps)

        assert abs(la_max - geo_max) < 1e-10, (
            f"LA-internal max exposure {la_max:.10f} != "
            f"geometry engine {geo_max:.10f}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  6. BB90 and BB108 intermediate codes
# ═══════════════════════════════════════════════════════════════════════

class TestIntermediateBBCodes:
    def test_bb90_parameters(self):
        spec = get_code('BB90')
        assert spec.n_data == 90
        assert spec.half == 45
        x_logs, _ = spec.logical_bases()
        assert x_logs.shape[0] == 8  # k=8

    def test_bb108_parameters(self):
        spec = get_code('BB108')
        assert spec.n_data == 108
        assert spec.half == 54
        x_logs, _ = spec.logical_bases()
        assert x_logs.shape[0] == 8  # k=8

    def test_bb90_embeddings_construct(self):
        spec = get_code('BB90')
        mono = get_embedding(spec, 'monomial_column')
        bi = get_embedding(spec, 'ibm_biplanar')
        assert mono is not None
        assert bi is not None

    def test_bb108_embeddings_construct(self):
        spec = get_code('BB108')
        mono = get_embedding(spec, 'monomial_column')
        bi = get_embedding(spec, 'ibm_biplanar')
        assert mono is not None
        assert bi is not None

    def test_biplanar_layer_split_bb90(self):
        """Bounded-thickness layer split must work for BB90."""
        spec = get_code('BB90')
        bi = get_embedding(spec, 'ibm_biplanar')
        g = bi.routing_geometry(control_reg='L', target_reg='Z',
                                term_name='B1', term=spec.B_terms[0],
                                transpose=True, name='B1')
        crossings = count_zero_distance_pairs(g.edge_polylines)
        assert crossings == 0

    def test_biplanar_layer_split_bb108(self):
        """Bounded-thickness layer split must work for BB108."""
        spec = get_code('BB108')
        bi = get_embedding(spec, 'ibm_biplanar')
        g = bi.routing_geometry(control_reg='L', target_reg='Z',
                                term_name='B1', term=spec.B_terms[0],
                                transpose=True, name='B1')
        crossings = count_zero_distance_pairs(g.edge_polylines)
        assert crossings == 0

    def test_pure_L_quotient_bb90(self):
        from bbstim.algebra import pure_L_quotient_dimension
        dk, dt, dq = pure_L_quotient_dimension(get_code('BB90'))
        assert dq > 0  # nontrivial quotient

    def test_pure_L_quotient_bb108(self):
        from bbstim.algebra import pure_L_quotient_dimension
        dk, dt, dq = pure_L_quotient_dimension(get_code('BB108'))
        assert dq > 0

    def test_bb108_benchmark_metadata(self):
        """BB108 with our polynomials is [[108,8,12]]."""
        spec = get_code('BB108')
        assert spec.benchmark_n == 108
        assert spec.benchmark_k == 8
        assert spec.benchmark_d == 12

    def test_bb108_pureL_weight_matches_benchmark_d(self):
        """Pure-q(L) min weight equals benchmark d=12 for our BB108 variant."""
        from bbstim.algebra import enumerate_pure_L_minwt_logicals
        spec = get_code('BB108')
        family = enumerate_pure_L_minwt_logicals(spec)
        pureL_wt = len(next(iter(family)))
        assert pureL_wt == 12
        assert spec.benchmark_d == 12
