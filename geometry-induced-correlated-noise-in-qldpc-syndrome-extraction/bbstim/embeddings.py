# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import json
import math
import random

import networkx as nx
import numpy as np

from .bbcode import BBCodeSpec, Shift
from .algebra import enumerate_pure_L_minwt_logicals, nullspace_gf2, rowspace_gf2, reference_family_hash

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Polyline = list[Point3]

@dataclass
class RoutingGeometry:
    name: str
    edge_polylines: dict[tuple[str, int, str, int], Polyline]

class MonomialColumnEmbedding:
    def __init__(self, spec: BBCodeSpec, x_positions: Mapping[str, float] | None = None, scale_y: float = 1.0):
        self.spec = spec
        self.x_positions = dict(x_positions or {'X': 0.0, 'L': 1.0, 'R': 2.0, 'Z': 3.0})
        self.scale_y = scale_y
        self.points: dict[tuple[str, int], Point3] = {}
        for reg in ('X', 'L', 'R', 'Z'):
            for i in range(spec.half):
                self.points[(reg, i)] = (self.x_positions[reg], self.scale_y * i, 0.0)

    def routing_geometry(
        self,
        *,
        control_reg: str,
        target_reg: str,
        term: Shift,
        transpose: bool = False,
        name: str | None = None,
        term_name: str | None = None,
    ) -> RoutingGeometry:
        out: dict[tuple[str, int, str, int], Polyline] = {}
        for i in range(self.spec.half):
            tgt = self.spec.mapped_target_index(i, term, transpose, target_reg)
            u = (control_reg, i)
            v = (target_reg, tgt)
            out[(u[0], u[1], v[0], v[1])] = [self.points[u], self.points[v]]
        return RoutingGeometry(name or term_name or 'mono', out)

    def params_dict(self) -> dict:
        return {'code': self.spec.name, 'embedding_type': 'monomial_column',
                'scale_y': self.scale_y, 'x_positions': self.x_positions}

    def params_hash(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(self.params_dict(), sort_keys=True).encode()).hexdigest()[:16]

class IBMBiplanarSurrogateEmbedding:
    """Layer-local surrogate inspired by the IBM bounded-thickness decomposition.

    All qubits sit on a common base plane (z=0).  Edges belonging to layer A
    (terms A2, A3, B3) are routed through the upper plane z=+h, and edges
    belonging to layer B (terms A1, B1, B2) are routed through the lower
    plane z=-h.

    Each edge is a 4-point 3D polyline:
        base(source) -> lift to routing plane -> traverse -> descend to base(target)

    This ensures:
      - Every vertex has ONE set of (x, y) coordinates (the base plane),
      - same-plane edges interact via their in-plane distances,
      - cross-plane edges are separated by 2h vertically.
    """

    # Layer A terms route through z=+h; layer B through z=-h.
    LAYER_A_TERMS = frozenset({'A2', 'A3', 'B3'})
    LAYER_B_TERMS = frozenset({'A1', 'B1', 'B2'})

    def __init__(self, spec: BBCodeSpec, h: float = 1.0):
        self.spec = spec
        self.h = h
        # Build layer subgraphs (for structural assertions).
        self.layer_a = self._build_layer_graph('A')
        self.layer_b = self._build_layer_graph('B')
        # Verify each layer is individually planar.
        for layer_name, lg in [('A', self.layer_a), ('B', self.layer_b)]:
            for comp in nx.connected_components(lg):
                sg = lg.subgraph(comp)
                if sg.number_of_edges() == 0:
                    continue
                ok, _ = nx.check_planarity(sg)
                if not ok:
                    raise ValueError(f'Layer {layer_name} component is not planar.')
        # Compute a single base-plane layout for ALL vertices using the
        # full Tanner graph (union of both layers).
        full_graph = nx.compose(self.layer_a, self.layer_b)
        pos = nx.spring_layout(full_graph, seed=42, iterations=200)
        self.base_coords: dict[tuple[str, int], Point2] = {
            node: (float(xy[0]), float(xy[1])) for node, xy in pos.items()
        }

    def _build_layer_graph(self, layer: str) -> nx.Graph:
        g = nx.Graph()
        for reg in ('X', 'L', 'R', 'Z'):
            for i in range(self.spec.half):
                g.add_node((reg, i))
        if layer == 'A':
            a_terms = self.spec.A_terms[1:]   # A2, A3
            b_terms = (self.spec.B_terms[2],)  # B3
        elif layer == 'B':
            a_terms = (self.spec.A_terms[0],)  # A1
            b_terms = self.spec.B_terms[:2]    # B1, B2
        else:
            raise ValueError(layer)
        for i in range(self.spec.half):
            x = ('X', i)
            z = ('Z', i)
            for t in a_terms:
                g.add_edge(x, ('L', self.spec.term_apply(i, t)))
                g.add_edge(z, ('R', self.spec.term_apply_T(i, t)))
            for t in b_terms:
                g.add_edge(x, ('R', self.spec.term_apply(i, t)))
                g.add_edge(z, ('L', self.spec.term_apply_T(i, t)))
        return g

    def _layer_z(self, term_name: str) -> float:
        """Return the z-coordinate of the routing plane for the given term."""
        if term_name in self.LAYER_A_TERMS:
            return +self.h
        if term_name in self.LAYER_B_TERMS:
            return -self.h
        raise ValueError(f'Unknown term {term_name!r}')

    def routing_geometry(
        self,
        *,
        control_reg: str,
        target_reg: str,
        term_name: str,
        term: Shift,
        transpose: bool = False,
        name: str | None = None,
    ) -> RoutingGeometry:
        z = self._layer_z(term_name)
        out: dict[tuple[str, int, str, int], Polyline] = {}
        for i in range(self.spec.half):
            tgt = self.spec.mapped_target_index(i, term, transpose, target_reg)
            u = (control_reg, i)
            v = (target_reg, tgt)
            pu = self.base_coords[u]
            pv = self.base_coords[v]
            # 4-point polyline: base -> lift -> traverse -> descend
            out[(u[0], u[1], v[0], v[1])] = [
                (pu[0], pu[1], 0.0),  # source at base plane
                (pu[0], pu[1], z),     # lift to routing plane
                (pv[0], pv[1], z),     # traverse in routing plane
                (pv[0], pv[1], 0.0),  # descend to base plane
            ]
        return RoutingGeometry(name or f'ibm_{term_name}', out)

class IBMToricBiplanarEmbedding:
    """Bounded-thickness IBM-inspired biplanar embedding with common base plane.

    All qubits are placed on a common toric base plane. Edges in layer A
    (A2, A3, B3) route through the upper routing region and edges in layer B
    (A1, B1, B2) through the lower routing region. To realize a simple routed
    embedding suitable for the crossing-set formalism, each simultaneously
    routed edge is assigned an infinitesimal lane offset inside its parent layer.

    This yields a crossing-free bounded-thickness realization of the IBM
    G_A/G_B decomposition while keeping all vertices on a common base plane and
    preserving the relevant same-round separation structure.
    """

    LAYER_A_TERMS = frozenset({'A2', 'A3', 'B3'})
    LAYER_B_TERMS = frozenset({'A1', 'B1', 'B2'})

    def __init__(self, spec: BBCodeSpec, h: float = 1.0, lane_eps: float | None = None):
        self.spec = spec
        self.h = float(h)
        self.lane_eps = float(lane_eps) if lane_eps is not None else self.h * 1e-3 / max(1, spec.half)
        self.base_coords: dict[tuple[str, int], Point2] = {}
        for i in range(spec.half):
            a, b = spec.ab(i)
            self.base_coords[('L', i)] = (2.0 * a, 2.0 * b)
            self.base_coords[('X', i)] = (2.0 * a + 1.0, 2.0 * b)
            self.base_coords[('Z', i)] = (2.0 * a, 2.0 * b + 1.0)
            self.base_coords[('R', i)] = (2.0 * a + 1.0, 2.0 * b + 1.0)

    def params_dict(self) -> dict:
        """Serializable parameter dictionary for provenance and hashing."""
        return {
            'code': self.spec.name,
            'embedding_type': 'ibm_biplanar',
            'h': self.h,
            'lane_eps': self.lane_eps,
            'layer_A': sorted(self.LAYER_A_TERMS),
            'layer_B': sorted(self.LAYER_B_TERMS),
        }

    def params_hash(self) -> str:
        """Stable hash of embedding parameters."""
        import hashlib
        return hashlib.sha256(json.dumps(self.params_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def _layer_sign(self, term_name: str) -> int:
        if term_name in self.LAYER_A_TERMS:
            return +1
        if term_name in self.LAYER_B_TERMS:
            return -1
        raise ValueError(f'Unknown term {term_name!r}')

    def _edge_lane_z(self, term_name: str, source_idx: int) -> float:
        sign = self._layer_sign(term_name)
        # Edge-specific infinitesimal lane separation inside the parent layer.
        return sign * self.h + sign * self.lane_eps * (source_idx + 1)

    def _edge_ports(self, term_name: str, source_idx: int, pu: Point2, pv: Point2) -> tuple[Point2, Point2]:
        # Small local offsets inside disjoint neighborhoods around each vertex.
        # These avoid accidental intersections between another edge's in-layer
        # segment and a vertical access line above a different vertex.
        import math
        family_order = {'A1':0,'A2':1,'A3':2,'B1':3,'B2':4,'B3':5}[term_name]
        angle = 2 * math.pi * ((source_idx + 0.37 * family_order) / max(1, self.spec.half))
        eps = min(0.15, 0.2 * min(1.0, self.h))
        du = (eps * math.cos(angle), eps * math.sin(angle))
        dv = (-du[0], -du[1])
        return (pu[0] + du[0], pu[1] + du[1]), (pv[0] + dv[0], pv[1] + dv[1])

    def routing_geometry(
        self,
        *,
        control_reg: str,
        target_reg: str,
        term_name: str,
        term: Shift,
        transpose: bool = False,
        name: str | None = None,
    ) -> RoutingGeometry:
        out: dict[tuple[str, int, str, int], Polyline] = {}
        for i in range(self.spec.half):
            tgt = self.spec.mapped_target_index(i, term, transpose, target_reg)
            u = (control_reg, i)
            v = (target_reg, tgt)
            pu = self.base_coords[u]
            pv = self.base_coords[v]
            z = self._edge_lane_z(term_name, i)
            port_u, port_v = self._edge_ports(term_name, i, pu, pv)
            out[(u[0], u[1], v[0], v[1])] = [
                (pu[0], pu[1], 0.0),
                (port_u[0], port_u[1], 0.0),
                (port_u[0], port_u[1], z),
                (port_v[0], port_v[1], z),
                (port_v[0], port_v[1], 0.0),
                (pv[0], pv[1], 0.0),
            ]
        return RoutingGeometry(name or f'toric_{term_name}', out)

# Keep the publication-facing name bound to the bounded-thickness implementation.
IBMBiplanarEmbedding = IBMToricBiplanarEmbedding


# ═══════════════════════════════════════════════════════════════════════
#  Logical-aware single-layer embedding  (Chapter 8 of the workbook)
# ═══════════════════════════════════════════════════════════════════════
#
#  Column-permutation embeddings (Definition 8.1) with row orderings
#  optimized via the theorem-facing objectives J_× and J_κ from
#  Definitions 8.3 and 8.5.  The optimizer uses SA warm start
#  (Remark 8.6) followed by deterministic swap descent (Definition 8.4,
#  Theorem 8.4: guaranteed finite termination at a 2-swap local min).


def _reference_family(spec: BBCodeSpec) -> list[set[int]]:
    """Reference family R_X (Definition 8.2).

    Delegates to algebra.enumerate_pure_L_minwt_logicals for the
    algebraic enumeration.  For BB72 this yields 36 weight-6 supports.
    """
    return enumerate_pure_L_minwt_logicals(spec)


def _support_crossing_graph_edges(
    spec: BBCodeSpec,
    sigma_L: list[int],
    sigma_Z: list[int],
    support: set[int],
    B_terms: list[Shift],
) -> set[tuple[int, int]]:
    """Build the edge set of the support-induced crossing graph C_φ^X[S].

    An edge {α, β} exists iff the q(L)→q(Z) segments for α, β cross
    in at least one B-round (Proposition 7, workbook):
        (σ_L[α] - σ_L[β]) × (σ_Z[B_r·α] - σ_Z[B_r·β]) < 0.
    """
    edges: set[tuple[int, int]] = set()
    support_list = sorted(support)
    for a_idx in range(len(support_list)):
        alpha = support_list[a_idx]
        for b_idx in range(a_idx + 1, len(support_list)):
            beta = support_list[b_idx]
            d_source = sigma_L[alpha] - sigma_L[beta]
            for term in B_terms:
                t_alpha = spec.term_apply(alpha, term)
                t_beta = spec.term_apply(beta, term)
                d_target = sigma_Z[t_alpha] - sigma_Z[t_beta]
                if d_source * d_target < 0:
                    edges.add((alpha, beta))
                    break  # one crossing round suffices
    return edges


def _matching_number(vertices: set[int], edges: set[tuple[int, int]]) -> int:
    """Maximum matching number of a general graph (Definition 31, workbook).

    Uses networkx max_weight_matching (Edmonds' blossom algorithm),
    which is exact for general (non-bipartite) graphs.  The previous
    augmenting-path implementation was only correct for bipartite graphs
    and could undercount on graphs with odd cycles.
    """
    if not edges:
        return 0
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)


def _J_cross(
    spec: BBCodeSpec,
    sigma_L: list[int],
    sigma_Z: list[int],
    reference_family: list[set[int]],
    B_terms: list[Shift],
) -> int:
    """Crossing-local objective J_× (Definition 8.3):
        J_×(σ_L, σ_Z; R_X) = max_{L ∈ R_X} ν_φ^X(L).
    """
    worst = 0
    for support in reference_family:
        edges = _support_crossing_graph_edges(spec, sigma_L, sigma_Z, support, B_terms)
        nu = _matching_number(support, edges)
        worst = max(worst, nu)
    return worst


def _segment_distance_2d(x1: float, y1: float, x2: float, y2: float,
                         x3: float, y3: float, x4: float, y4: float) -> float:
    """Exact minimum distance between two 2D line segments.

    Delegates to geometry._segment_distance for correctness, via
    3D points at z=0.  Caches the imported function for speed.
    """
    return _seg_dist_3d(
        (x1, y1, 0.0), (x2, y2, 0.0),
        (x3, y3, 0.0), (x4, y4, 0.0),
    )


# Import once at module level for speed
from .geometry import _segment_distance as _seg_dist_3d


def _precompute_term_maps(spec: BBCodeSpec, B_terms: list[Shift]) -> list[list[int]]:
    """Precompute term_apply(i, term) for all i and all B-terms."""
    return [[spec.term_apply(i, term) for i in range(spec.half)] for term in B_terms]


def _J_exposure(
    spec: BBCodeSpec,
    sigma_L: list[int],
    sigma_Z: list[int],
    reference_family: list[set[int]],
    B_terms: list[Shift],
    scale_y: float,
    x_L: float,
    x_Z: float,
    tau: float,
    J0: float,
    alpha: float,
    r0: float,
    _term_maps: list[list[int]] | None = None,
) -> float:
    """Distance-decay objective J_κ (Definition 8.5):
        J_κ(σ_L, σ_Z; R_X) = max_{L ∈ R_X} Expose_φ^X(L).

    Uses the regularized power-law kernel κ(d) = (1 + d/r0)^{-α}
    and the exact twirled pair probability p(d) = sin²(τ J0 κ(d)).

    The segment distance uses the exact 3D closest-approach formula
    from geometry.py, consistent with the Monte Carlo geometry engine.
    For the four-column straight-segment architecture this is exact.

    The _term_maps argument caches precomputed term_apply mappings
    across SA iterations for ~5× speedup.
    """
    if _term_maps is None:
        _term_maps = _precompute_term_maps(spec, B_terms)

    # Precompute support pair lists once
    worst = 0.0
    for support in reference_family:
        exposure = 0.0
        support_list = sorted(support)
        n_s = len(support_list)
        for a_idx in range(n_s):
            a = support_list[a_idx]
            y1a = scale_y * sigma_L[a]
            for b_idx in range(a_idx + 1, n_s):
                b = support_list[b_idx]
                y1b = scale_y * sigma_L[b]
                for tmap in _term_maps:
                    y2a = scale_y * sigma_Z[tmap[a]]
                    y2b = scale_y * sigma_Z[tmap[b]]

                    d = _segment_distance_2d(
                        x_L, y1a, x_Z, y2a,
                        x_L, y1b, x_Z, y2b,
                    )

                    kappa = (1.0 + d / r0) ** (-alpha)
                    theta = tau * J0 * kappa
                    p = math.sin(theta) ** 2
                    exposure += p
        worst = max(worst, exposure)
    return worst


def _swap_descent(
    spec: BBCodeSpec,
    sigma_L: list[int],
    sigma_Z: list[int],
    cost_fn,
) -> tuple[list[int], list[int]]:
    """Deterministic swap descent (Definition 8.4, Theorem 8.4).

    Repeatedly tries all 2-swaps in σ_L and σ_Z, accepts the first
    strictly improving swap, and halts when no improving swap exists.
    Guaranteed to terminate at a 2-swap local minimum.
    """
    n = len(sigma_L)
    current_cost = cost_fn(sigma_L, sigma_Z)
    improved = True
    while improved:
        improved = False
        # Try all swaps in σ_L
        for i in range(n):
            for j in range(i + 1, n):
                sigma_L[i], sigma_L[j] = sigma_L[j], sigma_L[i]
                new_cost = cost_fn(sigma_L, sigma_Z)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break  # restart scan
                sigma_L[i], sigma_L[j] = sigma_L[j], sigma_L[i]
            if improved:
                break
        if improved:
            continue
        # Try all swaps in σ_Z
        for i in range(n):
            for j in range(i + 1, n):
                sigma_Z[i], sigma_Z[j] = sigma_Z[j], sigma_Z[i]
                new_cost = cost_fn(sigma_L, sigma_Z)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                sigma_Z[i], sigma_Z[j] = sigma_Z[j], sigma_Z[i]
            if improved:
                break
    return sigma_L, sigma_Z


def optimize_row_order(
    spec: BBCodeSpec,
    *,
    objective: str = 'crossing',
    reference_family: list[set[int]] | None = None,
    n_sa_iter: int = 100_000,
    n_restarts: int = 20,
    seed: int = 42,
    tau: float = 1.0,
    J0: float = 0.08,
    alpha: float = 3.0,
    r0: float = 1.0,
    scale_y: float = 1.0,
    x_L: float = 1.0,
    x_Z: float = 3.0,
) -> tuple[list[int], list[int], dict]:
    """Optimize row permutations using Chapter 8 objectives.

    Phase 1: Multiple independent SA warm starts with n_sa_iter
             iterations each. The restart count is a numerical policy,
             not part of the theorem statement.
    Phase 2: Deterministic swap descent (Definition 8.4 / Theorem 8.4)
             on the best SA result, to a certified 2-swap local minimum.

    Args:
        objective: 'crossing' for J_× (Definition 8.3) or
                   'exposure' for J_κ (Definition 8.5).
        reference_family: Explicit list of support sets to optimize over.
            If None, uses the full algebraically-enumerated pure-L family.
        n_restarts: Number of independent SA restarts.

    Returns:
        (sigma_L, sigma_Z, info) where info contains optimization metrics
        including reference_family_source for provenance.
    """
    n = spec.half
    B_terms = list(spec.B_terms)
    if reference_family is None:
        reference_family = _reference_family(spec)
        family_source = f'algebraic_pure_L_{spec.name}'
    else:
        family_source = f'explicit_{len(reference_family)}_supports'

    if objective == 'crossing':
        def cost_fn(sL, sZ):
            return _J_cross(spec, sL, sZ, reference_family, B_terms)
    elif objective == 'exposure':
        term_maps = _precompute_term_maps(spec, B_terms)
        def cost_fn(sL, sZ):
            return _J_exposure(spec, sL, sZ, reference_family, B_terms,
                               scale_y, x_L, x_Z, tau, J0, alpha, r0,
                               _term_maps=term_maps)
    else:
        raise ValueError(f"Unknown objective {objective!r}; expected 'crossing' or 'exposure'.")

    identity_L = list(range(n))
    identity_Z = list(range(n))
    initial_cost = cost_fn(identity_L, identity_Z)

    # ── Phase 1: Multiple SA warm starts ──
    global_best_L, global_best_Z = identity_L[:], identity_Z[:]
    global_best_cost = initial_cost

    for restart in range(n_restarts):
        rng = random.Random(seed + restart)
        sigma_L = list(range(n))
        sigma_Z = list(range(n))
        # Shuffle initial state for restarts > 0
        if restart > 0:
            rng.shuffle(sigma_L)
            rng.shuffle(sigma_Z)

        current_cost = cost_fn(sigma_L, sigma_Z)
        best_L, best_Z, best_cost = sigma_L[:], sigma_Z[:], current_cost

        for step in range(n_sa_iter):
            T = max(0.01, 1.0 - step / n_sa_iter)
            sigma = sigma_L if rng.random() < 0.5 else sigma_Z
            i, j = rng.sample(range(n), 2)
            sigma[i], sigma[j] = sigma[j], sigma[i]

            new_cost = cost_fn(sigma_L, sigma_Z)
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_L, best_Z, best_cost = sigma_L[:], sigma_Z[:], new_cost
            else:
                sigma[i], sigma[j] = sigma[j], sigma[i]

        if best_cost < global_best_cost:
            global_best_L, global_best_Z = best_L[:], best_Z[:]
            global_best_cost = best_cost

    sa_cost = global_best_cost

    # ── Phase 2: Deterministic swap descent (Theorem 8.4) ──
    sigma_L, sigma_Z = _swap_descent(
        spec, global_best_L[:], global_best_Z[:], cost_fn,
    )
    final_cost = cost_fn(sigma_L, sigma_Z)

    info = {
        'objective': objective,
        'initial_cost': initial_cost,
        'sa_cost': sa_cost,
        'final_cost': final_cost,
        'n_restarts': n_restarts,
        'reference_family_source': family_source,
        'reference_family_size': len(reference_family),
        'reference_family_hash': reference_family_hash(reference_family),
        'reference_supports': [sorted(s) for s in reference_family],
    }
    return sigma_L, sigma_Z, info


class LogicalAwareColumnEmbedding:
    """Column-permutation embedding optimized via Chapter 8 objectives.

    Implements Definition 8.1 (column-permutation embedding) with row
    orderings found by SA warm start + deterministic swap descent
    (Definition 8.4, Theorem 8.4).

    The objective is either:
    - J_× = max_{L ∈ R_X} ν_φ^X(L)  (crossing-local, Definition 8.3)
    - J_κ = max_{L ∈ R_X} Expose_φ^X(L)  (distance-decay, Definition 8.5)

    where R_X is the reference family of all minimum-weight X-type
    logicals (Definition 8.2).

    If the optimized objective value is lower than that of the monomial
    embedding, the corresponding theorem-facing support-level bound improves
    on the chosen reference family. No direct finite-coupling LER guarantee is
    implied by this class alone.
    """

    def __init__(
        self,
        spec: BBCodeSpec,
        *,
        x_positions: Mapping[str, float] | None = None,
        scale_y: float = 1.0,
        objective: str = 'exposure',
        n_sa_iter: int = 100_000,
        seed: int = 42,
        tau: float = 1.0,
        J0: float = 0.08,
        alpha: float = 3.0,
        r0: float = 1.0,
    ):
        self.spec = spec
        self.x_positions = dict(x_positions or {'X': 0.0, 'L': 1.0, 'R': 2.0, 'Z': 3.0})
        self.scale_y = scale_y

        # Optimize row orderings using Chapter 8 objectives
        self.sigma_L, self.sigma_Z, self.opt_info = optimize_row_order(
            spec,
            objective=objective,
            n_sa_iter=n_sa_iter,
            seed=seed,
            tau=tau, J0=J0, alpha=alpha, r0=r0,
            scale_y=scale_y,
            x_L=self.x_positions['L'],
            x_Z=self.x_positions['Z'],
        )

        # Build points using optimized row positions (Definition 8.1)
        self.points: dict[tuple[str, int], Point3] = {}
        for reg, sigma in [('X', self.sigma_L), ('L', self.sigma_L),
                           ('R', self.sigma_Z), ('Z', self.sigma_Z)]:
            for i in range(spec.half):
                self.points[(reg, i)] = (
                    self.x_positions[reg],
                    self.scale_y * sigma[i],
                    0.0,
                )

    def routing_geometry(
        self,
        *,
        control_reg: str,
        target_reg: str,
        term: Shift,
        transpose: bool = False,
        name: str | None = None,
        term_name: str | None = None,
    ) -> RoutingGeometry:
        out: dict[tuple[str, int, str, int], Polyline] = {}
        for i in range(self.spec.half):
            tgt = self.spec.mapped_target_index(i, term, transpose, target_reg)
            u = (control_reg, i)
            v = (target_reg, tgt)
            out[(u[0], u[1], v[0], v[1])] = [self.points[u], self.points[v]]
        return RoutingGeometry(name or term_name or 'logical_aware', out)


class FixedPermutationColumnEmbedding:
    """Column-permutation embedding with frozen (pre-computed) permutations.

    Loads sigma_L, sigma_Z from a JSON config file instead of running SA.
    This ensures reproducible experiment construction without re-optimization.
    """

    def __init__(self, spec: BBCodeSpec, config_path: str | Path):
        payload = json.loads(Path(config_path).read_text())
        self.spec = spec
        self.sigma_L: list[int] = payload['sigma_L']
        self.sigma_Z: list[int] = payload['sigma_Z']
        self.x_positions = payload.get('x_positions', {'X': 0.0, 'L': 1.0, 'R': 2.0, 'Z': 3.0})
        scale_y = payload.get('scale_y', 1.0)

        assert len(self.sigma_L) == spec.half
        assert len(self.sigma_Z) == spec.half

        self.points: dict[tuple[str, int], Point3] = {}
        for reg, sigma in [('X', self.sigma_L), ('L', self.sigma_L),
                           ('R', self.sigma_Z), ('Z', self.sigma_Z)]:
            for i in range(spec.half):
                self.points[(reg, i)] = (
                    self.x_positions[reg],
                    scale_y * sigma[i],
                    0.0,
                )

    def routing_geometry(
        self,
        *,
        control_reg: str,
        target_reg: str,
        term: 'Shift',
        transpose: bool = False,
        name: str | None = None,
        term_name: str | None = None,
    ) -> RoutingGeometry:
        out: dict[tuple[str, int, str, int], Polyline] = {}
        for i in range(self.spec.half):
            tgt = self.spec.mapped_target_index(i, term, transpose, target_reg)
            u = (control_reg, i)
            v = (target_reg, tgt)
            out[(u[0], u[1], v[0], v[1])] = [self.points[u], self.points[v]]
        return RoutingGeometry(name or term_name or 'fixed_la', out)


class JsonPolylineEmbedding:
    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text())
        self.name: str = payload.get('name', 'json')
        self.edge_polylines: dict[tuple[str, int, str, int], Polyline] = {}
        for item in payload['edges']:
            key = (item['u_reg'], int(item['u_idx']), item['v_reg'], int(item['v_idx']))
            self.edge_polylines[key] = [tuple(map(float, p)) for p in item['polyline']]

    def routing_geometry(self) -> RoutingGeometry:
        return RoutingGeometry(self.name, self.edge_polylines)
