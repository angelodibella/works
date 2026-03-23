# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from collections.abc import Callable, Sequence
import itertools
import math
import numpy as np

Point3 = tuple[float, float, float]
Polyline = Sequence[Point3]
Kernel = Callable[[float], float]


def _segment_distance(p1, p2, q1, q2) -> float:
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    q1 = np.array(q1, float)
    q2 = np.array(q2, float)
    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    denom = a * c - b * b
    eps = 1e-12
    if denom < eps:
        s = 0.0
        t = e / c if c > eps else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    s = min(1, max(0, s))
    t = min(1, max(0, t))
    cp = p1 + s * u
    cq = q1 + t * v
    return float(np.linalg.norm(cp - cq))


def polyline_distance(poly1: Polyline, poly2: Polyline) -> float:
    best = math.inf
    for a, b in zip(poly1, poly1[1:]):
        for c, d in zip(poly2, poly2[1:]):
            best = min(best, _segment_distance(a, b, c, d))
    return best


def crossing_kernel(d: float) -> float:
    return 1.0 if abs(d) < 1e-12 else 0.0


def regularized_power_law_kernel(alpha: float, r0: float) -> Kernel:
    def k(d: float) -> float:
        return (1 + d / r0) ** (-alpha)
    return k


def exponential_kernel(xi: float) -> Kernel:
    def k(d: float) -> float:
        return math.exp(-d / xi)
    return k


def exact_twirled_pair_probability(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    return math.sin(tau * J0 * kernel(d)) ** 2


def weak_pair_probability(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    x = tau * J0 * kernel(d)
    return x * x


def pair_amplitude(d: float, J0: float, kernel: Kernel) -> float:
    """Microscopic coherent pair-coupling amplitude J0 * kappa(d)."""
    return J0 * kernel(d)


def pair_location_strength(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    """Dimensionless location strength tau * J0 * kappa(d), matching the AKP-style criterion."""
    return tau * J0 * kernel(d)


def pairwise_round_coefficients(
    edge_polylines: dict[tuple[str, int, str, int], Polyline],
    *,
    tau: float,
    J0: float,
    kernel: Kernel,
    use_weak_limit: bool = False,
) -> dict[tuple, float]:
    prob = weak_pair_probability if use_weak_limit else exact_twirled_pair_probability
    out: dict[tuple, float] = {}
    items = list(edge_polylines.items())
    for (e1, p1), (e2, p2) in itertools.combinations(items, 2):
        verts = {(e1[0], e1[1]), (e1[2], e1[3])} & {(e2[0], e2[1]), (e2[2], e2[3])}
        if verts:
            continue
        d = polyline_distance(p1, p2)
        out[(e1, e2)] = prob(d, tau=tau, J0=J0, kernel=kernel)
    return out


def pairwise_round_amplitudes(
    edge_polylines: dict[tuple[str, int, str, int], Polyline],
    *,
    J0: float,
    kernel: Kernel,
) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    items = list(edge_polylines.items())
    for (e1, p1), (e2, p2) in itertools.combinations(items, 2):
        verts = {(e1[0], e1[1]), (e1[2], e1[3])} & {(e2[0], e2[1]), (e2[2], e2[3])}
        if verts:
            continue
        d = polyline_distance(p1, p2)
        out[(e1, e2)] = pair_amplitude(d, J0=J0, kernel=kernel)
    return out


def pairwise_round_location_strengths(
    edge_polylines: dict[tuple[str, int, str, int], Polyline],
    *,
    tau: float,
    J0: float,
    kernel: Kernel,
) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    items = list(edge_polylines.items())
    for (e1, p1), (e2, p2) in itertools.combinations(items, 2):
        verts = {(e1[0], e1[1]), (e1[2], e1[3])} & {(e2[0], e2[1]), (e2[2], e2[3])}
        if verts:
            continue
        d = polyline_distance(p1, p2)
        out[(e1, e2)] = pair_location_strength(d, tau=tau, J0=J0, kernel=kernel)
    return out


def weighted_exposure_on_support(support_labels: Sequence[int], round_coeffs) -> float:
    S = set(support_labels)
    total = 0.0
    for coeffs in round_coeffs:
        for (e1, e2), p in coeffs.items():
            if e1[0] == 'L' and e2[0] == 'L' and e1[1] in S and e2[1] in S:
                total += p
    return total


def count_zero_distance_pairs(edge_polylines: dict) -> int:
    """Count pairs of edges with zero polyline distance (crossings)."""
    count = 0
    items = list(edge_polylines.items())
    for (e1, p1), (e2, p2) in itertools.combinations(items, 2):
        verts = {(e1[0], e1[1]), (e1[2], e1[3])} & {(e2[0], e2[1]), (e2[2], e2[3])}
        if verts:
            continue
        if polyline_distance(p1, p2) < 1e-12:
            count += 1
    return count


def aggregate_edge_metric(coeffs) -> dict:
    """Aggregate a symmetric pairwise coefficient map onto its incident edges."""
    agg: dict = {}
    for (e1, e2), val in coeffs.items():
        agg[e1] = agg.get(e1, 0.0) + val
        agg[e2] = agg.get(e2, 0.0) + val
    return agg


def aggregate_pair_probability(coeffs) -> dict:
    return aggregate_edge_metric(coeffs)


def aggregate_pair_amplitude(coeffs) -> dict:
    return aggregate_edge_metric(coeffs)


def aggregate_location_strength(coeffs) -> dict:
    return aggregate_edge_metric(coeffs)
