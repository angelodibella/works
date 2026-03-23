# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
"""GF(2) linear algebra and logical-family enumeration.

Implements the exact pure-q(L) quotient:

    ker(B^T) / T_L

where T_L = { λA : λB = 0 } is the L-projection of the stabilizer
subspace restricted to vectors whose R-component vanishes.

This separates scientific assumptions (which logicals are in the
reference family) from the optimizer, so the family is explicit
and auditable.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .bbcode import BBCodeSpec


def nullspace_gf2(M: np.ndarray) -> np.ndarray:
    """Basis for the GF(2) null space of M (rows of result)."""
    m, n = M.shape
    aug = np.hstack([M.T.copy(), np.eye(n, dtype=np.uint8)])
    pivot_row = 0
    for col in range(m):
        found = None
        for row in range(pivot_row, n):
            if aug[row, col]:
                found = row
                break
        if found is None:
            continue
        if found != pivot_row:
            aug[[pivot_row, found]] = aug[[found, pivot_row]]
        for row in range(n):
            if row != pivot_row and aug[row, col]:
                aug[row] = (aug[row] + aug[pivot_row]) % 2
        pivot_row += 1
    basis = []
    for row in range(n):
        if np.all(aug[row, :m] == 0) and np.any(aug[row, m:]):
            basis.append(aug[row, m:])
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, n), dtype=np.uint8)


def rowspace_gf2(M: np.ndarray) -> np.ndarray:
    """Row-echelon basis for the GF(2) row space of M."""
    R = M.copy()
    m, n = R.shape
    pivot_row = 0
    for col in range(n):
        found = None
        for row in range(pivot_row, m):
            if R[row, col]:
                found = row
                break
        if found is None:
            continue
        if found != pivot_row:
            R[[pivot_row, found]] = R[[found, pivot_row]]
        for row in range(m):
            if row != pivot_row and R[row, col]:
                R[row] = (R[row] + R[pivot_row]) % 2
        pivot_row += 1
    return R[:pivot_row]


# ═══════════════════════════════════════════════════════════════════════
#  Pure-q(L) logical-family algebra
# ═══════════════════════════════════════════════════════════════════════

def ker_BT_basis(spec: BBCodeSpec) -> np.ndarray:
    """Basis of ker(B^T) over GF(2).

    These are vectors u with B^T u = 0, i.e. X(u, 0) commutes with
    all Z-stabilizers.  Rows of the returned matrix.
    """
    B = spec.polynomial_matrix(spec.B_terms)
    return nullspace_gf2(B.T)


def TL_basis(spec: BBCodeSpec) -> np.ndarray:
    """Basis of the trivial subspace T_L = { λA : λB = 0 } over GF(2).

    T_L is the L-projection of stabilizer generators X(λA, λB) restricted
    to those with λB = 0.  Equivalently, T_L = im(A | ker(B)), the image
    of A acting on the left null space of B.

    Note: T_L ⊆ rowspace(A), with equality only when ker(B) = F_2^{half}.
    For BB codes, dim(T_L) < dim(rowspace(A)) in general.
    """
    A = spec.polynomial_matrix(spec.A_terms)
    B = spec.polynomial_matrix(spec.B_terms)
    # ker(B) as row vectors: λ with λB = 0, i.e. B^T λ^T = 0
    ker_B = nullspace_gf2(B.T)
    if len(ker_B) == 0:
        return np.zeros((0, spec.half), dtype=np.uint8)
    # T_L = { λA : λ ∈ ker(B) }
    TL_generators = (ker_B @ A) % 2
    return rowspace_gf2(TL_generators)


def pure_L_quotient_dimension(spec: BBCodeSpec) -> tuple[int, int, int]:
    """Dimensions of the pure-q(L) logical quotient.

    Returns (dim_ker_BT, dim_TL, quotient_dim) where
    quotient_dim = dim(ker(B^T)) - dim(T_L) is the dimension of
    the pure-q(L) X-logical space ker(B^T) / T_L.
    """
    dim_ker = len(ker_BT_basis(spec))
    dim_TL = len(TL_basis(spec))
    return dim_ker, dim_TL, dim_ker - dim_TL


def enumerate_pure_L_minwt_logicals(spec: BBCodeSpec) -> list[set[int]]:
    """Enumerate all minimum-weight pure-q(L) X-logical supports.

    A pure-q(L) X-logical is X(u, 0) where:
      1. B^T u = 0 (mod 2) — commutes with all Z-stabilizers
      2. u ∉ T_L — nontrivial modulo stabilizers, where
         T_L = { λA : λB = 0 } is the exact pure-q(L) trivial subspace
      3. wt(u) is minimized

    The nontriviality test uses the exact quotient ker(B^T) / T_L,
    NOT the looser rowspace(A) test.  For BB codes these give the same
    minimum-weight family, but the mathematical statement is sharper.
    """
    half = spec.half
    ker_BT = ker_BT_basis(spec)
    TL = TL_basis(spec)
    dim_ker = len(ker_BT)

    min_wt = half + 1
    all_supports: list[frozenset[int]] = []

    for mask in range(1, 2**dim_ker):
        u = np.zeros(half, dtype=np.uint8)
        for bit in range(dim_ker):
            if mask & (1 << bit):
                u = (u + ker_BT[bit]) % 2
        wt = int(u.sum())
        if wt == 0 or wt > min_wt:
            continue

        # Nontriviality: u ∉ T_L
        aug = np.vstack([TL, u.reshape(1, -1)])
        if len(rowspace_gf2(aug)) == len(TL):
            continue  # u is in T_L, hence trivial

        support = frozenset(int(i) for i in range(half) if u[i])
        if wt < min_wt:
            min_wt = wt
            all_supports = [support]
        elif wt == min_wt:
            all_supports.append(support)

    unique = sorted(set(all_supports), key=lambda s: sorted(s))
    return [set(s) for s in unique]


def reference_family_hash(family: list[set[int]]) -> str:
    """Stable hash for a reference family (order-independent)."""
    canon = str(sorted(tuple(sorted(s)) for s in family))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
