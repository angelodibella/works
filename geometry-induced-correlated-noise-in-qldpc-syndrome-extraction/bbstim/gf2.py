# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
import numpy as np


def as_uint8_matrix(a):
    arr = np.array(a, dtype=np.uint8, copy=True)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D GF(2) matrix.")
    return arr % 2


def rref_mod2(a: np.ndarray) -> tuple[np.ndarray, list[int]]:
    m = as_uint8_matrix(a)
    rows, cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for rr in range(r, rows):
            if m[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            m[[r, pivot]] = m[[pivot, r]]
        for rr in range(rows):
            if rr != r and m[rr, c]:
                m[rr, :] ^= m[r, :]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m, pivots


def rank_mod2(a: np.ndarray) -> int:
    return len(rref_mod2(a)[1])


def row_basis_mod2(a: np.ndarray) -> np.ndarray:
    rref, _ = rref_mod2(a)
    rows = [row for row in rref if np.any(row)]
    if not rows:
        return np.zeros((0, a.shape[1]), dtype=np.uint8)
    return np.array(rows, dtype=np.uint8)


def nullspace_mod2(a: np.ndarray) -> np.ndarray:
    a = as_uint8_matrix(a)
    rows, cols = a.shape
    rref, pivots = rref_mod2(a)
    pivset = set(pivots)
    free = [c for c in range(cols) if c not in pivset]
    if not free:
        return np.zeros((0, cols), dtype=np.uint8)
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for r, c in enumerate(pivots):
            v[c] = rref[r, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8)


def extend_basis_mod2(initial_basis: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if initial_basis.size == 0:
        current = np.zeros((0, candidates.shape[1]), dtype=np.uint8)
    else:
        current = row_basis_mod2(initial_basis)
    added = []
    for v in candidates:
        test = np.vstack([current, v]) if current.size else np.array([v], dtype=np.uint8)
        if rank_mod2(test) > (rank_mod2(current) if current.size else 0):
            current = row_basis_mod2(test)
            added.append(v.copy())
    if not added:
        return np.zeros((0, candidates.shape[1]), dtype=np.uint8)
    return np.array(added, dtype=np.uint8)


def logical_basis_css(hx: np.ndarray, hz: np.ndarray):
    row_hx = row_basis_mod2(hx)
    row_hz = row_basis_mod2(hz)
    x_log = extend_basis_mod2(row_hx, nullspace_mod2(hz))
    z_log = extend_basis_mod2(row_hz, nullspace_mod2(hx))
    return x_log, z_log


def hamming_weight(v: np.ndarray) -> int:
    return int(np.sum(np.array(v, dtype=np.uint8) % 2))
