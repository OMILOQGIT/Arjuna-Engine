"""Z2 qubit tapering (Bravyi, Gambetta, Mezzacapo, Temme 2017), Z-type symmetries.

1. Find all Z-strings commuting with every Hamiltonian term (GF(2) null space of the
   terms' X-parts).
2. RREF -> one pivot qubit per generator; Clifford U = (X_q + Z^z)/sqrt(2) maps Z^z -> X_q.
3. Replace X_q by its eigenvalue in the Hartree-Fock sector and drop qubit q.
"""
from __future__ import annotations

import numpy as np

from .pauli import commutes, mul_terms, popcount


def _gf2_nullspace(rows: list[int], n: int) -> list[int]:
    M = np.array([[(r >> q) & 1 for q in range(n)] for r in rows], dtype=np.uint8) if rows else np.zeros((0, n), np.uint8)
    M = M.copy()
    piv_cols, r = [], 0
    for c in range(n):
        if r >= M.shape[0]:
            break
        pr = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if pr is None:
            continue
        M[[r, pr]] = M[[pr, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv_cols.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv_cols]
    basis = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv_cols):
            v[pc] = M[i, f]
        basis.append(int(sum(int(b) << q for q, b in enumerate(v))))
    return basis


def _rref(vecs: list[int], n: int):
    vecs = list(vecs)
    pivots = []
    r = 0
    for c in range(n):
        pr = next((i for i in range(r, len(vecs)) if (vecs[i] >> c) & 1), None)
        if pr is None:
            continue
        vecs[r], vecs[pr] = vecs[pr], vecs[r]
        for i in range(len(vecs)):
            if i != r and (vecs[i] >> c) & 1:
                vecs[i] ^= vecs[r]
        pivots.append(c)
        r += 1
        if r == len(vecs):
            break
    return vecs[:r], pivots


class Tapering:
    def __init__(self, H: dict, n: int, ref_bits: int):
        self.n = n
        xs = list({x for (x, _z) in H})
        syms = _gf2_nullspace(xs, n)
        self.generators, self.pivots = _rref(syms, n)
        self.eigs = [(-1) ** popcount(z & ref_bits) for z in self.generators]
        self.keep = [q for q in range(n) if q not in self.pivots]
        self.n_tapered = len(self.keep)
        self.ref_bits = self._compress(ref_bits)

    def _compress(self, v: int) -> int:
        return sum(((v >> q) & 1) << i for i, q in enumerate(self.keep))

    def commutes_with_symmetries(self, op: dict) -> bool:
        return all(commutes(x, z, 0, s) for (x, z) in op for s in self.generators)

    def taper(self, op: dict, tol=1e-12) -> dict:
        out: dict = {}
        for (x, z), c in op.items():
            for s, q in zip(self.generators, self.pivots):
                if (z >> q) & 1:  # anticommutes with X_q -> P . Z^s . X_q
                    x, z, ph1 = mul_terms(x, z, 0, s)
                    x, z, ph2 = mul_terms(x, z, 1 << q, 0)
                    c = c * ph1 * ph2
            for q, lam in zip(self.pivots, self.eigs):
                assert not (z >> q) & 1, "pivot qubit still carries Z/Y after Clifford"
                if (x >> q) & 1:
                    c = c * lam
            key = (self._compress(x), self._compress(z))
            out[key] = out.get(key, 0) + c
        return {k: v for k, v in out.items() if abs(v) > tol}
