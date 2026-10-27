"""Minimal Pauli algebra + Jordan-Wigner mapping + sparse matrices.

A Pauli term is (x, z) bitmasks with P(x,z) = i^{popcount(x&z)} X^x Z^z, so (1,1) on a
qubit is Y. Operators are dicts {(x, z): complex coeff}. Qubit q <-> bit q.
Spin-orbital ordering (qiskit-style block): alpha 0..n-1, beta n..2n-1.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

_I_POW = np.array([1, 1j, -1, -1j])


def popcount(v: int) -> int:
    return bin(v).count("1")


def mul_terms(x1, z1, x2, z2):
    x3, z3 = x1 ^ x2, z1 ^ z2
    k = popcount(x1 & z1) + popcount(x2 & z2) - popcount(x3 & z3) + 2 * popcount(z1 & x2)
    return x3, z3, _I_POW[k % 4]


def op_mul(A: dict, B: dict, tol=1e-14) -> dict:
    out = defaultdict(complex)
    for (x1, z1), c1 in A.items():
        for (x2, z2), c2 in B.items():
            x3, z3, ph = mul_terms(x1, z1, x2, z2)
            out[(x3, z3)] += c1 * c2 * ph
    return {k: v for k, v in out.items() if abs(v) > tol}


def op_add(*ops, tol=1e-12) -> dict:
    out = defaultdict(complex)
    for o in ops:
        for k, v in o.items():
            out[k] += v
    return {k: v for k, v in out.items() if abs(v) > tol}


def op_scale(A: dict, s) -> dict:
    return {k: v * s for k, v in A.items()}


def commutes(x1, z1, x2, z2) -> bool:
    return (popcount(x1 & z2) + popcount(z1 & x2)) % 2 == 0


def commutator(A: dict, B: dict) -> dict:
    out = defaultdict(complex)
    for (x1, z1), c1 in A.items():
        for (x2, z2), c2 in B.items():
            if not commutes(x1, z1, x2, z2):
                x3, z3, ph = mul_terms(x1, z1, x2, z2)
                out[(x3, z3)] += 2 * c1 * c2 * ph
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def label(x: int, z: int, n: int) -> str:
    """Qiskit-style label (qubit 0 rightmost)."""
    s = []
    for q in range(n):
        xb, zb = (x >> q) & 1, (z >> q) & 1
        s.append("IXZY"[xb + 2 * zb])
    return "".join(reversed(s))


# ------------------------------------------------------------------ Jordan-Wigner
def jw_ladder(p: int, dagger: bool) -> dict:
    zs = (1 << p) - 1  # Z on qubits < p
    xp = 1 << p
    # a_p = Z..Z (X + iY)/2 ; Y = P(x=1,z=1) in our convention
    sgn = -1j if dagger else 1j
    return {(xp, zs): 0.5, (xp, zs | xp): 0.5 * sgn}


def fermion_product(ops: list[tuple[int, bool]]) -> dict:
    out = {(0, 0): 1.0 + 0j}
    for p, d in ops:
        out = op_mul(out, jw_ladder(p, d))
    return out


def spin_orbital_integrals(h, g):
    """Spatial h_pq, (pq|rs) -> spin-orbital h and physicist <pq|rs> (block ordering)."""
    n = h.shape[0]
    N = 2 * n
    hs = np.zeros((N, N))
    hs[:n, :n] = hs[n:, n:] = h
    spin = np.array([0] * n + [1] * n)
    sp_idx = np.arange(N) % n
    # <pq|rs> = (pr|qs) delta(sp,sr) delta(sq,ss)
    G = g[np.ix_(sp_idx, sp_idx, sp_idx, sp_idx)]  # (p r | q s) indexed [p,r,q,s]
    phys = G.transpose(0, 2, 1, 3)                 # [p,q,r,s] = (pr|qs)
    mask = (spin[:, None, None, None] == spin[None, None, :, None]) & \
           (spin[None, :, None, None] == spin[None, None, None, :])
    return hs, phys * mask


def qubit_hamiltonian(h, g, ecore, tol=1e-12) -> dict:
    hs, phys = spin_orbital_integrals(h, g)
    N = hs.shape[0]
    terms = [{(0, 0): complex(ecore)}]
    for p in range(N):
        for q in range(N):
            if abs(hs[p, q]) > tol:
                terms.append(op_scale(fermion_product([(p, True), (q, False)]), hs[p, q]))
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            for r in range(N):
                for s in range(N):
                    if r == s or abs(phys[p, q, r, s]) < tol:
                        continue
                    terms.append(op_scale(fermion_product([(p, True), (q, True), (s, False), (r, False)]),
                                          0.5 * phys[p, q, r, s]))
    H = op_add(*terms)
    return {k: v.real for k, v in H.items() if abs(v.real) > tol}  # Hermitian -> real coeffs


def s_squared(n_orb: int) -> dict:
    """Total spin S^2 = S_- S_+ + S_z + S_z^2 in the same JW ordering (alpha p -> p, beta p -> p + n_orb)."""
    n = n_orb
    sz = op_add(*[op_scale(fermion_product([(p, True), (p, False)]), 0.5) for p in range(n)],
                *[op_scale(fermion_product([(p + n, True), (p + n, False)]), -0.5) for p in range(n)])
    sp = op_add(*[fermion_product([(p, True), (p + n, False)]) for p in range(n)])
    sm = op_add(*[fermion_product([(p + n, True), (p, False)]) for p in range(n)])
    s2 = op_add(op_mul(sm, sp), sz, op_mul(sz, sz))
    return {k: v.real for k, v in s2.items() if abs(v.real) > 1e-12}


# ------------------------------------------------------------------ sparse matrices
def to_sparse(op: dict, n: int) -> sp.csr_matrix:
    dim = 1 << n
    b = np.arange(dim)
    byx = defaultdict(list)
    for (x, z), c in op.items():
        byx[x].append((z, c))
    rows, cols, vals = [], [], []
    for x, zs in byx.items():
        d = np.zeros(dim, complex)
        for z, c in zs:
            par = np.zeros(dim, np.int64)
            zz = z
            q = 0
            while zz:
                if zz & 1:
                    par ^= (b >> q) & 1
                zz >>= 1
                q += 1
            d += c * _I_POW[popcount(x & z) % 4] * (1 - 2 * par)
        nz = np.abs(d) > 1e-14
        rows.append((b ^ x)[nz])
        cols.append(b[nz])
        vals.append(d[nz])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))


def sector_indices(n_qubits: int, n_alpha: int, n_beta: int, n_orb: int) -> np.ndarray:
    b = np.arange(1 << n_qubits)
    amask, bmask = (1 << n_orb) - 1, ((1 << n_orb) - 1) << n_orb
    pa = np.array([popcount(int(v)) for v in (b & amask)])
    pb = np.array([popcount(int(v)) for v in (b & bmask)])
    return b[(pa == n_alpha) & (pb == n_beta)]


def hf_bitstring(n_orb: int, n_alpha: int, n_beta: int) -> int:
    return ((1 << n_alpha) - 1) | (((1 << n_beta) - 1) << n_orb)


def exact_ground(Hs: sp.csr_matrix, idx: np.ndarray):
    sub = Hs[idx][:, idx]
    if len(idx) <= 800:
        w, v = np.linalg.eigh(sub.toarray())
        e0, v0 = w[0], v[:, 0]
    else:
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(sub.tocsc(), k=1, which="SA", tol=1e-12, maxiter=20000)
        e0, v0 = w[0], v[:, 0]
    psi = np.zeros(Hs.shape[0], complex)
    psi[idx] = v0
    return float(np.real(e0)), psi
