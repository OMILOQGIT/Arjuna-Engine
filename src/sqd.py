"""Minimal sample-based quantum diagonalisation (SQD), after Robledo-Moreno et al. (IBM, 2024).

1. sample bitstrings from the (noisy) circuit output distribution, with readout flips
2. map tapered bitstrings back to Slater determinants (pivot bits fixed by symmetry sector)
3. configuration recovery: fix wrong-particle-number strings by flipping the bits whose
   value disagrees most with the average orbital occupancies of the good samples
4. build the spin-closed determinant subspace (unique alpha strings x unique beta strings)
   and diagonalise H there -> variational energy
The real package is qiskit-addon-sqd; this is a teaching re-implementation for the mock.
"""
from __future__ import annotations

import numpy as np

from .pauli import popcount
from .taper import Tapering


def untaper_bits(tp: Tapering, t_bits: np.ndarray) -> np.ndarray:
    full = np.zeros_like(t_bits)
    for i, q in enumerate(tp.keep):
        full |= ((t_bits >> i) & 1) << q
    for s, q, lam in zip(tp.generators, tp.pivots, tp.eigs):
        want = 0 if lam == 1 else 1
        par = np.array([popcount(int(b & s)) & 1 for b in full])
        full |= ((par ^ want) & 1) << q
    return full


def sample(rho: np.ndarray, shots: int, n: int, p_ro: float, rng) -> np.ndarray:
    p = np.clip(np.real(np.diag(rho)), 0, None)
    p /= p.sum()
    bits = rng.choice(len(p), size=shots, p=p)
    if p_ro > 0:
        flips = (rng.random((shots, n)) < p_ro).astype(np.int64)
        bits ^= (flips << np.arange(n)).sum(axis=1)
    return bits


def _recover(strings: np.ndarray, n_orb: int, n_el: int, occ: np.ndarray) -> np.ndarray:
    out = []
    for s in strings:
        s = int(s)
        cnt = popcount(s)
        bits = [(s >> q) & 1 for q in range(n_orb)]
        while cnt != n_el:
            if cnt > n_el:  # remove electron from the occupied orbital with lowest avg occupancy
                q = min((q for q in range(n_orb) if bits[q]), key=lambda q: occ[q])
                bits[q] = 0
                cnt -= 1
            else:
                q = max((q for q in range(n_orb) if not bits[q]), key=lambda q: occ[q])
                bits[q] = 1
                cnt += 1
        out.append(sum(b << q for q, b in enumerate(bits)))
    return np.array(out, dtype=np.int64)


def sqd_energy(Hs_full, full_bits: np.ndarray, n_orb: int, n_alpha: int, n_beta: int, recover=True):
    amask = (1 << n_orb) - 1
    a = full_bits & amask
    b = (full_bits >> n_orb) & amask
    good = np.array([popcount(int(x)) == n_alpha for x in a]) & np.array([popcount(int(y)) == n_beta for y in b])
    frac_good = float(good.mean())
    if recover and good.any():
        occ_a = np.array([np.mean((a[good] >> q) & 1) for q in range(n_orb)])
        occ_b = np.array([np.mean((b[good] >> q) & 1) for q in range(n_orb)])
        a = _recover(a, n_orb, n_alpha, occ_a)
        b = _recover(b, n_orb, n_beta, occ_b)
    else:
        a, b = a[good], b[good]
    strings = np.unique(np.concatenate([a, b]))  # spin-closed (n_alpha == n_beta)
    dets = np.array(sorted(int(x) | (int(y) << n_orb) for x in strings for y in strings))
    sub = Hs_full[dets][:, dets]
    if len(dets) <= 800:
        e0 = float(np.linalg.eigvalsh(sub.toarray())[0])
    else:
        from scipy.sparse.linalg import eigsh
        e0 = float(eigsh(sub.tocsc(), k=1, which="SA", tol=1e-10)[0][0])
    return e0, len(dets), frac_good
