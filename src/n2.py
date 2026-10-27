"""N2 / STO-3G active-space problems along the bond-breaking coordinate.

CAS(6e,6o): the 3 occupied MOs with most N-2p character (2x pi_u, 3sigma_g) + all 3
virtuals (2x pi_g, 3sigma_u) -> 12 spin orbitals = 12 qubits (before tapering).
FV(10e,8o): full valence (freeze only the two 1s cores) -> 16 qubits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .integrals import molecule_integrals
from .scf import mo_integrals, rhf_robust

P2_AOS = [2, 3, 4, 7, 8, 9]  # STO-3G AO order per N: 1s, 2s, 2px, 2py, 2pz

# D2h operations for N2 on the z axis (centre at R/2) as signed AO permutations: (sx, sy, sz, swap atoms)
_D2H = [(1, 1, 1, 0), (-1, -1, 1, 0), (-1, 1, -1, 1), (1, -1, -1, 1), (-1, -1, -1, 1), (1, 1, -1, 1), (1, -1, 1, 0), (-1, 1, 1, 0)]


def _rep(sx, sy, sz, sw):
    d = np.array([1, 1, sx, sy, sz], float)
    R = np.zeros((10, 10))
    for a in range(5):
        if sw:
            R[a + 5, a] = d[a]
            R[a, a + 5] = d[a]
        else:
            R[a, a] = d[a]
            R[a + 5, a + 5] = d[a]
    return R


_REPS = [_rep(*o) for o in _D2H]


def d2h_symmetrise(F):
    """Average a one-electron AO matrix over D2h (keeps RHF orbitals symmetry-adapted at every geometry)."""
    return sum(R @ F @ R.T for R in _REPS) / len(_REPS)


@dataclass
class Problem:
    r: float
    name: str
    h: np.ndarray
    g: np.ndarray
    ecore: float
    n_orb: int
    n_alpha: int
    n_beta: int
    e_rhf_full: float
    meta: dict = field(default_factory=dict)


def _reduce(h, g, enuc, frozen, active):
    """Frozen-core effective Hamiltonian for arbitrary frozen/active index lists."""
    f, a = np.array(frozen, int), np.array(active, int)
    ecore = enuc + 2 * h[f, f].sum() + 2 * np.einsum("iijj->", g[np.ix_(f, f, f, f)]) \
        - np.einsum("ijji->", g[np.ix_(f, f, f, f)])
    heff = h[np.ix_(a, a)] + 2 * np.einsum("pqii->pq", g[np.ix_(a, a, f, f)]) \
        - np.einsum("piiq->pq", g[np.ix_(a, f, f, a)])
    return heff, g[np.ix_(a, a, a, a)].copy(), float(ecore)


def _align_degenerate(C, eps, tol=1e-5):
    """Rotate degenerate pi pairs to pure px / py so point-group symmetries become
    Z-strings under Jordan-Wigner (more qubits can then be tapered). Energy-neutral."""
    PY = 3  # py on first N
    i = 0
    while i < len(eps) - 1:
        if abs(eps[i + 1] - eps[i]) < tol:
            a, b = C[:, i].copy(), C[:, i + 1].copy()
            th = np.arctan2(-a[PY], b[PY])
            C[:, i] = np.cos(th) * a + np.sin(th) * b
            C[:, i + 1] = -np.sin(th) * a + np.cos(th) * b
            i += 2
        else:
            i += 1
    return C


def n2_problems(r: float, guess_C=None):
    ints = molecule_integrals([("N", [0, 0, 0]), ("N", [0, 0, r])])
    scf = rhf_robust(ints, guesses=[guess_C] if guess_C is not None else (), sym=d2h_symmetrise)
    C = _align_degenerate(scf["C"].copy(), scf["eps"])
    h, g = mo_integrals(ints, C)
    SC = ints["S"] @ C
    w2p = np.array([(C[P2_AOS, i] * SC[P2_AOS, i]).sum() for i in range(C.shape[1])])
    occ = list(range(7))
    core = [0, 1]
    val_occ = [i for i in occ if i not in core]
    act_occ = sorted(sorted(val_occ, key=lambda i: -w2p[i])[:3])
    frozen_cas = core + [i for i in val_occ if i not in act_occ]
    virt = [7, 8, 9]
    probs = {}
    hc, gc, ec = _reduce(h, g, ints["enuc"], frozen_cas, act_occ + virt)
    probs["cas66"] = Problem(r, "CAS(6e,6o)", hc, gc, ec, 6, 3, 3, scf["e"],
                             {"active": act_occ + virt, "frozen": frozen_cas, "w2p": w2p.tolist()})
    hf_, gf, ef = _reduce(h, g, ints["enuc"], core, val_occ + virt)
    probs["fv108"] = Problem(r, "FV(10e,8o)", hf_, gf, ef, 8, 5, 5, scf["e"],
                             {"active": val_occ + virt, "frozen": core})
    return probs, C
