"""Gaussian molecular integrals (McMurchie-Davidson) for small STO-3G calculations.

Pure NumPy/SciPy. Supports s and p Cartesian shells, which is all STO-3G needs for
first-row atoms. Validated in tests against published RHF energies.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import hyp1f1


def factorial2(n: int) -> float:
    """Double factorial with (-1)!! = 1 (scipy returns 0 for negatives)."""
    return float(np.prod(np.arange(n, 0, -2))) if n > 0 else 1.0

ANG2BOHR = 1.0 / 0.52917721092

# STO-3G (EMSL Basis Set Exchange)
_C1S = [0.15432897, 0.53532814, 0.44463454]
_C2S = [-0.09996723, 0.39951283, 0.70011547]
_C2P = [0.15591627, 0.60768372, 0.39195739]
STO3G = {
    "H": [("s", [3.42525091, 0.62391373, 0.16885540], _C1S)],
    "N": [
        ("s", [99.1061690, 18.0523120, 4.8856602], _C1S),
        ("s", [3.7804559, 0.8784966, 0.2857144], _C2S),
        ("p", [3.7804559, 0.8784966, 0.2857144], _C2P),
    ],
    "O": [
        ("s", [130.7093200, 23.8088610, 6.4436083], _C1S),
        ("s", [5.0331513, 1.1695961, 0.3803890], _C2S),
        ("p", [5.0331513, 1.1695961, 0.3803890], _C2P),
    ],
}
ZNUC = {"H": 1, "N": 7, "O": 8}


@dataclass
class BasisFunction:
    origin: np.ndarray
    shell: tuple          # (l, m, n)
    exps: np.ndarray
    coefs: np.ndarray     # includes primitive + contraction normalisation

    @property
    def L(self) -> int:
        return sum(self.shell)


def _normalise(bf: BasisFunction) -> None:
    l, m, n = bf.shell
    L = l + m + n
    norm_prim = np.sqrt(
        2 ** (2 * L + 1.5) * bf.exps ** (L + 1.5)
        / (factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1) * np.pi ** 1.5)
    )
    c = bf.coefs * norm_prim
    pref = np.pi ** 1.5 * factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1) / 2.0 ** L
    s = sum(c[i] * c[j] / (bf.exps[i] + bf.exps[j]) ** (L + 1.5)
            for i in range(len(c)) for j in range(len(c)))
    bf.coefs = c / np.sqrt(pref * s)


def build_basis(atoms: list[tuple[str, np.ndarray]]) -> list[BasisFunction]:
    """atoms: list of (symbol, xyz in bohr)."""
    out = []
    for sym, xyz in atoms:
        for kind, exps, coefs in STO3G[sym]:
            shells = [(0, 0, 0)] if kind == "s" else [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            for sh in shells:
                bf = BasisFunction(np.asarray(xyz, float), sh, np.array(exps, float), np.array(coefs, float))
                _normalise(bf)
                out.append(bf)
    return out


# ---------------------------------------------------------------- Hermite machinery
def _E(i, j, t, Qx, a, b):
    """Hermite expansion coefficient E^{ij}_t (recursive)."""
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == j == t == 0:
        return np.exp(-q * Qx * Qx)
    if j == 0:
        return (1 / (2 * p)) * _E(i - 1, j, t - 1, Qx, a, b) - (q * Qx / a) * _E(i - 1, j, t, Qx, a, b) \
            + (t + 1) * _E(i - 1, j, t + 1, Qx, a, b)
    return (1 / (2 * p)) * _E(i, j - 1, t - 1, Qx, a, b) + (q * Qx / b) * _E(i, j - 1, t, Qx, a, b) \
        + (t + 1) * _E(i, j - 1, t + 1, Qx, a, b)


def _boys(n, T):
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0)


def _R_table(tmax, umax, vmax, p, PC):
    """Hermite Coulomb integrals R^0_{tuv} for all t<=tmax.. via table recursion."""
    T = p * PC @ PC
    nmax = tmax + umax + vmax
    R = np.zeros((nmax + 1, tmax + 1, umax + 1, vmax + 1))
    for n in range(nmax + 1):
        R[n, 0, 0, 0] = (-2 * p) ** n * _boys(n, T)
    X, Y, Z = PC
    for n in range(nmax - 1, -1, -1):
        for t in range(tmax + 1):
            for u in range(umax + 1):
                for v in range(vmax + 1):
                    if t == u == v == 0 or t + u + v > nmax - n:
                        continue
                    if t > 0:
                        val = X * R[n + 1, t - 1, u, v]
                        if t > 1:
                            val += (t - 1) * R[n + 1, t - 2, u, v]
                    elif u > 0:
                        val = Y * R[n + 1, t, u - 1, v]
                        if u > 1:
                            val += (u - 1) * R[n + 1, t, u - 2, v]
                    else:
                        val = Z * R[n + 1, t, u, v - 1]
                        if v > 1:
                            val += (v - 1) * R[n + 1, t, u, v - 2]
                    R[n, t, u, v] = val
    return R[0]


@dataclass
class _Pair:
    p: float
    P: np.ndarray
    coef: float
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray


def _pairs(A: BasisFunction, B: BasisFunction) -> list[_Pair]:
    out = []
    (l1, m1, n1), (l2, m2, n2) = A.shell, B.shell
    AB = A.origin - B.origin
    for ca, a in zip(A.coefs, A.exps):
        for cb, b in zip(B.coefs, B.exps):
            p = a + b
            P = (a * A.origin + b * B.origin) / p
            Ex = np.array([_E(l1, l2, t, AB[0], a, b) for t in range(l1 + l2 + 1)])
            Ey = np.array([_E(m1, m2, t, AB[1], a, b) for t in range(m1 + m2 + 1)])
            Ez = np.array([_E(n1, n2, t, AB[2], a, b) for t in range(n1 + n2 + 1)])
            out.append(_Pair(p, P, ca * cb, Ex, Ey, Ez))
    return out


def _kinetic_prim(a, la, A, b, lb, B):
    def ov(i, j, k, l, m, n):
        AB = A - B
        return (_E(i, l, 0, AB[0], a, b) * _E(j, m, 0, AB[1], a, b) * _E(k, n, 0, AB[2], a, b)
                * (np.pi / (a + b)) ** 1.5)
    l1, m1, n1 = la
    l2, m2, n2 = lb
    t0 = b * (2 * (l2 + m2 + n2) + 3) * ov(l1, m1, n1, l2, m2, n2)
    t1 = -2 * b ** 2 * (ov(l1, m1, n1, l2 + 2, m2, n2) + ov(l1, m1, n1, l2, m2 + 2, n2)
                        + ov(l1, m1, n1, l2, m2, n2 + 2))
    t2 = -0.5 * (l2 * (l2 - 1) * ov(l1, m1, n1, l2 - 2, m2, n2) + m2 * (m2 - 1) * ov(l1, m1, n1, l2, m2 - 2, n2)
                 + n2 * (n2 - 1) * ov(l1, m1, n1, l2, m2, n2 - 2))
    return t0 + t1 + t2


def one_electron(basis, nuclei):
    """Returns S, T, V. nuclei: list of (Z, xyz_bohr)."""
    n = len(basis)
    S, T, V = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        A, B = basis[i], basis[j]
        s = v = 0.0
        for pr in _pairs(A, B):
            s += pr.coef * pr.Ex[0] * pr.Ey[0] * pr.Ez[0] * (np.pi / pr.p) ** 1.5
            for Zc, C in nuclei:
                R = _R_table(len(pr.Ex) - 1, len(pr.Ey) - 1, len(pr.Ez) - 1, pr.p, pr.P - C)
                v += -Zc * pr.coef * (2 * np.pi / pr.p) * np.einsum("t,u,v,tuv->", pr.Ex, pr.Ey, pr.Ez, R)
        t = sum(ca * cb * _kinetic_prim(a, A.shell, A.origin, b, B.shell, B.origin)
                for ca, a in zip(A.coefs, A.exps) for cb, b in zip(B.coefs, B.exps))
        S[i, j] = S[j, i] = s
        T[i, j] = T[j, i] = t
        V[i, j] = V[j, i] = v
    return S, T, V


def two_electron(basis):
    """(ij|kl) chemist notation, 8-fold symmetry exploited."""
    n = len(basis)
    pairs = {}
    for i in range(n):
        for j in range(i + 1):
            pairs[(i, j)] = _pairs(basis[i], basis[j])
    keys = list(pairs)
    eri = np.zeros((n, n, n, n))
    for a_idx, (i, j) in enumerate(keys):
        for (k, l) in keys[: a_idx + 1]:
            val = 0.0
            for P in pairs[(i, j)]:
                for Q in pairs[(k, l)]:
                    p, q = P.p, Q.p
                    alpha = p * q / (p + q)
                    tm, um, vm = len(P.Ex) - 1, len(P.Ey) - 1, len(P.Ez) - 1
                    tn, un, vn = len(Q.Ex) - 1, len(Q.Ey) - 1, len(Q.Ez) - 1
                    R = _R_table(tm + tn, um + un, vm + vn, alpha, P.P - Q.P)
                    s = 0.0
                    for t in range(tm + 1):
                        for u in range(um + 1):
                            for v in range(vm + 1):
                                e1 = P.Ex[t] * P.Ey[u] * P.Ez[v]
                                if e1 == 0.0:
                                    continue
                                for tau in range(tn + 1):
                                    for nu in range(un + 1):
                                        for phi in range(vn + 1):
                                            s += (e1 * Q.Ex[tau] * Q.Ey[nu] * Q.Ez[phi]
                                                  * (-1) ** (tau + nu + phi) * R[t + tau, u + nu, v + phi])
                    val += P.coef * Q.coef * s * 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
            for (a, b, c, d) in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                 (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                eri[a, b, c, d] = val
    return eri


def nuclear_repulsion(nuclei):
    e = 0.0
    for (Za, A), (Zb, B) in itertools.combinations(nuclei, 2):
        e += Za * Zb / np.linalg.norm(A - B)
    return e


def molecule_integrals(atoms_ang: list[tuple[str, list[float]]]):
    """atoms in Angstrom -> dict(S,T,V,eri,enuc,nelec)."""
    atoms = [(s, np.array(x, float) * ANG2BOHR) for s, x in atoms_ang]
    return molecule_integrals_bohr(atoms)


def molecule_integrals_bohr(atoms):
    basis = build_basis(atoms)
    nuclei = [(ZNUC[s], np.asarray(x, float)) for s, x in atoms]
    S, T, V = one_electron(basis, nuclei)
    eri = two_electron(basis)
    return {"S": S, "T": T, "V": V, "eri": eri, "enuc": nuclear_repulsion(nuclei),
            "nelec": sum(ZNUC[s] for s, _ in atoms)}
