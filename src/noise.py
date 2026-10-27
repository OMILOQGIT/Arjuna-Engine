"""Gate-level density-matrix simulator with a parametrised hardware noise model.

Noise model (per gate, Pauli-twirled / depolarising approximation):
  * 2q gate (CNOT): depolarising, average gate error r2  -> lambda = 4/3 r2
  * 1q gate:        depolarising, average gate error r1  -> lambda = 2   r1
  * readout:        symmetric bit flip p_ro on every measured qubit
Default numbers are order-of-magnitude for current superconducting devices and are
meant to be replaced with a real backend calibration (qiskit FakeBackend) on laptops.
Connectivity is all-to-all (no SWAP routing) -> CNOT counts are a lower bound.

Mitigation implemented: readout-error mitigation (exact inverse for symmetric flips),
zero-noise extrapolation by folding EVERY gate (scale 1,3,5: G -> G G^dag G), with linear /
Richardson / exponential (known fully-mixed asymptote) fits, and probabilistic error
cancellation (PEC) with its sampling cost. Depolarising noise is unitarily covariant, so a
folded gate equals the gate followed by the noise channel applied `fold` times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pauli import popcount, to_sparse

H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
RX90 = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)  # Rx(pi/2): maps Y-basis to Z-basis


@dataclass
class NoiseModel:
    r2: float = 3e-3
    r1: float = 2.5e-4
    p_ro: float = 1.5e-2

    def scaled(self, s):
        return NoiseModel(self.r2 * s, self.r1 * s, self.p_ro)


def compile_rotations(ops: list[dict], thetas, n: int):
    """exp(theta*A) for A = sum_j (i a_j) P_j (commuting strings) -> list of (x, z, angle)
    meaning exp(-i angle/2 P)."""
    rots = []
    for A, th in zip(ops, thetas):
        for (x, z), c in A.items():
            a = np.imag(c)  # A anti-Hermitian: coefficients purely imaginary
            if abs(a) < 1e-14:
                continue
            # exp(theta * i a P) = exp(-i (angle/2) P) with angle = -2 a theta
            rots.append((x, z, -2.0 * a * th))
    return rots


def rotation_gates(rots, n: int):
    """Decompose Pauli rotations into H / Rx(pi/2) basis changes, CNOT ladders and Rz."""
    gates = []
    for x, z, ang in rots:
        qs = [q for q in range(n) if ((x | z) >> q) & 1]
        if not qs:
            continue
        pre, post = [], []
        for q in qs:
            xb, zb = (x >> q) & 1, (z >> q) & 1
            if xb and not zb:     # X
                pre.append(("1q", q, H_GATE))
                post.append(("1q", q, H_GATE))
            elif xb and zb:       # Y
                pre.append(("1q", q, RX90))
                post.append(("1q", q, RX90.conj().T))
        ladder = [("cx", qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
        rz = np.diag([np.exp(-1j * ang / 2), np.exp(1j * ang / 2)])
        gates += pre + ladder + [("1q", qs[-1], rz)] + ladder[::-1] + post
    return gates


class DensitySim:
    def __init__(self, n: int):
        self.n = n
        self.dim = 1 << n
        idx = np.arange(self.dim)
        self._cx_perm = {}
        self._idx = idx

    def _perm(self, c, t):
        key = (c, t)
        if key not in self._cx_perm:
            i = self._idx
            self._cx_perm[key] = np.where((i >> c) & 1, i ^ (1 << t), i)
        return self._cx_perm[key]

    def _t(self, rho):
        return rho.reshape([2] * (2 * self.n))

    def _ax(self, q):  # tensor axis of qubit q (row); column axis = +n
        return self.n - 1 - q

    def apply_1q(self, rho, q, U):
        """rho -> U rho U^dag on qubit q (vectorised: one matmul per side)."""
        A, B = 1 << (self.n - 1 - q), 1 << q
        R = np.matmul(U, rho.reshape(A, 2, B * self.dim))                       # rows
        R = np.matmul(U.conj(), R.reshape(self.dim * A, 2, B))                  # columns: rho U^dag
        return R.reshape(self.dim, self.dim)

    def apply_cx(self, rho, c, t):
        p = self._perm(c, t)
        return rho[np.ix_(p, p)]

    def depolarize(self, rho, qubits, lam):
        """(1-lam) rho + lam Tr_S(rho) (x) I/2^k on the qubit set S (k = 1 or 2)."""
        if lam <= 0:
            return rho
        n = self.n
        if len(qubits) == 1:
            q = qubits[0]
            A, B = 1 << (n - 1 - q), 1 << q
            R = rho.reshape(A, 2, B, A, 2, B)
            tr = (R[:, 0, :, :, 0, :] + R[:, 1, :, :, 1, :]) * (lam / 2)
            out = (1 - lam) * R
            out[:, 0, :, :, 0, :] += tr
            out[:, 1, :, :, 1, :] += tr
            return out.reshape(self.dim, self.dim)
        q1, q2 = sorted(qubits)
        A, B, C = 1 << (n - 1 - q2), 1 << (q2 - q1 - 1), 1 << q1
        R = rho.reshape(A, 2, B, 2, C, A, 2, B, 2, C)
        tr = sum(R[:, i, :, j, :, :, i, :, j, :] for i in (0, 1) for j in (0, 1)) * (lam / 4)
        out = (1 - lam) * R
        for i in (0, 1):
            for j in (0, 1):
                out[:, i, :, j, :, :, i, :, j, :] += tr
        return out.reshape(self.dim, self.dim)

    def run(self, gates, ref_bits: int, noise: NoiseModel | None, fold: int = 1):
        rho = np.zeros((self.dim, self.dim), complex)
        rho[ref_bits, ref_bits] = 1.0  # basis-state prep (X gates) assumed error-free
        l2 = 4 / 3 * noise.r2 if noise else 0.0
        l1 = 2 * noise.r1 if noise else 0.0
        for g in gates:
            if g[0] == "1q":
                rho = self.apply_1q(rho, g[1], g[2])
                for _ in range(fold):  # U U^dag U with noise after each == U then noise x fold
                    rho = self.depolarize(rho, [g[1]], l1)
            else:
                for _ in range(fold):  # fold=3 -> CX CX CX (identity-equivalent extra pairs)
                    rho = self.apply_cx(rho, g[1], g[2])
                    rho = self.depolarize(rho, [g[1], g[2]], l2)
        return rho


class Estimator:
    """Energy estimator: readout error, optional readout mitigation, shot noise."""

    def __init__(self, H: dict, n: int):
        self.n = n
        self.const = sum(c for (x, z), c in H.items() if x == 0 and z == 0).real
        self.terms = [(x, z, c.real if np.iscomplexobj(c) else c) for (x, z), c in H.items() if (x, z) != (0, 0)]
        self.mats = [to_sparse({(x, z): 1.0}, n) for x, z, _ in self.terms]
        self.weights = np.array([popcount(x | z) for x, z, _ in self.terms])
        self.coefs = np.array([c for _, _, c in self.terms])

    def pauli_expectations(self, rho):
        return np.array([np.real((M.multiply(rho.T)).sum()) for M in self.mats])

    def energy(self, rho, p_ro=0.0, rem=False, shots=None, rng=None):
        ev = self.pauli_expectations(rho)
        damp = (1 - 2 * p_ro) ** self.weights
        meas = ev * damp
        if shots:
            rng = rng or np.random.default_rng(0)
            var = np.clip(1 - meas ** 2, 0, None) / shots
            meas = meas + rng.normal(size=meas.shape) * np.sqrt(var)
        if rem:
            meas = meas / damp
        return float(self.const + self.coefs @ meas)


def zne(scales, values, kind="richardson", asymptote=None):
    s, v = np.asarray(scales, float), np.asarray(values, float)
    if kind == "linear":
        A = np.vstack([np.ones_like(s), s]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return float(coef[0])
    if kind == "richardson":  # polynomial of degree len-1 through all points
        return float(np.polyval(np.polyfit(s, v, len(s) - 1), 0.0))
    if kind == "exponential":
        # exact a + b exp(c s) through three equally spaced points (free asymptote)
        v1, v3, v5 = v
        q = (v5 - v3) / (v3 - v1) if v3 != v1 else float("nan")
        if not q > 0 or abs(q - 1) < 1e-12:
            return float("nan")
        be = (v3 - v1) / (q - 1)
        return float(v1 - be + be / np.sqrt(q))
    if kind == "exponential_asym":
        # v(s) = E_inf - B exp(-c s) with the fully-mixed limit E_inf known (identity coefficient of H);
        # least squares on ln(E_inf - v) (as mitiq ExpFactory with asymptote)
        if asymptote is None:
            raise ValueError("exponential ZNE needs the fully-mixed asymptote")
        d = asymptote - v
        if np.any(d <= 0):
            return float("nan")
        slope, icpt = np.polyfit(s, np.log(d), 1)
        return float(asymptote - np.exp(icpt))
    raise ValueError(kind)


def pec_overhead(gates, noise: NoiseModel) -> float:
    """Sampling overhead gamma of PEC for local depolarising noise (shots scale ~gamma^2)."""
    l2, l1 = 4 / 3 * noise.r2, 2 * noise.r1
    g2 = 1 + 15 * l2 / (8 * (1 - l2))
    g1 = 1 + 3 * l1 / (2 * (1 - l1))
    n2 = sum(1 for g in gates if g[0] == "cx")
    n1 = len(gates) - n2
    return float(g2 ** n2 * g1 ** n1)


def count_gates(gates):
    n2 = sum(1 for g in gates if g[0] == "cx")
    return {"cx": n2, "1q": len(gates) - n2}
