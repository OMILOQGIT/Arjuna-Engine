"""ADAPT-VQE (Grimsley et al. 2019) on a statevector, fermionic UCCSD operator pool.

Each pool generator A = T - T^dagger (single or double excitation, JW-mapped) satisfies
A^3 = -A, so exp(theta A) = 1 + sin(theta) A + (1 - cos(theta)) A^2 exactly. Gradients of
the ansatz parameters use the adjoint method.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .pauli import fermion_product, op_add, op_scale, popcount, to_sparse


def uccsd_pool(n_orb: int, n_alpha: int, n_beta: int) -> list[tuple[str, dict]]:
    occ_a, vir_a = range(n_alpha), range(n_alpha, n_orb)
    occ_b, vir_b = range(n_orb, n_orb + n_beta), range(n_orb + n_beta, 2 * n_orb)
    pool = []

    def add(label, creators, annihilators):
        T = fermion_product([(p, True) for p in creators] + [(p, False) for p in annihilators])
        Td = fermion_product([(p, True) for p in reversed(annihilators)] + [(p, False) for p in reversed(creators)])
        A = op_add(T, op_scale(Td, -1))
        if A:
            pool.append((label, A))

    for i, a in itertools.product(occ_a, vir_a):
        add(f"s a{i}->a{a}", [a], [i])
    for i, a in itertools.product(occ_b, vir_b):
        add(f"s b{i}->b{a}", [a], [i])
    for (i, j), (a, b) in itertools.product(itertools.combinations(occ_a, 2), itertools.combinations(vir_a, 2)):
        add(f"d a{i}a{j}->a{a}a{b}", [a, b], [j, i])
    for (i, j), (a, b) in itertools.product(itertools.combinations(occ_b, 2), itertools.combinations(vir_b, 2)):
        add(f"d b{i}b{j}->b{a}b{b}", [a, b], [j, i])
    for i, j, a, b in itertools.product(occ_a, occ_b, vir_a, vir_b):
        add(f"d a{i}b{j}->a{a}b{b}", [a, b], [j, i])
    return pool


@dataclass
class AdaptResult:
    energies: list = field(default_factory=list)   # energy after each ADAPT iteration
    max_grads: list = field(default_factory=list)
    ops: list = field(default_factory=list)        # chosen pool indices
    labels: list = field(default_factory=list)
    thetas: np.ndarray = None
    state: np.ndarray = None
    theta_history: list = field(default_factory=list)  # optimal thetas after each iteration


class Adapt:
    def __init__(self, H: dict, pool_ops: list[dict], n_qubits: int, ref_bits: int):
        self.n = n_qubits
        self.H = to_sparse(H, n_qubits).tocsr()
        self.A = [to_sparse(op, n_qubits).tocsr() for op in pool_ops]
        self.ref = np.zeros(1 << n_qubits, complex)
        self.ref[ref_bits] = 1.0

    @staticmethod
    def _apply(A, th, v):
        Av = A @ v
        return v + np.sin(th) * Av + (1 - np.cos(th)) * (A @ Av)

    def state(self, ops, thetas):
        v = self.ref
        for k, th in zip(ops, thetas):
            v = self._apply(self.A[k], th, v)
        return v

    def energy_grad(self, thetas, ops):
        psi = self.state(ops, thetas)
        lam = self.H @ psi
        e = float(np.real(np.vdot(psi, lam)))
        g = np.zeros(len(ops))
        phi = psi
        for idx in range(len(ops) - 1, -1, -1):
            A = self.A[ops[idx]]
            g[idx] = 2 * np.real(np.vdot(lam, A @ phi))
            phi = self._apply(A, -thetas[idx], phi)
            lam = self._apply(A, -thetas[idx], lam)
        return e, g

    def run(self, e_exact=None, grad_tol=1e-4, max_ops=60, energy_tol=None, labels=None, verbose=False, callback=None):
        res = AdaptResult()
        ops, thetas = [], np.zeros(0)
        psi = self.ref
        res.energies.append(float(np.real(np.vdot(psi, self.H @ psi))))
        for it in range(max_ops):
            Hpsi = self.H @ psi
            grads = np.array([abs(2 * np.real(np.vdot(Hpsi, A @ psi))) for A in self.A])
            k = int(np.argmax(grads))
            res.max_grads.append(float(grads[k]))
            if grads[k] < grad_tol:
                break
            ops.append(k)
            thetas = np.append(thetas, 0.0)
            opt = minimize(self.energy_grad, thetas, args=(ops,), jac=True, method="BFGS",
                           options={"gtol": 1e-8, "maxiter": 2000})
            thetas = opt.x
            psi = self.state(ops, thetas)
            res.energies.append(float(opt.fun))
            res.theta_history.append(thetas.copy())
            if verbose:
                err = "" if e_exact is None else f" err {1e3 * (opt.fun - e_exact):.4f} mHa"
                print(f"  iter {it + 1:2d} op {labels[k] if labels else k} |g|={grads[k]:.2e} E={opt.fun:.8f}{err}",
                      flush=True)
            if callback is not None:  # progress hook (web backend); may raise to cancel
                callback(it + 1, float(opt.fun), labels[k] if labels else k, float(grads[k]))
            if energy_tol is not None and e_exact is not None and opt.fun - e_exact < energy_tol:
                break
        res.ops, res.thetas, res.state = ops, thetas, psi
        res.labels = [labels[k] for k in ops] if labels else ops
        return res


def cnot_count_naive(ops_paulis: list[dict]) -> int:
    """CNOT ladder cost, all-to-all connectivity: each Pauli rotation of weight w costs 2(w-1)."""
    total = 0
    for op in ops_paulis:
        for (x, z) in op:
            w = popcount(x | z)
            total += 2 * max(w - 1, 0)
    return total
