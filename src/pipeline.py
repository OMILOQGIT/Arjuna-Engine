"""End-to-end helpers: symmetric-branch problems, tapering, ADAPT, circuit compilation."""
from __future__ import annotations

from dataclasses import dataclass

from .adapt import Adapt, uccsd_pool
from .n2 import Problem, n2_problems
from .noise import compile_rotations, count_gates, rotation_gates
import numpy as np

from .pauli import exact_ground, hf_bitstring, op_add, op_scale, qubit_hamiltonian, s_squared, sector_indices, to_sparse
from .taper import Tapering

EQ = 1.098
SPIN_PENALTY = 0.5  # Ha; H + mu*S^2 pins the singlet (at dissociation high-spin states become degenerate/lower)


def problems_along(geoms: list[float]) -> dict[float, dict[str, Problem]]:
    """Follow the symmetric RHF branch: start at equilibrium, continue outward and inward."""
    out = {}
    probs, C_eq = n2_problems(EQ)
    out[EQ] = probs
    C = C_eq
    for r in sorted(g for g in geoms if g > EQ):
        out[r], C = n2_problems(r, C)
    C = C_eq
    for r in sorted((g for g in geoms if g < EQ), reverse=True):
        out[r], C = n2_problems(r, C)
    return {r: out[r] for r in sorted(out) if r in geoms or r == EQ}


@dataclass
class Prepared:
    P: Problem
    H: dict
    e_exact: float          # exact SINGLET ground energy (lowest eigenvalue of H + mu S^2)
    tp: Tapering
    Ht: dict                # tapered physical Hamiltonian (what a chip measures)
    labels: list
    pool_full: list
    pool_t: list
    Hpen_t: dict = None     # tapered H + mu S^2 (what ADAPT minimises)
    S2_t: dict = None
    e_lowest_any: float = None  # lowest eigenvalue of H in the Sz=0 sector, any spin
    s2_lowest_any: float = None


def prepare(P: Problem, mu: float = SPIN_PENALTY) -> Prepared:
    nq = 2 * P.n_orb
    H = qubit_hamiltonian(P.h, P.g, P.ecore)
    S2 = s_squared(P.n_orb)
    idx = sector_indices(nq, P.n_alpha, P.n_beta, P.n_orb)
    e_any, psi_any = exact_ground(to_sparse(H, nq), idx)
    s2_any = float(np.real(np.vdot(psi_any, to_sparse(S2, nq) @ psi_any)))
    Hpen = op_add(H, op_scale(S2, mu))
    e_ex, _ = exact_ground(to_sparse(Hpen, nq), idx)
    tp = Tapering(H, nq, hf_bitstring(P.n_orb, P.n_alpha, P.n_beta))
    assert tp.commutes_with_symmetries(S2), "S^2 must commute with the tapering symmetries"
    keep = [(lab, A) for lab, A in uccsd_pool(P.n_orb, P.n_alpha, P.n_beta) if tp.commutes_with_symmetries(A)]
    return Prepared(P, H, e_ex, tp, tp.taper(H), [l for l, _ in keep], [A for _, A in keep],
                    [tp.taper(A) for _, A in keep], Hpen_t=tp.taper(Hpen), S2_t=tp.taper(S2),
                    e_lowest_any=e_any, s2_lowest_any=s2_any)


def run_adapt(pr: Prepared, max_ops=80, grad_tol=1e-5, verbose=False, callback=None):
    """ADAPT on H + mu S^2 (spin-pinned); res.energies are then re-evaluated with the physical H."""
    ad = Adapt(pr.Hpen_t, pr.pool_t, pr.tp.n_tapered, pr.tp.ref_bits)
    res = ad.run(e_exact=pr.e_exact, grad_tol=grad_tol, max_ops=max_ops, labels=pr.labels, verbose=verbose, callback=callback)
    Hs, S2s = to_sparse(pr.Ht, pr.tp.n_tapered), to_sparse(pr.S2_t, pr.tp.n_tapered)
    states = [ad.ref] + [ad.state(res.ops[:k + 1], th) for k, th in enumerate(res.theta_history)]
    res.energies_pen = res.energies
    res.energies = [float(np.real(np.vdot(s, Hs @ s))) for s in states]
    res.s2 = [float(np.real(np.vdot(s, S2s @ s))) for s in states]
    return res


def first_chem_acc(res, e_exact, thr=1.6e-3):
    return next((k for k, e in enumerate(res.energies) if e - e_exact < thr), None)


def circuit_for(pr: Prepared, res, k: int):
    """Gate list for the first k ADAPT operators with their optimal parameters at iteration k."""
    if k == 0:
        return []
    ops = [pr.pool_t[i] for i in res.ops[:k]]
    rots = compile_rotations(ops, res.theta_history[k - 1], pr.tp.n_tapered)
    return rotation_gates(rots, pr.tp.n_tapered)


def untapered_cx(pr: Prepared, res, k: int) -> int:
    ops = [pr.pool_full[i] for i in res.ops[:k]]
    rots = compile_rotations(ops, res.theta_history[k - 1], 2 * pr.P.n_orb)
    return count_gates(rotation_gates(rots, 2 * pr.P.n_orb))["cx"]
