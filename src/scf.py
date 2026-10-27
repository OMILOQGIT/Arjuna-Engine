"""Restricted Hartree-Fock with DIIS, plus MO-basis integral transforms."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def rhf(ints, max_iter=300, tol=1e-11, guess_C=None, sym=None):
    """RHF with DIIS. `sym` (optional) symmetrises the Fock matrix every iteration (point-group averaging),
    which keeps the orbitals exactly symmetry-adapted so symmetry-forbidden Pauli terms vanish."""
    S, H = ints["S"], ints["T"] + ints["V"]
    eri, nocc = ints["eri"], ints["nelec"] // 2
    if guess_C is None:
        _, C = eigh(H, S)
    else:  # orthonormalise the guess in the new metric
        C = guess_C @ np.linalg.inv(np.linalg.cholesky(guess_C.T @ S @ guess_C).T)
    D = 2 * C[:, :nocc] @ C[:, :nocc].T
    focks, errs = [], []
    e_old = 0.0
    for it in range(max_iter):
        J = np.einsum("pqrs,rs->pq", eri, D)
        K = np.einsum("prqs,rs->pq", eri, D)
        F = H + J - 0.5 * K
        if sym is not None:
            F = sym(F)
        e = 0.5 * np.sum(D * (H + F)) + ints["enuc"]
        err = F @ D @ S - S @ D @ F
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        if len(focks) > 1:
            m = len(focks)
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0
            for a in range(m):
                for b in range(m):
                    B[a, b] = np.sum(errs[a] * errs[b])
            rhs = np.zeros(m + 1)
            rhs[m] = -1
            try:
                c = np.linalg.solve(B, rhs)[:m]
                F = sum(ci * Fi for ci, Fi in zip(c, focks))
            except np.linalg.LinAlgError:
                pass
        eps, C = eigh(F, S)
        D = 2 * C[:, :nocc] @ C[:, :nocc].T
        if abs(e - e_old) < tol and np.linalg.norm(err) < 1e-9:
            return {"e": e, "C": C, "eps": eps, "converged": True, "iters": it}
        e_old = e
    return {"e": e, "C": C, "eps": eps, "converged": False, "iters": max_iter}


def rhf_robust(ints, guesses=(), n_random=12, seed=0, continuation=True, sym=None):
    """Lowest converged RHF over core, GWH, supplied and random guesses.

    The core-Hamiltonian guess alone lands on an excited SCF solution for N2.
    With continuation=True and a supplied guess (previous geometry), the solution
    reached from that guess is returned: this follows the symmetry-adapted branch
    instead of jumping to a lower, symmetry-broken RHF at long bond lengths.
    """
    if continuation and guesses:
        r = rhf(ints, guess_C=guesses[0], sym=sym)
        if r["converged"]:
            return r
    S, H = ints["S"], ints["T"] + ints["V"]
    cands = [None]
    gwh = 0.875 * S * (np.diag(H)[:, None] + np.diag(H)[None, :])
    np.fill_diagonal(gwh, np.diag(H))
    cands.append(eigh(gwh, S)[1])
    cands += list(guesses)
    rng = np.random.default_rng(seed)
    cands += [rng.normal(size=S.shape) for _ in range(n_random)]
    best = None
    for C0 in cands:
        r = rhf(ints, guess_C=C0, sym=sym)
        if r["converged"] and (best is None or r["e"] < best["e"] - 1e-9):
            best = r
    return best


def mo_integrals(ints, C):
    h = C.T @ (ints["T"] + ints["V"]) @ C
    g = np.einsum("pqrs,pi,qj,rk,sl->ijkl", ints["eri"], C, C, C, C, optimize=True)
    return h, g


def active_space(h, g, enuc, n_frozen, n_active):
    """Frozen-core effective Hamiltonian on orbitals [n_frozen, n_frozen+n_active)."""
    fc = slice(0, n_frozen)
    act = slice(n_frozen, n_frozen + n_active)
    ecore = enuc + 2 * np.einsum("ii->", h[fc, fc]) + np.einsum("iijj->", 2 * g[fc, fc, fc, fc]) \
        - np.einsum("ijji->", g[fc, fc, fc, fc])
    h_eff = h[act, act] + 2 * np.einsum("pqii->pq", g[act, act, fc, fc]) - np.einsum("piiq->pq", g[act, fc, fc, act])
    return h_eff, g[act, act, act, act].copy(), float(ecore)
