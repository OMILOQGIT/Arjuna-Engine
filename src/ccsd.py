"""Spin-orbital CCSD and (T) (Stanton & Gauss 1991 intermediates; Crawford projects 5/6).

Reference = the first `nocc` spin orbitals of the supplied basis. Orbitals are given in
the block (alpha, beta) ordering used elsewhere; we reorder so occupied come first.
"""
from __future__ import annotations

import numpy as np

from .pauli import spin_orbital_integrals


def _reorder(n_orb, n_alpha, n_beta):
    occ = list(range(n_alpha)) + list(range(n_orb, n_orb + n_beta))
    vir = list(range(n_alpha, n_orb)) + list(range(n_orb + n_beta, 2 * n_orb))
    return occ + vir, len(occ)


def ccsd_t(h, g, ecore, n_alpha, n_beta, max_iter=400, tol=1e-10, ndiis=8, damp=0.0, t_init=None):
    n_orb = h.shape[0]
    hs, phys = spin_orbital_integrals(h, g)
    perm, no = _reorder(n_orb, n_alpha, n_beta)
    hs = hs[np.ix_(perm, perm)]
    phys = phys[np.ix_(perm, perm, perm, perm)]
    A = phys - phys.transpose(0, 1, 3, 2)  # <pq||rs>
    N = hs.shape[0]
    o, v = slice(0, no), slice(no, N)
    f = hs + np.einsum("piqi->pq", A[:, o, :, o])
    e_hf = ecore + np.einsum("ii->", hs[o, o]) + 0.5 * np.einsum("ijij->", A[o, o, o, o])

    fo, fv = np.diag(f)[o], np.diag(f)[v]
    Dia = fo[:, None] - fv[None, :]
    Dijab = fo[:, None, None, None] + fo[None, :, None, None] - fv[None, None, :, None] - fv[None, None, None, :]
    t1 = f[o, v] / Dia
    t2 = A[o, o, v, v] / Dijab
    fov = f[o, v]
    foo_off = f[o, o] - np.diag(np.diag(f[o, o]))
    fvv_off = f[v, v] - np.diag(np.diag(f[v, v]))

    def energy(t1, t2):
        return (np.einsum("ia,ia->", fov, t1) + 0.25 * np.einsum("ijab,ijab->", A[o, o, v, v], t2)
                + 0.5 * np.einsum("ijab,ia,jb->", A[o, o, v, v], t1, t1))

    if t_init is not None:
        t1, t2 = t_init
    e_old = energy(t1, t2)
    converged = False
    vecs, errv = [], []
    for _ in range(max_iter):
        tau_t = t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1))
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
        Fae = fvv_off - 0.5 * np.einsum("me,ma->ae", fov, t1) + np.einsum("mf,mafe->ae", t1, A[o, v, v, v]) \
            - 0.5 * np.einsum("mnaf,mnef->ae", tau_t, A[o, o, v, v])
        Fmi = foo_off + 0.5 * np.einsum("me,ie->mi", fov, t1) + np.einsum("ne,mnie->mi", t1, A[o, o, o, v]) \
            + 0.5 * np.einsum("inef,mnef->mi", tau_t, A[o, o, v, v])
        Fme = fov + np.einsum("nf,mnef->me", t1, A[o, o, v, v])
        Wmnij = A[o, o, o, o] + np.einsum("je,mnie->mnij", t1, A[o, o, o, v]) \
            - np.einsum("ie,mnje->mnij", t1, A[o, o, o, v]) + 0.25 * np.einsum("ijef,mnef->mnij", tau, A[o, o, v, v])
        Wabef = A[v, v, v, v] - np.einsum("mb,amef->abef", t1, A[v, o, v, v]) \
            + np.einsum("ma,bmef->abef", t1, A[v, o, v, v]) + 0.25 * np.einsum("mnab,mnef->abef", tau, A[o, o, v, v])
        Wmbej = A[o, v, v, o] + np.einsum("jf,mbef->mbej", t1, A[o, v, v, v]) \
            - np.einsum("nb,mnej->mbej", t1, A[o, o, v, o]) \
            - np.einsum("jnfb,mnef->mbej", 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1), A[o, o, v, v])

        r1 = fov + np.einsum("ie,ae->ia", t1, Fae) - np.einsum("ma,mi->ia", t1, Fmi) \
            + np.einsum("imae,me->ia", t2, Fme) - np.einsum("nf,naif->ia", t1, A[o, v, o, v]) \
            - 0.5 * np.einsum("imef,maef->ia", t2, A[o, v, v, v]) - 0.5 * np.einsum("mnae,nmei->ia", t2, A[o, o, v, o])
        Fae2 = Fae - 0.5 * np.einsum("mb,me->be", t1, Fme)
        Fmi2 = Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme)
        P1 = np.einsum("ijae,be->ijab", t2, Fae2)
        P2 = np.einsum("imab,mj->ijab", t2, Fmi2)
        P3 = np.einsum("imae,mbej->ijab", t2, Wmbej) - np.einsum("ie,ma,mbej->ijab", t1, t1, A[o, v, v, o])
        P4 = np.einsum("ie,abej->ijab", t1, A[v, v, v, o])
        P5 = np.einsum("ma,mbij->ijab", t1, A[o, v, o, o])
        r2 = A[o, o, v, v] + (P1 - P1.transpose(0, 1, 3, 2)) - (P2 - P2.transpose(1, 0, 2, 3)) \
            + 0.5 * np.einsum("mnab,mnij->ijab", tau, Wmnij) + 0.5 * np.einsum("ijef,abef->ijab", tau, Wabef) \
            + (P3 - P3.transpose(1, 0, 2, 3) - P3.transpose(0, 1, 3, 2) + P3.transpose(1, 0, 3, 2)) \
            + (P4 - P4.transpose(1, 0, 2, 3)) - (P5 - P5.transpose(0, 1, 3, 2))
        t1n, t2n = r1 / Dia, r2 / Dijab
        t1n = (1-damp)*t1n + damp*t1; t2n = (1-damp)*t2n + damp*t2
        vec = np.concatenate([t1n.ravel(), t2n.ravel()]); ev = vec - np.concatenate([t1.ravel(), t2.ravel()])
        vecs.append(vec); errv.append(ev)
        if len(vecs) > ndiis: vecs.pop(0); errv.pop(0)
        if ndiis and len(vecs) > 2:
            m = len(vecs); B = -np.ones((m+1, m+1)); B[m, m] = 0
            B[:m, :m] = np.array([[a @ b for b in errv] for a in errv])
            rhs = np.zeros(m+1); rhs[m] = -1
            try:
                c = np.linalg.solve(B, rhs)[:m]; vec = sum(ci*vi for ci, vi in zip(c, vecs))
                t1n = vec[:t1.size].reshape(t1.shape); t2n = vec[t1.size:].reshape(t2.shape)
            except np.linalg.LinAlgError: pass
        e_new = energy(t1n, t2n)
        if not np.isfinite(e_new) or abs(e_new) > 50:
            break
        dt = max(np.abs(t1n - t1).max(), np.abs(t2n - t2).max())
        t1, t2 = t1n, t2n
        if abs(e_new - e_old) < tol and dt < 1e-8:
            converged = True
            e_old = e_new
            break
        e_old = e_new

    out = {"hf": float(e_hf), "converged": converged, "t": (t1, t2), "e_t": None}
    if not converged:
        out.update(ccsd=float("nan"), ccsd_t=float("nan"))
        return out
    out["ccsd"] = float(e_hf + e_old)
    # (T): Crawford project #6, canonical orbitals assumed (diagonal Fock)
    Doo = fo
    Dvv = fv
    Dijkabc = (Doo[:, None, None, None, None, None] + Doo[None, :, None, None, None, None]
               + Doo[None, None, :, None, None, None] - Dvv[None, None, None, :, None, None]
               - Dvv[None, None, None, None, :, None] - Dvv[None, None, None, None, None, :])

    def P_abc(X):  # P(i/jk) style permutation over last three (a/bc)
        return X - X.transpose(0, 1, 2, 4, 3, 5) - X.transpose(0, 1, 2, 5, 4, 3)

    def P_ijk(X):
        return X - X.transpose(1, 0, 2, 3, 4, 5) - X.transpose(2, 1, 0, 3, 4, 5)

    disc = np.einsum("ia,jkbc->ijkabc", t1, A[o, o, v, v])
    disc = P_ijk(P_abc(disc)) / Dijkabc
    conn1 = np.einsum("jkae,eibc->ijkabc", t2, A[v, o, v, v])
    conn2 = np.einsum("imbc,majk->ijkabc", t2, A[o, v, o, o])
    conn = P_ijk(P_abc(conn1 - conn2)) / Dijkabc
    e_t = np.einsum("ijkabc,ijkabc,ijkabc->", conn, Dijkabc, conn + disc) / 36.0
    out["ccsd_t"] = float(out["ccsd"] + e_t); out["e_t"] = float(e_t)
    return out
