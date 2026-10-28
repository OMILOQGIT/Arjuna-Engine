# Production port. Written against PySCF 2.14.0 docs/source (https://pyscf.org/user.html,
# https://github.com/pyscf/pyscf/tree/v2.14.0) and block2 0.5.4 (https://block2.readthedocs.io)
# on 2026-10-27. Not executed in the authoring environment (no package access); run
# port/test_port.py on a laptop first.
"""N2 chemistry with PySCF -> our ``src.n2.Problem`` (same fields), plus classical benchmarks.

Orbital conventions (must match src/):
  * ``Problem.h``  = frozen-core effective one-body integrals over the active orbitals
                     (``mc.get_h1eff()``),
  * ``Problem.g``  = full 4-index chemist-notation (pq|rs) over the active orbitals
                     (``ao2mo.restore(1, mc.get_h2eff(), ncas)``),
  * ``Problem.ecore`` = nuclear repulsion + frozen-core energy (second return of get_h1eff),
  * active orbitals are ordered BY D2h IRREP (reference occupations per irrep from the pinned RHF
    irrep_nelec; Fock energy inside an irrep), occupied first, so the Hartree-Fock determinant is
    "lowest n_alpha alpha + lowest n_beta beta" exactly as ``src.pauli.hf_bitstring`` assumes. We do
    NOT sort by CI occupation (bonding/antibonding occupations approach 1 at dissociation).
  * exact references (CASCI/CASSCF/FCI) are constrained to the SINGLET (fcisolver.spin=0 + fix_spin_
    with ss=0), mirroring the prototype's spin penalty H + mu*S^2 (src.pipeline.prepare(P, mu)).

Symmetric-branch RHF: the molecule is built with D2h symmetry, so RHF orbitals can never
break spatial symmetry (pi orbitals stay pure px / py => point-group parities are Z-strings
under Jordan-Wigner => 12 -> 7 qubit tapering, as in src/n2.py ``_align_degenerate``).
Along the bond coordinate we additionally pin the per-irrep occupation pattern found at
equilibrium (``mf.irrep_nelec``) and continue the density matrix from the previous geometry.

Doc references (verified in source at the tags above):
  gto.M / symmetry       https://pyscf.org/user/gto.html
  scf.RHF, irrep_nelec   https://pyscf.org/user/scf.html  (examples/scf/13-symmetry.py)
  mf.stability(return_status=True) -> (mo_i, mo_e, stable_i, stable_e)   pyscf/scf/stability.py
  mf.newton()            https://pyscf.org/user/scf.html#second-order-scf
  mcscf.CASCI/CASSCF, get_h1eff, get_h2eff, sort_mo, sort_mo_by_irrep
                         https://pyscf.org/user/mcscf.html
  mcscf.avas.avas(mf, ['N 2p'])   https://pyscf.org/user/mcscf.html#avas  (pyscf/mcscf/avas.py)
  ao2mo.restore(1, eri, n)        https://pyscf.org/user/ao2mo.html
  cc.CCSD(mf, frozen=..).ccsd_t() https://pyscf.org/user/cc.html
  custom Hamiltonian RHF/CCSD     examples/scf/40-customizing_hamiltonian.py,
                                  examples/cc/40-ccsd_custom_hamiltonian.py
  fci.direct_spin1.kernel(h1, g, norb, nelec, ecore=)   https://pyscf.org/user/ci.html
  mrpt.NEVPT(mc).kernel()         https://pyscf.org/user/mrpt.html
  block2 DMRGDriver               https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.n2 import Problem  # noqa: E402  (same dataclass our pipeline consumes)

R_EQ = 1.098  # Angstrom
# D2h irreps of N2 along z: sigma_g -> Ag, sigma_u -> B1u, pi_u -> B2u/B3u, pi_g -> B2g/B3g
CAS66_IRREPS = {"Ag": 1, "B1u": 1, "B2u": 1, "B3u": 1, "B2g": 1, "B3g": 1}
CAS66_CORE_IRREPS = {"Ag": 2, "B1u": 2}  # 1sg, 1su (1s cores) + 2sg, 2su


# ----------------------------------------------------------------------------- molecule / RHF
def build_mol(r: float, basis: str = "sto-3g", symmetry: str | bool = "D2h", verbose: int = 0):
    """N2 at bond length ``r`` (Angstrom). Same geometry convention as src/n2.py."""
    from pyscf import gto

    return gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis=basis, unit="Angstrom",
                 symmetry=symmetry, verbose=verbose)


def run_rhf(mol, dm0=None, irrep_nelec: dict | None = None, conv_tol: float = 1e-10,
            check_stability: bool = True, follow_instability: bool = False):
    """Robust RHF that stays on the symmetric branch.

    1. DIIS RHF from ``dm0`` (density of a neighbouring geometry) with optional pinned
       per-irrep occupations; 2. if not converged -> second-order solver ``mf.newton()``;
    3. internal stability analysis is *reported* (``mf._qc02_stable``). Only when
       ``follow_instability`` is True do we step along the unstable mode (that leads to the
       lower, symmetry-broken branch the prototype deliberately avoids).
    """
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = 300
    if irrep_nelec and mol.symmetry:
        mf.irrep_nelec = dict(irrep_nelec)
    mf.kernel(dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.make_rdm1() if dm0 is None else dm0)
    stable = None
    if check_stability:
        try:
            mo_i, _mo_e, stable_i, _stable_e = mf.stability(return_status=True)
            stable = bool(stable_i)
            if follow_instability and not stable_i:
                dm = mf.make_rdm1(mo_i, mf.mo_occ)
                mf = mf.newton()
                mf.kernel(dm)
        except TypeError:  # very old PySCF without return_status (we pin >=2.12)
            stable = None
    mf._qc02_stable = stable  # noqa: SLF001 - informational tag
    return mf


# ----------------------------------------------------------------------------- active spaces
D2H_ORDER = ["Ag", "B1u", "B2u", "B3u", "B2g", "B3g", "B1g", "Au"]


def _active_irreps_and_energies(mc):
    """D2h irrep label and Fock-diagonal energy of each active orbital (columns ncore..ncore+ncas).
    Labels via pyscf.symm.label_orb_symm (pyscf/symm/addons.py); energies e_p = C_p^T F C_p."""
    from pyscf import symm

    mf, mol = mc._scf, mc.mol  # noqa: SLF001 - documented attribute of CASBase
    C = mc.mo_coeff[:, mc.ncore: mc.ncore + mc.ncas]
    eps = np.einsum("pi,pq,qi->i", C, mf.get_fock(), C)
    labels = None
    if mol.symmetry:
        labels = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, C, s=mf.get_ovlp(), check=False))
    return labels, eps


def _irrep_order(mc, na: int):
    """Symmetry-based active-orbital order (replaces occupation sorting, unstable at dissociation).

    The reference determinant is defined by IRREP OCCUPATIONS: n_occ(irrep) = RHF electrons in that
    irrep / 2 minus frozen-core orbitals of that irrep (RHF irrep occupations are pinned along the
    curve, see problems_along). Inside an irrep, orbitals are ordered by Fock energy, which keeps
    bonding below antibonding. Order = occupied (by D2H_ORDER, then energy) + virtual (same).
    Returns (perm, labels) or (None, None) without symmetry."""
    labels, eps = _active_irreps_and_energies(mc)
    if labels is None:
        return None, None
    mf, mol = mc._scf, mc.mol  # noqa: SLF001
    from pyscf import symm

    core_lab = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mc.mo_coeff[:, : mc.ncore],
                                        s=mf.get_ovlp(), check=False)) if mc.ncore else []
    tot = {k: int(np.sum(v)) // 2 for k, v in mf.get_irrep_nelec().items()}
    n_occ = {k: tot.get(k, 0) - core_lab.count(k) for k in set(labels)}
    rank = {k: i for i, k in enumerate(D2H_ORDER)}
    occ, vir = [], []
    for k in sorted(set(labels), key=lambda x: rank.get(x, 99)):
        idx = sorted((i for i, lab in enumerate(labels) if lab == k), key=lambda i: eps[i])
        occ += idx[: max(n_occ[k], 0)]
        vir += idx[max(n_occ[k], 0):]
    if len(occ) != na:  # inconsistent occupation pattern -> let the caller fall back
        return None, labels
    return np.array(occ + vir), labels


def _reorder(h: np.ndarray, g: np.ndarray, perm: np.ndarray):
    return h[np.ix_(perm, perm)], g[np.ix_(perm, perm, perm, perm)]


def _w2p(mol, mf) -> np.ndarray:
    """Mulliken-like N-2p weight of every MO (same criterion as src/n2.py)."""
    idx = mol.search_ao_label(["N 2p"])
    S = mf.get_ovlp()
    C = mf.mo_coeff
    SC = S @ C
    return np.array([(C[idx, i] * SC[idx, i]).sum() for i in range(C.shape[1])])


def _problem_from_mc(mc, r: float, name: str, e_rhf: float, meta: dict) -> Problem:
    from pyscf import ao2mo

    ncas = mc.ncas
    na, nb = (int(x) for x in mc.nelecas)  # PySCF stores nelecas as a tuple (alpha, beta)
    h1, ecore = mc.get_h1eff()
    g = ao2mo.restore(1, mc.get_h2eff(), ncas)
    perm, labels = _irrep_order(mc, na)
    how = "irrep"
    if perm is None:  # no symmetry: aufbau by Fock energy (NOT by CI occupation)
        _, eps = _active_irreps_and_energies(mc)
        perm, how = np.argsort(eps, kind="stable"), "fock_energy"
    h1, g = _reorder(np.asarray(h1), np.asarray(g), perm)
    meta = dict(meta, active_perm=perm.tolist(), active_order=how,
                active_irreps=[labels[i] for i in perm] if labels else None)
    return Problem(r, name, h1, g, float(ecore), ncas, int(na), int(nb), float(e_rhf), meta)


def _singlet(mc):
    """Constrain CASCI/CASSCF to the singlet: fcisolver.spin = 0 and S^2 penalty (fix_spin_, ss=0)."""
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0)
    return mc


def cas66_rhf_irrep(mf, r: float) -> tuple[Problem, object]:
    """Cross-check mode (default for STO-3G) = the prototype's CAS(6e,6o) chosen BY IRREP: CASCI on
    RHF orbitals, active {3sg (Ag), 3su (B1u), pi_u (B2u,B3u), pi_g (B2g,B3g)}, frozen {1sg,2sg,1su,2su}."""
    from pyscf import mcscf

    mc = _singlet(mcscf.CASCI(mf, 6, 6))
    mo = mc.sort_mo_by_irrep(CAS66_IRREPS, CAS66_CORE_IRREPS, mo_coeff=mf.mo_coeff)
    mc.kernel(mo)
    P = _problem_from_mc(mc, r, "CAS(6e,6o)", mf.e_tot, {"mode": "rhf_irrep", "e_casci": float(mc.e_tot)})
    return P, mc


def cas66_rhf_w2p(mf, r: float) -> tuple[Problem, object]:
    """Prototype's literal criterion (3 occupied MOs with most N-2p weight + 3 lowest virtuals).
    Kept for comparison only; use cas66_rhf_irrep (identical set when the RHF is D2h-adapted)."""
    from pyscf import mcscf

    mol = mf.mol
    nocc = mol.nelectron // 2
    w = _w2p(mol, mf)
    val_occ = list(range(2, nocc))
    act_occ = sorted(sorted(val_occ, key=lambda i: -w[i])[:3])
    act = act_occ + [nocc, nocc + 1, nocc + 2]
    mc = _singlet(mcscf.CASCI(mf, 6, 6))
    mo = mc.sort_mo(act, base=0)  # base=0: 0-based indices
    mc.kernel(mo)
    P = _problem_from_mc(mc, r, "CAS(6e,6o)", mf.e_tot, {"mode": "rhf_w2p", "active_mo": act,
                                                          "e_casci": float(mc.e_tot)})
    return P, mc


def cas_rhf(mf, r: float, ncas: int, nelecas: int = 10, ncore: int = 2) -> tuple[Problem, object]:
    """General CASCI on RHF orbitals: freeze the ``ncore`` lowest (1s) orbitals, take the next ``ncas``.
    ncas=8 -> the prototype's FV(10e,8o); ncas=16 in cc-pVDZ -> a 32-qubit space for LUCJ+SQD on
    hardware. Active ORDER is by irrep (see _irrep_order); selection is by orbital energy."""
    from pyscf import mcscf

    mc = mcscf.CASCI(mf, ncas, nelecas)
    _singlet(mc)
    mc.mo_coeff = mc.sort_mo(list(range(ncore, ncore + ncas)), base=0)
    if ncas <= 12:  # CASCI/FCI reference only where it is cheap; larger spaces -> DMRG / SQD
        mc.kernel(mc.mo_coeff)
    name = "FV(10e,8o)" if (ncas, nelecas) == (8, 10) else f"CAS({nelecas}e,{ncas}o)"
    P = _problem_from_mc(mc, r, name, mf.e_tot,
                         {"mode": "rhf_cas", "e_casci": float(mc.e_tot) if mc.e_tot is not None else None})
    return P, mc


def fv108_rhf(mf, r: float) -> tuple[Problem, object]:
    """Prototype's full-valence FV(10e,8o): CASCI on RHF orbitals, freeze the two 1s cores."""
    return cas_rhf(mf, r, 8, 10)


def cas66_casscf(mf, r: float, selection: str = "irrep", guess_mo=None) -> tuple[Problem, object]:
    """Production mode: CASSCF(6,6) orbitals; initial active space by D2h irreps (default) or
    AVAS on N 2p (``selection='avas'``)."""
    from pyscf import mcscf

    if selection == "avas":
        from pyscf.mcscf import avas

        ncas, nelecas, mo = avas.avas(mf, ["N 2p"], canonicalize=True)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
    else:
        mc = mcscf.CASSCF(mf, 6, 6)
        # method of the symmetry-adapted CASSCF class (pyscf/mcscf/mc1step_symm.py)
        mo = guess_mo if guess_mo is not None else \
            mc.sort_mo_by_irrep(CAS66_IRREPS, CAS66_CORE_IRREPS, mo_coeff=mf.mo_coeff)
    _singlet(mc)
    mc.conv_tol = 1e-10
    mc.max_cycle_macro = 200
    mc.kernel(mo)
    meta = {"mode": f"casscf_{selection}", "e_casscf": float(mc.e_tot), "converged": bool(mc.converged)}
    P = _problem_from_mc(mc, r, f"CASSCF({sum(mc.nelecas)}e,{mc.ncas}o)", mf.e_tot, meta)
    return P, mc


def parse_cas(cas: str) -> tuple[int, int]:
    """'cas66' -> (6, 6); 'fv108' -> (10, 8); 'cas10e16o' -> (10, 16)."""
    if cas == "cas66":
        return 6, 6
    if cas == "fv108":
        return 10, 8
    import re

    m = re.fullmatch(r"cas(\d+)e(\d+)o", cas)
    if not m:
        raise ValueError(f"unknown active space {cas!r} (use cas66, fv108, cas<N>e<M>o)")
    return int(m.group(1)), int(m.group(2))


def n2_problem_pyscf(r: float, basis: str = "sto-3g", cas: str = "cas66", mode: str | None = None,
                     dm0=None, irrep_nelec: dict | None = None):
    """Build one Problem. ``cas``: 'cas66', 'fv108' or 'cas<N>e<M>o' (e.g. 'cas10e16o' in cc-pVDZ).
    CAS(6e,6o) ``mode``: 'rhf_irrep' (STO-3G default, = prototype), 'rhf_w2p', 'casscf_irrep'
    (production default), 'casscf_avas'. Other spaces: CASCI on RHF orbitals. Returns (P, mf, mc)."""
    if mode is None:
        mode = "rhf_irrep" if basis.lower().replace("-", "") == "sto3g" else "casscf_irrep"
    mol = build_mol(r, basis)
    mf = run_rhf(mol, dm0=dm0, irrep_nelec=irrep_nelec)
    nel, norb = parse_cas(cas)
    if cas != "cas66":
        P, mc = cas_rhf(mf, r, norb, nel)
    elif mode == "rhf_irrep":
        P, mc = cas66_rhf_irrep(mf, r)
    elif mode == "rhf_w2p":
        P, mc = cas66_rhf_w2p(mf, r)
    else:
        P, mc = cas66_casscf(mf, r, selection=mode.split("_", 1)[1])
    P.meta.update({"basis": basis, "rhf_stable_internal": getattr(mf, "_qc02_stable", None),
                   "irrep_nelec": _irrep_nelec(mf)})
    return P, mf, mc


def _irrep_nelec(mf) -> dict | None:
    try:
        return {k: int(np.sum(v)) for k, v in mf.get_irrep_nelec().items()}
    except Exception:  # noqa: BLE001 - no symmetry
        return None


def problems_along(rs: list[float], basis: str = "sto-3g", cas: str = "cas66", mode: str | None = None):
    """Follow the symmetric RHF branch: equilibrium first, then outward / inward, pinning the
    equilibrium irrep occupations and continuing the density matrix (cf. src/pipeline.py)."""
    P_eq, mf_eq, mc_eq = n2_problem_pyscf(R_EQ, basis, cas, mode)
    pattern = mf_eq.get_irrep_nelec() if mf_eq.mol.symmetry else None
    out = {R_EQ: (P_eq, mf_eq, mc_eq)}
    for branch in (sorted(r for r in rs if r > R_EQ), sorted((r for r in rs if r < R_EQ), reverse=True)):
        dm = mf_eq.make_rdm1()
        for r in branch:
            P, mf, mc = n2_problem_pyscf(r, basis, cas, mode, dm0=dm, irrep_nelec=pattern)
            out[r] = (P, mf, mc)
            dm = mf.make_rdm1()
    return {r: out[r] for r in sorted(out) if r in rs or r == R_EQ}


# ----------------------------------------------------------------------------- benchmarks
def full_space_cc(mf, frozen: int = 2) -> dict:
    """Full-basis CCSD and CCSD(T) with the 1s cores frozen (context curves)."""
    from pyscf import cc

    out = {"hf": float(mf.e_tot)}
    try:
        mycc = cc.CCSD(mf, frozen=frozen)
        mycc.conv_tol = 1e-9
        mycc.max_cycle = 500
        mycc.kernel()
        ok = bool(mycc.converged)
        out.update(ccsd=float(mycc.e_tot) if ok else float("nan"),
                   ccsd_t=float(mycc.e_tot + mycc.ccsd_t()) if ok else float("nan"), cc_converged=ok)
    except Exception as exc:  # noqa: BLE001 - CC genuinely diverges at long bonds
        out.update(ccsd=float("nan"), ccsd_t=float("nan"), cc_converged=False, error=repr(exc))
    return out


def model_mf(P: Problem):
    """RHF on the active-space model Hamiltonian (reference for active-space CC / LUCJ).
    Pattern of examples/scf/40-customizing_hamiltonian.py."""
    from pyscf import ao2mo, gto, scf

    n = P.n_orb
    mol = gto.M(verbose=0)
    mol.nelectron = P.n_alpha + P.n_beta
    mol.spin = P.n_alpha - P.n_beta
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: P.h
    mf.get_ovlp = lambda *a: np.eye(n)
    mf.energy_nuc = lambda *a: P.ecore
    mf._eri = ao2mo.restore(8, P.g, n)  # noqa: SLF001 - documented PySCF hook
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    dm0 = np.zeros((n, n))
    dm0[np.arange(P.n_alpha), np.arange(P.n_alpha)] = 2.0  # start from the HF determinant
    mf.kernel(dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(dm0)
    return mf


def exact_singlet(P: Problem, max_norb: int = 12) -> dict:
    """FCI constrained to the singlet: direct_spin1 solver with spin=0 and an S^2 level shift
    (fci.addons.fix_spin_(solver, ss=0)); also the unconstrained lowest Sz=0 root for reference.
    At long bonds the triplet/quintet become (near-)degenerate or lower, so the constraint matters
    (this mirrors the prototype's pipeline.prepare(P, mu): H + mu*S^2, pr.e_exact = singlet)."""
    from pyscf import fci

    if P.n_orb > max_norb:
        return {"exact": None, "note": f"FCI skipped for {P.n_orb} orbitals (use dmrg_energy)"}
    nelec = (P.n_alpha, P.n_beta)
    e_any, ci_any = fci.direct_spin1.kernel(P.h, P.g, P.n_orb, nelec, ecore=P.ecore, conv_tol=1e-12, max_cycle=500)
    solver = fci.direct_spin1.FCI()
    solver.spin = 0
    solver.conv_tol = 1e-12
    solver.max_cycle = 500
    fci.addons.fix_spin_(solver, ss=0)
    e_s, ci_s = solver.kernel(P.h, P.g, P.n_orb, nelec, ecore=P.ecore)
    s2 = float(fci.spin_op.spin_square0(ci_s, P.n_orb, nelec)[0])
    s2_any = float(fci.spin_op.spin_square0(ci_any, P.n_orb, nelec)[0])
    return {"exact": float(e_s), "s2": s2, "exact_any_spin": float(e_any), "s2_any_spin": s2_any}


def active_space_benchmarks(P: Problem) -> dict:
    """HF / CCSD / CCSD(T) / exact singlet (FCI) on the identical active-space Hamiltonian.

    NOTE (panel): in CAS(6e,6o) with D2h-adapted orbitals the (T) correction is identically zero
    (prototype results/baseline.json: ccsd == ccsd_t for cas66 at every R), so CCSD(T) must be
    benchmarked in the full-valence space (``full_valence_benchmarks``) or the full space
    (``full_space_cc``), not in CAS(6e,6o)."""
    from pyscf import cc

    out = exact_singlet(P)
    if P.n_orb == 6 and P.n_alpha == 3:
        out["ccsd_t_note"] = "(T)=0 identically in CAS(6e,6o)/D2h; use full_valence or full-space CCSD(T)"
    mf = model_mf(P)
    out["hf"] = float(mf.e_tot)
    try:
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.max_cycle = 500
        mycc.kernel()
        ok = bool(mycc.converged)
        out.update(ccsd=float(mycc.e_tot) if ok else float("nan"),
                   ccsd_t=float(mycc.e_tot + mycc.ccsd_t()) if ok else float("nan"), cc_converged=ok)
    except Exception as exc:  # noqa: BLE001
        out.update(ccsd=float("nan"), ccsd_t=float("nan"), cc_converged=False, error=repr(exc))
    return out


def full_valence_benchmarks(mf, r: float) -> dict:
    """CCSD(T) vs exact singlet in FV(10e,8o) on RHF orbitals (where (T) is non-trivial)."""
    P, _ = cas_rhf(mf, r, 8, 10)
    return active_space_benchmarks(P)


def nevpt2_correction(mc) -> float | None:
    """Strongly-contracted NEVPT2 correction on top of a CASCI/CASSCF object (optional)."""
    try:
        from pyscf import mrpt

        return float(mrpt.NEVPT(mc).kernel())
    except Exception:  # noqa: BLE001
        return None


def dmrg_energy(P: Problem, bond_dims=(250, 500, 500), n_sweeps: int = 20, scratch: str = "./tmp_block2",
                n_threads: int = 4) -> float | None:
    """DMRG reference with block2 (SU2 mode). Returns None if block2 is not installed.
    API: block2 0.5.4 pyblock2/driver/core.py (DMRGDriver.initialize_system / get_qc_mpo /
    get_random_mps / dmrg), tutorial https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html
    """
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        print("[dmrg] block2 not installed -> skipped (pip install block2)")
        return None
    driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2, n_threads=n_threads)
    driver.initialize_system(n_sites=P.n_orb, n_elec=P.n_alpha + P.n_beta, spin=P.n_alpha - P.n_beta)
    mpo = driver.get_qc_mpo(h1e=P.h, g2e=P.g, ecore=P.ecore, iprint=0)
    ket = driver.get_random_mps(tag="GS", bond_dim=bond_dims[0], nroots=1)
    sched_m = list(bond_dims) + [bond_dims[-1]] * max(0, n_sweeps - len(bond_dims))
    noises = [1e-4] * (n_sweeps // 2) + [0.0] * (n_sweeps - n_sweeps // 2)
    thrds = [1e-10] * n_sweeps
    e = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=sched_m[:n_sweeps], noises=noises,
                    thrds=thrds, iprint=0)
    return float(e)


def rotate_problem(P: Problem, C: np.ndarray, tag: str = "model-HF basis") -> Problem:
    """Express P in a rotated orthonormal active basis (columns of C): h' = C^T h C, g' likewise."""
    h = C.T @ P.h @ C
    g = np.einsum("pqrs,pi,qj,rk,sl->ijkl", P.g, C, C, C, C, optimize=True)
    return replace(P, h=h, g=g, meta=dict(P.meta, basis_rotation=tag))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PySCF N2 active-space builder + benchmarks")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--r", type=float, nargs="+", default=[R_EQ])
    ap.add_argument("--cas", default="cas66", choices=["cas66", "fv108"])
    ap.add_argument("--mode", default=None, choices=[None, "rhf_w2p", "casscf_irrep", "casscf_avas"])
    a = ap.parse_args()
    for r, (P, mf, mc) in problems_along(a.r, a.basis, a.cas, a.mode).items():
        b = active_space_benchmarks(P)
        print(f"R={r:.3f} RHF={mf.e_tot:.6f} stable={mf._qc02_stable} {P.name} "
              f"exact={b['exact']:.6f} CCSD(T)={b.get('ccsd_t', float('nan')):.6f}")
