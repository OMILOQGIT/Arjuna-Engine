# Production port. Written against PySCF 2.14.0, Qiskit SDK 2.6.1, qiskit-nature 0.8.0,
# qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, qiskit-addon-sqd 0.14.0, mitiq 1.1.0, ffsim 0.0.85
# docs/source on 2026-10-27 (URLs in port/README.md). Not executed in the authoring environment
# (no package access); run port/test_port.py on a laptop first.
"""End-to-end QC-02 pipeline on the real stack, one CLI:

    python port/run_all.py --basis sto-3g --r 1.098 2.1
    python port/run_all.py --basis cc-pvdz --r 1.098 1.5 2.1 --backend FakeFez --lucj

Stages (each skipped with a message if its library is missing):
  chem (PySCF) -> classical benchmarks -> qubit H (ours + qiskit-nature cross-check) -> our ADAPT
  -> Qiskit circuit + transpiled counts on a fake IBM backend -> Aer noisy estimate (+ZNE, +REM)
  -> SQD (qiskit-addon-sqd) on noisy samples of a shallow circuit -> [optional LUCJ via ffsim].
Results: printed table + JSON in results/port/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline import first_chem_acc, prepare, run_adapt  # noqa: E402

CHEM_ACC = 1.6e-3


def have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def stage(name: str, out: dict, fn, *a, needs: tuple[str, ...] = (), **kw):
    """Run one stage; record result or the reason it was skipped/failed."""
    missing = [m for m in needs if not have(m)]
    if missing:
        msg = f"skipped: missing {', '.join(missing)} (pip install -r requirements-laptop.txt)"
        print(f"  [{name}] {msg}")
        out[name] = {"status": msg}
        return None
    t = time.time()
    try:
        val = fn(*a, **kw)
        print(f"  [{name}] done in {time.time() - t:.1f}s")
        return val
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, keep going
        print(f"  [{name}] FAILED: {exc!r}")
        traceback.print_exc(limit=3)
        out[name] = {"status": f"failed: {exc!r}"}
        return None


# ----------------------------------------------------------------------------- chemistry
def build_problems(rs, basis: str, cas: str, mode: str | None):
    """{r: (Problem, mf_or_None, mc_or_None)} -- PySCF if available, else our STO-3G mock."""
    if have("pyscf"):
        from chem_pyscf import problems_along

        return problems_along(list(rs), basis, cas, mode)
    if basis.lower().replace("-", "") != "sto3g":
        raise RuntimeError("PySCF is required for basis != sto-3g")
    print("  [chem] PySCF missing -> using the verified NumPy mock (src/n2.py), STO-3G only")
    from src.pipeline import problems_along

    return {r: (p[cas], None, None) for r, p in problems_along(list(rs)).items()}


def build_problem(r: float, basis: str = "sto-3g", cas: str = "cas66", mode: str | None = None):
    return build_problems([r], basis, cas, mode)[r][0]


def classical(P, mf, mc, dmrg: bool) -> dict:
    from chem_pyscf import active_space_benchmarks, dmrg_energy, full_space_cc, nevpt2_correction

    from chem_pyscf import full_valence_benchmarks

    out = {"active": active_space_benchmarks(P)}
    if mf is not None:
        out["full"] = full_space_cc(mf)
        if P.n_orb == 6:  # (T) == 0 identically in CAS(6e,6o)/D2h -> benchmark CCSD(T) in full valence
            out["full_valence"] = full_valence_benchmarks(mf, P.r)
    if mc is not None:
        out["nevpt2_corr"] = nevpt2_correction(mc)
    if dmrg:
        out["dmrg"] = dmrg_energy(P)
    return out


# ----------------------------------------------------------------------------- qubits
def nature_check(P, pr) -> dict:
    from qubit_nature import ground_energy_in_sector, nature_qubit_hamiltonian, sparse_pauli_to_our

    H_jw, _ = nature_qubit_hamiltonian(P, taper=False)
    e_n = ground_energy_in_sector(H_jw, P.n_orb, P.n_alpha, P.n_beta)
    ours = pr.H
    theirs = sparse_pauli_to_our(H_jw)
    keys = set(ours) | set(theirs)
    max_diff = max(abs(ours.get(k, 0) - theirs.get(k, 0)) for k in keys)
    H_t, _ = nature_qubit_hamiltonian(P, taper=True)
    e_any = getattr(pr, "e_lowest_any", None) or pr.e_exact  # sector ground state, any spin
    return {"nature_jw_ground": e_n, "delta_vs_ours": e_n - e_any, "max_coeff_diff": float(max_diff),
            "nature_tapered_qubits": H_t.num_qubits, "nature_terms": len(H_jw)}


# ----------------------------------------------------------------------------- noisy sampling for SQD
def sample_noisy(qc_measured, backend, shots: int, seed: int = 3):
    """Aer SamplerV2 on the fake backend noise model; returns BitArray (register 'meas')."""
    from qiskit_aer.primitives import SamplerV2

    from circuits_qiskit import transpile_isa
    from run_aer import noisy_simulator

    isa = transpile_isa(qc_measured, backend)
    sampler = SamplerV2.from_backend(noisy_simulator(backend, method="automatic", seed=seed))
    return sampler.run([isa], shots=shots).result()[0].data.meas


def sqd_stage(P, pr, res, k: int, backend, shots: int, max_frac: float) -> dict:
    """Honest SQD: capped subspace (fraction printed), random-bitstring and CISD-ranked baselines
    at the same size. CAS(6e,6o) has only 400 determinants -> results there are NOT evidence."""
    from circuits_qiskit import circuit_from_adapt
    from sqd_ibm import (bit_array_to_ints, cisd_ranked_baseline, honest_report, max_dim_for_fraction,
                         random_baseline, run_sqd, run_sqd_teaching, untaper_bit_array)

    if P.n_orb <= 6:
        print("  [sqd] WARNING: tiny space (CAS(6e,6o) = 400 dets); use --cas fv108 for a meaningful test")
    qc = circuit_from_adapt(pr, res, k, tapered=True, measure=True)
    bits_t = sample_noisy(qc, backend, shots)
    bits_full = untaper_bit_array(pr.tp, bits_t)
    if not have("qiskit_addon_sqd"):
        print("  [sqd] qiskit-addon-sqd missing -> teaching SQD (src/sqd.py), UNCAPPED")
        out = run_sqd_teaching(P, bit_array_to_ints(bits_full))
        out.update(k=k, shots=shots, err_mHa=1e3 * (out["energy"] - pr.e_exact))
        return out
    md = max_dim_for_fraction(P, max_frac)
    kw = dict(samples_per_batch=min(300, shots), num_batches=5, max_iterations=10, max_dim=md)
    out = {"device": run_sqd(P, bits_full, **kw), "max_dim_per_spin": md}
    out["random"] = random_baseline(P, int(bits_full.num_shots), **kw)
    if have("pyscf"):
        try:
            out["cisd_ranked"] = cisd_ranked_baseline(P, md)
        except Exception as exc:  # noqa: BLE001
            out["cisd_ranked"] = {"status": f"failed: {exc!r}"}
    for name in ("device", "random", "cisd_ranked"):
        if "energy" in out.get(name, {}):
            honest_report(name, out[name], pr.e_exact)
    out.update(k=k, shots=shots, energy=out["device"]["energy"], err_mHa=1e3 * (out["device"]["energy"] - pr.e_exact),
               random_err_mHa=1e3 * (out["random"]["energy"] - pr.e_exact),
               subspace_fraction=out["device"]["subspace_fraction"])
    return out


def lucj_stage(P, backend, shots: int, max_frac: float) -> dict:
    from lucj_ffsim import lucj_circuit, lucj_statevector_energy
    from sqd_ibm import run_sqd

    got = lucj_circuit(P, n_reps=1, backend=backend)
    if got is None:
        return {"status": "skipped"}
    qc, isa, P_mo, pairs = got
    from qiskit_aer.primitives import SamplerV2

    from run_aer import noisy_simulator

    bits = SamplerV2.from_backend(noisy_simulator(backend, method="automatic")).run([isa], shots=shots).result()[0].data.meas
    from sqd_ibm import max_dim_for_fraction, random_baseline

    kw = dict(samples_per_batch=min(300, shots), num_batches=5, max_iterations=10,
              max_dim=max_dim_for_fraction(P_mo, max_frac))
    sq = run_sqd(P_mo, bits, **kw)
    sq["random_baseline"] = random_baseline(P_mo, int(bits.num_shots), **kw)
    two_q = sum(v for kk, v in isa.count_ops().items() if kk in ("cz", "ecr", "cx"))
    return {"lucj_ideal": lucj_statevector_energy(P_mo, qc), "two_qubit": int(two_q), "pairs": pairs, "sqd": sq}


# ----------------------------------------------------------------------------- main
def run_geometry(r, P, mf, mc, args) -> dict:
    out: dict = {"r": r, "problem": P.name, "e_rhf_full": P.e_rhf_full, "meta": P.meta}
    print(f"\n=== R = {r:.3f} A  {P.name}  basis={args.basis} ===")
    if mf is not None:
        val = stage("classical", out, classical, P, mf, mc, args.dmrg, needs=("pyscf",))
        if val is not None:
            out["classical"] = val
    pr = stage("qubits", out, prepare, P, mu=args.mu)  # H + mu*S^2 for ADAPT; pr.e_exact = singlet
    if pr is None:
        return out
    out["qubits"] = {"n_qubits": 2 * P.n_orb, "n_tapered": pr.tp.n_tapered, "terms": len(pr.H),
                     "terms_tapered": len(pr.Ht), "exact": pr.e_exact, "pool": len(pr.pool_t), "mu": args.mu,
                     "e_lowest_any_spin": getattr(pr, "e_lowest_any", None),
                     "s2_lowest_any_spin": getattr(pr, "s2_lowest_any", None)}
    val = stage("nature", out, nature_check, P, pr, needs=("qiskit_nature",))
    if val is not None:
        out["nature"] = val
    res = stage("adapt", out, run_adapt, pr, max_ops=args.max_ops)
    if res is None:
        return out
    k_ca = first_chem_acc(res, pr.e_exact)
    k_final = len(res.ops)
    out["adapt"] = {"energies": res.energies, "labels": res.labels, "k_chem_acc": k_ca, "n_ops": k_final,
                    "final_err_mHa": 1e3 * (res.energies[-1] - pr.e_exact), "s2": getattr(res, "s2", None)}
    if not have("qiskit"):
        print("  [qiskit] skipped: qiskit missing")
        return out
    from circuits_qiskit import circuit_from_adapt, statevector_energy, transpile_counts
    from qubit_nature import our_to_sparse_pauli

    H_t = our_to_sparse_pauli(pr.Ht, pr.tp.n_tapered)
    k_use = k_ca if k_ca is not None else k_final
    qc = circuit_from_adapt(pr, res, k_use, tapered=True)
    out["circuit"] = {"k": k_use, "sv_energy": statevector_energy(qc, H_t),
                      "adapt_energy": res.energies[k_use]}
    if have("qiskit_ibm_runtime"):
        tc = stage("transpile", out, transpile_counts, qc, args.backend, needs=("qiskit_ibm_runtime",))
        if tc is not None:
            out["transpile"] = tc
        tc_full = stage("transpile_untapered", out, transpile_counts,
                        circuit_from_adapt(pr, res, k_use, tapered=False), args.backend)
        if tc_full is not None:
            out["transpile_untapered"] = tc_full
    else:
        print("  [transpile] skipped: qiskit-ibm-runtime (fake backends) missing")
        return out
    from circuits_qiskit import fake_backend

    backend = fake_backend(args.backend)
    k_sh = min(args.k_shallow, k_final)
    if have("qiskit_aer"):
        from run_aer import noisy_report

        val = stage("aer_noisy", out, noisy_report, circuit_from_adapt(pr, res, k_sh, tapered=True), H_t, backend,
                    shots=args.shots, try_runtime_local=args.runtime_local)
        if val is not None:
            val["k"] = k_sh
            val["adapt_energy_k"] = res.energies[k_sh]
            out["aer_noisy"] = val
        val = stage("sqd", out, sqd_stage, P, pr, res, k_sh, backend, args.sqd_shots, args.max_frac)
        if val is not None:
            out["sqd"] = val
        if args.lucj:
            val = stage("lucj", out, lucj_stage, P, backend, args.sqd_shots, args.max_frac,
                        needs=("ffsim", "pyscf", "qiskit_addon_sqd"))
            if val is not None:
                out["lucj"] = val
    else:
        print("  [aer] skipped: qiskit-aer missing")
    return out


def fmt(x, scale=1.0, nd=3):
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * scale:.{nd}f}"


def table(rows: list[dict]):
    print("\n" + "=" * 118)
    print(f"{'R':>6} {'exact':>13} {'CCSD(T)err':>10} {'ADAPT ops':>9} {'ADAPTerr':>9} {'2q(tap)':>8} "
          f"{'2q(full)':>8} {'noisy err':>10} {'ZNE-R err':>10} {'REM err':>9} {'SQD err':>9} {'SQD frac':>9} {'rand err':>9}  (errors in mHa)")
    for o in rows:
        ex = o.get("qubits", {}).get("exact")
        if ex is None:
            print(f"{o['r']:6.3f}  (incomplete: see JSON)")
            continue
        fv = o.get("classical", {}).get("full_valence")
        ccsdt = o.get("classical", {}).get("active", {}).get("ccsd_t")
        if fv and fv.get("exact") is not None:  # (T) is zero in CAS(6e,6o): report the FV(10e,8o) error
            ccsdt = None if fv.get("ccsd_t") is None else fv["ccsd_t"] - fv["exact"] + ex
        an = o.get("aer_noisy", {})
        rel = (lambda v: None if v is None else v - ex)
        print(f"{o['r']:6.3f} {ex:13.6f} {fmt(rel(ccsdt), 1e3):>10} {str(o.get('adapt', {}).get('k_chem_acc')):>9} "
              f"{fmt(o.get('adapt', {}).get('final_err_mHa')):>9} {str(o.get('transpile', {}).get('two_qubit', '-')):>8} "
              f"{str(o.get('transpile_untapered', {}).get('two_qubit', '-')):>8} "
              f"{fmt(rel(an.get('noisy')), 1e3, 1):>10} {fmt(rel(an.get('zne', {}).get('richardson')), 1e3, 1):>10} "
              f"{fmt(rel(an.get('sampled', {}).get('rem')), 1e3, 1):>9} {fmt(o.get('sqd', {}).get('err_mHa')):>9} "
              f"{fmt(o.get('sqd', {}).get('subspace_fraction'), 1.0, 3):>9} {fmt(o.get('sqd', {}).get('random_err_mHa')):>9}")
    print("ZNE/REM/noisy columns refer to the shallow circuit (k_shallow ops); compare with its ideal energy in the JSON.")
    print("CCSD(T)err: FV(10e,8o) error when the active space is CAS(6e,6o) ((T)=0 there). SQD: capped subspace;"
          " 'rand err' = same SQD call on uniform random bitstrings. Exact = singlet.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--r", type=float, nargs="+", default=[1.098, 2.1])
    ap.add_argument("--cas", default="cas66", help="cas66 | fv108 | cas<N>e<M>o (PySCF only)")
    ap.add_argument("--mu", type=float, default=0.5, help="spin penalty H + mu*S^2 for ADAPT (prototype default)")
    ap.add_argument("--max-frac", type=float, default=0.05, help="SQD subspace cap: dim/full_dim <= max_frac")
    ap.add_argument("--mode", default=None, choices=[None, "rhf_irrep", "rhf_w2p", "casscf_irrep", "casscf_avas"],
                    help="CAS(6e,6o) orbitals (default: rhf_irrep for sto-3g, casscf_irrep otherwise)")
    ap.add_argument("--backend", default="FakeFez", help="class name in qiskit_ibm_runtime.fake_provider")
    ap.add_argument("--max-ops", type=int, default=60)
    ap.add_argument("--k-shallow", type=int, default=4, help="ADAPT ops in the circuit used for noisy runs/SQD")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--sqd-shots", type=int, default=3000)
    ap.add_argument("--dmrg", action="store_true")
    ap.add_argument("--lucj", action="store_true")
    ap.add_argument("--runtime-local", action="store_true", help="also try client-side runtime Estimator locally")
    args = ap.parse_args(argv)

    print("libraries:", {m: have(m) for m in ("pyscf", "qiskit", "qiskit_nature", "qiskit_aer", "qiskit_ibm_runtime",
                                              "qiskit_addon_sqd", "mitiq", "ffsim", "pyblock2")})
    probs = build_problems(args.r, args.basis, args.cas, args.mode)
    rows = [run_geometry(r, *probs[r], args) for r in sorted(probs) if r in args.r]
    table(rows)
    dest = ROOT / "results" / "port"
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"run_{args.basis.replace('-', '')}_{args.cas}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, default=lambda o: o.tolist()
                            if hasattr(o, "tolist") else str(o)))
    print(f"saved {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
