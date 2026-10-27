"""Noisy execution of the chemically-accurate ADAPT circuit + error-mitigation ablation.

For R = 1.098 A (equilibrium) and 2.10 A (strongly correlated; CCSD(T) fails):
  1. truncate ADAPT at the first operator count reaching chemical accuracy (noiseless)
  2. validate: noiseless gate-level density-matrix sim == ADAPT statevector energy
  3. ablation at a default hardware-like noise model: raw / REM / REM+ZNE(linear, Richardson,
     exponential) / PEC (unbiased in expectation; report sampling overhead)
  4. sweep the 2q error rate to find where mitigated error < 1.6 mHa
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.noise import DensitySim, Estimator, NoiseModel, count_gates, pec_overhead, zne  # noqa: E402
from src.pipeline import circuit_for, first_chem_acc, prepare, problems_along, run_adapt  # noqa: E402

CHEM = 1.6e-3
SCALES = [1, 3, 5]
R2_SWEEP = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
geoms = [1.098, 2.10]
probs_at = problems_along([0.90, 1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10])
out = {}


def mitigated(sim, est, gates, ref, nm):
    vals = {}
    rhos = {s: sim.run(gates, ref, nm, fold=s) for s in SCALES}
    vals["raw"] = est.energy(rhos[1], p_ro=nm.p_ro)
    rem = [est.energy(rhos[s], p_ro=nm.p_ro, rem=True) for s in SCALES]
    vals["rem"] = rem[0]
    vals["rem+zne_lin"] = zne(SCALES[:2], rem[:2], "linear")
    vals["rem+zne_rich"] = zne(SCALES, rem, "richardson")
    vals["rem+zne_exp"] = zne(SCALES, rem, "exponential")
    vals["rem+zne_exp_asym"] = zne(SCALES, rem, "exponential_asym", asymptote=est.const)
    return vals, rhos[1]


for r in geoms:
    t0 = time.time()
    pr = prepare(probs_at[r]["cas66"])
    res = run_adapt(pr)
    # depth with headroom: first circuit whose noise-free error is <= half the 1.6 mHa budget,
    # so the noise analysis is not dominated by an ansatz that only just reaches the target
    k = first_chem_acc(res, pr.e_exact, thr=0.8e-3)
    gates = circuit_for(pr, res, k)
    n = pr.tp.n_tapered
    sim, est = DensitySim(n), Estimator(pr.Ht, n)
    ideal = est.energy(sim.run(gates, pr.tp.ref_bits, None))
    assert abs(ideal - res.energies[k]) < 1e-7, (ideal, res.energies[k])
    gc = count_gates(gates)
    print(f"\nR={r}: {n} qubits, ADAPT ops to chem.acc k={k}, gates {gc}; noiseless circuit "
          f"err {1e3*(ideal-pr.e_exact):.4f} mHa (matches statevector)", flush=True)

    nm = NoiseModel()
    vals, rho1 = mitigated(sim, est, gates, pr.tp.ref_bits, nm)
    gamma = pec_overhead(gates, nm)
    # PEC with exact knowledge of the (depolarising) noise is unbiased: mean = ideal energy.
    # Its cost: estimator std inflates by gamma -> shots needed for 1.6 mHa precision.
    ev = est.pauli_expectations(rho1)
    sigma1 = float(np.sqrt(np.sum(est.coefs ** 2 * np.clip(1 - ev ** 2, 0, None))))  # 1 shot per term
    budget = CHEM - abs(ideal - pr.e_exact)  # precision left after the ansatz's own error
    shots_pec = (gamma * sigma1 / budget) ** 2  # per Pauli term, no grouping
    shots_plain = (sigma1 / budget) ** 2
    # PEC is not listed as an energy: with exact noise knowledge its mean equals the noise-free
    # circuit by construction; only its cost (gamma, shots) is reported.
    abl = {kname: {"E": v, "err_mHa": 1e3 * (v - pr.e_exact)} for kname, v in vals.items()}
    for kname, d in abl.items():
        print(f"   {kname:14s} err {d['err_mHa']:+10.3f} mHa", flush=True)
    print(f"   PEC sampling overhead gamma={gamma:.3e} -> ~{shots_pec:.2e} shots/term for the remaining budget "
          f"(vs {shots_plain:.2e} unmitigated)", flush=True)

    sweep = []
    for r2 in R2_SWEEP:
        nm_s = NoiseModel(r2=r2, r1=r2 / 12, p_ro=1.5e-2)
        v, _ = mitigated(sim, est, gates, pr.tp.ref_bits, nm_s)
        g = pec_overhead(gates, nm_s)
        row = {"r2": r2, **{kk: 1e3 * (vv - pr.e_exact) for kk, vv in v.items()}, "pec_gamma": g}
        sweep.append(row)
        print(f"   r2={r2:.0e}: raw {row['raw']:+9.3f}  rem {row['rem']:+9.3f}  zne_rich {row['rem+zne_rich']:+9.3f}"
              f"  zne_exp {row['rem+zne_exp']:+9.3f} mHa  PEC gamma {g:.2e}", flush=True)
    out[str(r)] = {"qubits": n, "k_chem_acc": k, "gates": gc, "noiseless_err_mHa": 1e3 * (ideal - pr.e_exact),
                   "noise_model": vars(nm), "ablation": abl, "pec_gamma": gamma, "shots_pec_per_term": shots_pec,
                   "shots_plain_per_term": shots_plain, "sweep": sweep, "seconds": time.time() - t0}
(ROOT / "results" / "noise.json").write_text(json.dumps(out, indent=1))
print("saved results/noise.json")
