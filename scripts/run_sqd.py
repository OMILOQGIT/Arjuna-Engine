"""SQD probe on FV(10e,8o) at R = 2.10 A (CCSD(T) does not converge there).

Compares, at the default noise model, the noisy VQE energy estimate (with REM) of a
shallow ADAPT circuit against SQD energies from samples of the *same noisy circuit*.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.noise import DensitySim, Estimator, NoiseModel, count_gates  # noqa: E402
from src.pauli import sector_indices, to_sparse  # noqa: E402
from src.pipeline import circuit_for, prepare, problems_along, run_adapt  # noqa: E402
from src.sqd import sample, sqd_energy, untaper_bits  # noqa: E402

R = 2.10
KS = [4, 8]                     # shallow circuits: first k ADAPT operators
SHOTS = [100, 300, 1000, 3000, 10000]
probs_at = problems_along([1.20, 1.35, 1.50, 1.70, 1.90, R])
P = probs_at[R]["fv108"]
pr = prepare(P)
t = time.time()
res = run_adapt(pr, max_ops=max(KS))
print(f"FV108 R={R}: {2*P.n_orb}->{pr.tp.n_tapered} qubits, sector dim "
      f"{len(sector_indices(2*P.n_orb, P.n_alpha, P.n_beta, P.n_orb))}, exact {pr.e_exact:.6f} ({time.time()-t:.0f}s)",
      flush=True)
Hs_full = to_sparse(pr.H, 2 * P.n_orb).tocsr()
n = pr.tp.n_tapered
sim, est, nm = DensitySim(n), Estimator(pr.Ht, n), NoiseModel()
rng = np.random.default_rng(7)
out = {"r": R, "exact": pr.e_exact, "qubits_tapered": n, "noise": vars(nm), "runs": []}
for k in KS:
    gates = circuit_for(pr, res, k)
    t = time.time()
    rho = sim.run(gates, pr.tp.ref_bits, nm)
    e_vqe_noisy = est.energy(rho, p_ro=nm.p_ro, rem=True)
    e_vqe_ideal = res.energies[k]
    print(f" k={k} ops, {count_gates(gates)}: ideal VQE err {1e3*(e_vqe_ideal-pr.e_exact):.2f} mHa, "
          f"noisy VQE(+REM) err {1e3*(e_vqe_noisy-pr.e_exact):.2f} mHa  ({time.time()-t:.0f}s sim)", flush=True)
    for shots in SHOTS:
        tb = sample(rho, shots, n, nm.p_ro, rng)
        fb = untaper_bits(pr.tp, tb)
        e, dim, fg = sqd_energy(Hs_full, fb, P.n_orb, P.n_alpha, P.n_beta)
        print(f"   SQD shots={shots:6d}: subspace {dim:5d} dets, good-N frac {fg:.2f}, err {1e3*(e-pr.e_exact):8.3f} mHa",
              flush=True)
        out["runs"].append({"k": k, "cx": count_gates(gates)["cx"], "shots": shots, "dim": dim, "frac_good": fg,
                            "err_mHa": 1e3 * (e - pr.e_exact), "vqe_noisy_err_mHa": 1e3 * (e_vqe_noisy - pr.e_exact),
                            "vqe_ideal_err_mHa": 1e3 * (e_vqe_ideal - pr.e_exact)})
(ROOT / "results" / "sqd.json").write_text(json.dumps(out, indent=1))
print("saved results/sqd.json")
