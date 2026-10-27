"""Order-of-magnitude fault-tolerant resource estimate for the chemically accurate ADAPT circuits.

Revised after expert-panel review. Assumptions (all stated, deliberately simple):
  * On error-corrected (logical) qubits we use NO zero-noise extrapolation; only the noise-induced part of
    the error must fit in the budget left after the ansatz's own (noise-free) error:
        budget = 1.6 mHa - |E_circuit(noise-free) - E_exact|
  * Noise-induced error is linear in the gate error at small error (checked on the sweep):
        slope = (|E_REM(r2_min) - E_exact| - ansatz_err) / r2_min   with r1 = r2/12 as in the sweep
    -> required logical two-qubit error  r2_L = budget / slope
  * Surface code: per-round, per-patch logical error p_L(d) = 0.1 (p/p_th)^((d+1)/2), p = 1e-3, p_th = 1e-2.
    A lattice-surgery CNOT takes ~d rounds on ~3 patches -> per-gate logical error  3 d p_L(d).
  * Footprint: 2 d^2 physical qubits per logical patch x 2 for routing space. Magic-state (T) factories are
    EXCLUDED and usually dominate the footprint. Circuit CNOT counts are upper bounds (naive ladders).
  * T gates: each Rz synthesised to eps_rot = 1e-4 rad with ~3 log2(1/eps) T gates (Ross-Selinger).
  * Even fault tolerant, VQE still needs ~1e5-1e6 shots per Pauli term; the standard fault-tolerant route
    for chemistry is quantum phase estimation (qubitization), not estimated here.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import circuit_for, first_chem_acc, prepare, problems_along, run_adapt  # noqa: E402

P_PHYS, P_TH, EPS_ROT = 1e-3, 1e-2, 1e-4
noise = json.loads((ROOT / "results" / "noise.json").read_text())
probs_at = problems_along([1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10])
out = {"assumptions": __doc__.strip().splitlines()[2:]}
for r in ("1.098", "2.1"):
    pr = prepare(probs_at[float(r)]["cas66"])
    res = run_adapt(pr)
    k = first_chem_acc(res, pr.e_exact, thr=0.8e-3)  # same circuit as the noise study
    gates = circuit_for(pr, res, k)
    n_cx = sum(g[0] == "cx" for g in gates)
    n_rz = sum(1 for g in gates if g[0] == "1q" and abs(g[2][0, 1]) < 1e-12 and abs(g[2][1, 0]) < 1e-12)
    ansatz = abs(noise[r]["noiseless_err_mHa"])
    budget = 1.6 - ansatz
    sw = sorted(noise[r]["sweep"], key=lambda s: s["r2"])
    s0 = sw[0]
    slope = (abs(s0["rem"]) - ansatz) / s0["r2"]  # mHa per unit two-qubit error (REM only, no ZNE)
    r2_target = budget / slope
    d = next(d for d in range(3, 61, 2) if 3 * d * 0.1 * (P_PHYS / P_TH) ** ((d + 1) / 2) <= r2_target)
    n_log = pr.tp.n_tapered
    t_per_rz = math.ceil(3 * math.log2(1 / EPS_ROT))
    out[r] = {"logical_qubits": n_log, "cx": n_cx, "rz": n_rz, "ansatz_err_mHa": ansatz, "budget_mHa": budget,
              "noise_slope_mHa_per_unit_r2": slope, "r2_logical_target": r2_target, "code_distance": d,
              "physical_qubits_data_and_routing": n_log * 2 * d * d * 2, "t_gates": n_rz * t_per_rz,
              "note": "factories excluded; CNOT counts are upper bounds; VQE shot cost not included"}
    print(f"R={r}: {n_log} logical qubits, {n_cx} CX, {n_rz} Rz; logical 2q error target ~{r2_target:.1e} "
          f"-> distance {d}, ~{n_log * 2 * d * d * 2:,} physical qubits (data + routing, no factories), ~{n_rz * t_per_rz:,} T gates")
(ROOT / "results" / "ft_estimate.json").write_text(json.dumps(out, indent=1))
