"""Full PES: classical baselines + tapered ADAPT-VQE + circuit resources (symmetric RHF branch)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ccsd import ccsd_t  # noqa: E402
from src.noise import count_gates  # noqa: E402
from src.pipeline import circuit_for, first_chem_acc, prepare, problems_along, run_adapt, untapered_cx  # noqa: E402

GEOMS = [0.90, 1.00, 1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70, 3.00]
FV_ADAPT_AT = set(GEOMS)
probs_at = problems_along(GEOMS)
rows = []
for r in GEOMS:
    t = time.time()
    row = {"r": r, "e_rhf": probs_at[r]["cas66"].e_rhf_full}
    for key, P in probs_at[r].items():
        pr = prepare(P)
        cc = ccsd_t(P.h, P.g, P.ecore, P.n_alpha, P.n_beta)
        d = {"exact": pr.e_exact, "exact_lowest_any_spin": pr.e_lowest_any, "s2_lowest_any_spin": pr.s2_lowest_any,
             "hf": cc["hf"], "ccsd": cc["ccsd"], "ccsd_t": cc["ccsd_t"], "e_t": cc.get("e_t"),
             "cc_converged": cc["converged"], "qubits": 2 * P.n_orb, "qubits_tapered": pr.tp.n_tapered,
             "pauli_terms": len(pr.H), "pauli_terms_tapered": len(pr.Ht), "pool": len(pr.pool_t)}
        if key == "cas66" or r in FV_ADAPT_AT:
            res = run_adapt(pr, max_ops=80 if key == "cas66" else 120)
            k = first_chem_acc(res, pr.e_exact)
            d["adapt"] = {"energies": res.energies, "labels": res.labels, "n_ops": len(res.ops),
                          "final": res.energies[-1], "k_chem_acc": k, "s2_final": res.s2[-1]}
            if k:
                g = count_gates(circuit_for(pr, res, k))
                d["adapt"]["circuit_at_chem_acc"] = {**g, "cx_untapered": untapered_cx(pr, res, k)}
        row[key] = d
    rows.append(row)
    c = row["cas66"]
    ca = c["adapt"]
    cx = ca.get("circuit_at_chem_acc", {}).get("cx")
    print(f"R={r:.3f} CAS66 {c['qubits']}->{c['qubits_tapered']}q | CCSD(T) err {1e3*(c['ccsd_t']-c['exact']):+8.2f} mHa | "
          f"ADAPT err {1e3*(ca['final']-c['exact']):.5f} mHa, chem.acc @ {ca['k_chem_acc']} ops ({cx} CX) | "
          f"FV {row['fv108']['qubits']}->{row['fv108']['qubits_tapered']}q "
          f"{'ADAPT err %.4f mHa @%s ops' % (1e3*(row['fv108']['adapt']['final']-row['fv108']['exact']), row['fv108']['adapt']['k_chem_acc']) if 'adapt' in row['fv108'] else ''}"
          f" ({time.time()-t:.0f}s)", flush=True)
(ROOT / "results" / "pes.json").write_text(json.dumps(rows, indent=1))
print("saved results/pes.json")
