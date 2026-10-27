"""Classical baseline along the N2 bond: HF, CCSD, CCSD(T), exact (FCI) per active space."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ccsd import ccsd_t  # noqa: E402
from src.n2 import n2_problems  # noqa: E402
from src.pauli import exact_ground, qubit_hamiltonian, sector_indices, to_sparse  # noqa: E402

GEOMS = [0.90, 1.00, 1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70, 3.00]
rows, C = [], None
for r in GEOMS:
    t = time.time()
    probs, C = n2_problems(r, C)
    row = {"r": r, "e_rhf": probs["cas66"].e_rhf_full}
    for key, P in probs.items():
        nq = 2 * P.n_orb
        Hs = to_sparse(qubit_hamiltonian(P.h, P.g, P.ecore), nq)
        e_ex, _ = exact_ground(Hs, sector_indices(nq, P.n_alpha, P.n_beta, P.n_orb))
        cc = ccsd_t(P.h, P.g, P.ecore, P.n_alpha, P.n_beta)
        row[key] = {"exact": e_ex, "hf": cc["hf"], "ccsd": cc["ccsd"], "ccsd_t": cc["ccsd_t"],
                    "cc_converged": cc["converged"]}
    rows.append(row)
    c, f = row["cas66"], row["fv108"]
    print(f"R={r:.3f}  CAS66 exact {c['exact']:.6f}  CCSD(T)-exact {1e3*(c['ccsd_t']-c['exact']):+8.2f} mHa | "
          f"FV108 exact {f['exact']:.6f}  CCSD(T)-exact {1e3*(f['ccsd_t']-f['exact']):+8.2f} mHa  ({time.time()-t:.1f}s)",
          flush=True)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "baseline.json").write_text(json.dumps(rows, indent=1))
print("saved results/baseline.json")
