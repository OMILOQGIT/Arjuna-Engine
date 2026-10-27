"""Validate integrals + RHF against published STO-3G energies."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.integrals import molecule_integrals, molecule_integrals_bohr  # noqa: E402
from src.scf import rhf, rhf_robust  # noqa: E402


def check(name, got, ref, tol=1e-6):
    ok = abs(got - ref) < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got:.9f} ref {ref:.9f} diff {got - ref:+.2e}")
    return ok


results = []
# H2, R = 1.4 bohr: Szabo & Ostlund RHF/STO-3G = -1.1167 (4 dp)
ints = molecule_integrals_bohr([("H", [0, 0, 0]), ("H", [0, 0, 1.4])])
results.append(check("H2 RHF/STO-3G (Szabo-Ostlund)", rhf(ints)["e"], -1.1167, tol=1e-4))

# H2O, Crawford programming project #3 geometry (bohr): RHF/STO-3G = -74.942079928192
t = time.time()
ints = molecule_integrals_bohr([
    ("O", [0.000000000000, -0.143225816552, 0.0]),
    ("H", [1.638036840407, 1.136548822547, 0.0]),
    ("H", [-1.638036840407, 1.136548822547, 0.0]),
])
results.append(check("H2O enuc (Crawford)", ints["enuc"], 8.002367061810450, tol=1e-8))
results.append(check("H2O RHF/STO-3G (Crawford)", rhf(ints)["e"], -74.942079928192, tol=1e-6))
print(f"H2O integrals+SCF {time.time() - t:.1f}s")

t = time.time()
ints = molecule_integrals([("N", [0, 0, 0]), ("N", [0, 0, 1.098])])
res = rhf_robust(ints)
# CCCBDB / literature RHF/STO-3G for N2 near equilibrium: -107.4959
results.append(check("N2 R=1.098A RHF/STO-3G (lit. -107.4959)", res["e"], -107.4959, tol=2e-4))
print(f"N2 robust SCF {time.time() - t:.1f}s")
sys.exit(0 if all(results) else 1)
