"""Validate qubit Hamiltonian (JW), FCI and CCSD(T) on H2O/STO-3G (Crawford projects 5/6)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ccsd import ccsd_t  # noqa: E402
from src.integrals import molecule_integrals_bohr  # noqa: E402
from src.pauli import exact_ground, hf_bitstring, qubit_hamiltonian, sector_indices, to_sparse  # noqa: E402
from src.scf import mo_integrals, rhf  # noqa: E402

ok = True
ints = molecule_integrals_bohr([
    ("O", [0.000000000000, -0.143225816552, 0.0]),
    ("H", [1.638036840407, 1.136548822547, 0.0]),
    ("H", [-1.638036840407, 1.136548822547, 0.0]),
])
scf = rhf(ints)
h, g = mo_integrals(ints, scf["C"])
cc = ccsd_t(h, g, ints["enuc"], 5, 5)
for name, got, ref in [("CCSD corr", cc["ccsd"] - scf["e"], -0.070680088376),
                       ("(T) corr", cc["ccsd_t"] - cc["ccsd"], -0.000099877272)]:
    good = abs(got - ref) < 1e-8
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} H2O {name}: {got:.12f} ref {ref:.12f}")

t = time.time()
H = qubit_hamiltonian(h, g, ints["enuc"])
Hs = to_sparse(H, 14)
hf = hf_bitstring(7, 5, 5)
e_hf_q = Hs[hf, hf].real
good = abs(e_hf_q - scf["e"]) < 1e-9
ok &= good
print(f"{'PASS' if good else 'FAIL'} <HF|H_qubit|HF> = {e_hf_q:.10f} vs RHF {scf['e']:.10f}  ({len(H)} Pauli terms, {time.time()-t:.1f}s)")
e_fci, _ = exact_ground(Hs, sector_indices(14, 5, 5, 7))
good = (e_fci <= cc["ccsd_t"] + 1e-9) and abs(e_fci - cc["ccsd_t"]) < 5e-4
ok &= good
print(f"{'PASS' if good else 'FAIL'} FCI {e_fci:.10f} vs CCSD(T) {cc['ccsd_t']:.10f} (diff {1e3*(cc['ccsd_t']-e_fci):.4f} mHa)")
sys.exit(0 if ok else 1)
