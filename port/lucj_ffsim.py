# Production port. Written against ffsim 0.0.85 (source tag v0.0.85;
# https://qiskit-community.github.io/ffsim/) and PySCF 2.14.0 docs on 2026-10-27. Not executed in
# the authoring environment (no package access); run port/test_port.py on a laptop first.
"""LUCJ ansatz circuit with ffsim (as in IBM's SQD paper, Robledo-Moreno et al., arXiv:2405.05068).

Recipe (IBM SQD tutorial https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization
and ffsim API https://qiskit-community.github.io/ffsim/api/ffsim.html):
  1. CCSD on the active-space model Hamiltonian -> t1 (nocc, nvir), t2 (nocc, nocc, nvir, nvir).
  2. ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=..., interaction_pairs=(pairs_aa, pairs_ab))
  3. circuit: ffsim.qiskit.PrepareHartreeFockJW(norb, nelec) + ffsim.qiskit.UCJOpSpinBalancedJW(op),
     on 2*norb qubits with alpha orbitals on qubits 0..norb-1 (= our block ordering), measure_all.
  4. transpile with ffsim.qiskit.generate_lucj_pass_manager(backend, norb, connectivity,
     interaction_pairs, optimization_level=3) -> (pass_manager, pairs_ab actually realisable); build
     the operator with the RETURNED pairs_ab. Fallback: generate_preset_pass_manager with
     ``pm.pre_init = ffsim.qiskit.PRE_INIT``.

BASIS CAVEAT: the CCSD amplitudes live in the basis of the model-space RHF orbitals, which for
CASSCF active orbitals is a rotation of our Problem basis. We therefore return the Problem rotated
into that basis (chem_pyscf.rotate_problem) and SQD must use that rotated Problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def ccsd_amplitudes(P):
    """(t1, t2, P_mo) from CCSD on the active-space model Hamiltonian (PySCF RCCSD)."""
    from pyscf import cc

    from chem_pyscf import model_mf, rotate_problem

    mf = model_mf(P)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.max_cycle = 500
    mycc.kernel()
    if not mycc.converged:
        print("[lucj] warning: CCSD not converged; amplitudes used as-is for the ansatz guess")
    return mycc.t1, mycc.t2, rotate_problem(P, mf.mo_coeff)


def default_pairs(norb: int, connectivity: str = "heavy-hex"):
    """alpha-alpha nearest neighbours; alpha-beta every 4th orbital on heavy-hex (paper choice),
    every orbital on square lattices."""
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    step = 4 if connectivity == "heavy-hex" else 1
    pairs_ab = [(p, p) for p in range(0, norb, step)]
    return pairs_aa, pairs_ab


def lucj_circuit(P, n_reps: int = 1, backend=None, connectivity: str = "heavy-hex", optimize: bool = False):
    """Returns (logical_circuit_with_measure, isa_circuit_or_None, P_mo, pairs).

    ``optimize=True`` uses ffsim's compressed double factorization (needs jax; slower)."""
    try:
        import ffsim
    except ImportError:
        print("[lucj] ffsim not installed -> skipped (pip install 'ffsim[qiskit]')")
        return None
    from qiskit import QuantumCircuit, QuantumRegister

    t1, t2, P_mo = ccsd_amplitudes(P)
    norb, nelec = P.n_orb, (P.n_alpha, P.n_beta)
    pairs_aa, pairs_ab = default_pairs(norb, connectivity)
    pm = None
    if backend is not None:
        try:
            pm, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
                backend=backend, norb=norb, connectivity=connectivity,
                interaction_pairs=(pairs_aa, pairs_ab), optimization_level=3)
        except Exception as exc:  # noqa: BLE001 - older ffsim: no generate_lucj_pass_manager
            print(f"[lucj] generate_lucj_pass_manager unavailable ({exc!r}); using preset PM + PRE_INIT")
            from qiskit.transpiler import generate_preset_pass_manager

            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
            pm.pre_init = ffsim.qiskit.PRE_INIT
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=n_reps,
                                                   interaction_pairs=(pairs_aa, pairs_ab), optimize=optimize)
    qubits = QuantumRegister(2 * norb, name="q")
    qc = QuantumCircuit(qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), qubits)
    qc.measure_all()
    isa = pm.run(qc) if pm is not None else None
    return qc, isa, P_mo, (pairs_aa, pairs_ab)


def lucj_statevector_energy(P_mo, qc) -> float | None:
    """Noiseless LUCJ energy (sanity check) using our JW Hamiltonian in the model-HF basis."""
    try:
        from qiskit.quantum_info import Statevector

        from qubit_nature import our_to_sparse_pauli
        from src.pauli import qubit_hamiltonian
    except ImportError:
        return None
    H = our_to_sparse_pauli(qubit_hamiltonian(P_mo.h, P_mo.g, P_mo.ecore), 2 * P_mo.n_orb)
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return float(np.real(sv.expectation_value(H)))
