# Production port. Written against qiskit-nature 0.8.0 (stable/0.8 source,
# https://qiskit-community.github.io/qiskit-nature/) and Qiskit SDK 2.6.1
# (https://quantum.cloud.ibm.com/docs/api/qiskit) docs on 2026-10-27. Not executed in the
# authoring environment (no package access); run port/test_port.py on a laptop first.
"""Qubit Hamiltonians with qiskit-nature + converters between our Pauli dicts and SparsePauliOp.

ORDERING CONVENTIONS (all three agree, so no permutation is ever needed):
  * Spin orbitals: block ordering, alpha 0..n-1 then beta n..2n-1. Ours (src/pauli.py) and
    qiskit-nature's FermionicOp register (ElectronicIntegrals alpha block first, beta block
    second) are identical.
  * Qubits: spin orbital p <-> qubit p, Jordan-Wigner a_p = Z_0..Z_{p-1} (X_p + iY_p)/2 in both
    (qiskit_nature/second_q/mappers/jordan_wigner_mapper.py ``pauli_table``).
  * Pauli labels: Qiskit strings are little-endian, qubit 0 is the RIGHTMOST character;
    ``src.pauli.label`` already produces this.
  * Integers / bitstrings: bit q of a basis-state index = qubit q (Statevector, BitArray).

PAULI PHASE CONVENTION: our term (x, z) means P(x,z) = i^{popcount(x&z)} X^x Z^z. On a qubit with
x=z=1 that is i*X*Z = Y (a proper, Hermitian Pauli Y). Hence P(x,z) is *exactly* the tensor
product of proper Paulis named by ``label(x, z, n)`` and the coefficient carries over unchanged.
Qiskit's SparsePauliOp keeps its Paulis phase-free (group phase absorbed into ``coeffs``,
see qiskit/quantum_info/operators/symplectic/sparse_pauli_op.py ``__init__``); we still apply
``(-1j)**phase`` on the way back, defensively.

The pure-Python helpers (``our_to_label_list`` / ``label_list_to_our``) need no Qiskit and are
unit-tested in the authoring sandbox; the Qiskit wrappers import lazily.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pauli import label  # noqa: E402

_CHAR_XZ = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}


# ----------------------------------------------------------------------------- pure converters
def our_to_label_list(op: dict, n: int, tol: float = 0.0) -> list[tuple[str, complex]]:
    """{(x, z): c} -> [(qiskit_label, c)] with qubit 0 rightmost; Y needs no extra phase."""
    return [(label(x, z, n), complex(c)) for (x, z), c in op.items() if abs(c) > tol]


def label_list_to_our(items, tol: float = 0.0) -> dict:
    """[(qiskit_label, c)] -> {(x, z): c}. Labels must be phase-free (no leading '-', 'i')."""
    out: dict = {}
    for lab, c in items:
        n = len(lab)
        x = z = 0
        for pos, ch in enumerate(lab):
            q = n - 1 - pos  # rightmost character = qubit 0
            xb, zb = _CHAR_XZ[ch]
            x |= xb << q
            z |= zb << q
        out[(x, z)] = out.get((x, z), 0) + complex(c)
    return {k: v for k, v in out.items() if abs(v) > tol}


def label_matrix(lab: str) -> np.ndarray:
    """Dense matrix of a Qiskit-ordered Pauli label (numpy only; for small-n checks)."""
    mats = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1.0, -1.0])}
    M = np.array([[1.0 + 0j]])
    for ch in lab:  # leftmost char = highest qubit = most significant kron factor
        M = np.kron(M, mats[ch])
    return M


# ----------------------------------------------------------------------------- Qiskit wrappers
def our_to_sparse_pauli(op: dict, n: int):
    """Our Pauli dict -> qiskit.quantum_info.SparsePauliOp (SparsePauliOp.from_list).
    https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp#from_list"""
    from qiskit.quantum_info import SparsePauliOp

    items = our_to_label_list(op, n)
    if not items:
        return SparsePauliOp("I" * n, coeffs=[0.0])
    return SparsePauliOp.from_list(items, num_qubits=n)


def sparse_pauli_to_our(spo, tol: float = 0.0) -> dict:
    """SparsePauliOp -> our dict, reading the symplectic arrays (paulis.x/.z: [term, qubit])."""
    xs = np.asarray(spo.paulis.x, dtype=bool)
    zs = np.asarray(spo.paulis.z, dtype=bool)
    ph = np.asarray(spo.paulis.phase)  # 0 for SparsePauliOp; defensive
    w = 1 << np.arange(spo.num_qubits, dtype=object)
    out: dict = {}
    for xr, zr, p, c in zip(xs, zs, ph, spo.coeffs):
        x, z = int(np.dot(xr.astype(object), w)), int(np.dot(zr.astype(object), w))
        out[(x, z)] = out.get((x, z), 0) + complex(c) * (-1j) ** int(p)
    return {k: v for k, v in out.items() if abs(v) > tol}


# ----------------------------------------------------------------------------- qiskit-nature
def nature_problem(P):
    """ElectronicStructureProblem from our Problem (h = h1eff, g = chemist (pq|rs), ecore).

    * ElectronicEnergy.from_raw_integrals(h1_a, h2_aa, ..., auto_index_order=True) detects the
      chemists' order of ``g`` and converts it to physicists' order internally
      (qiskit_nature/second_q/operators/electronic_integrals.py).
    * ``nuclear_repulsion_energy`` is stored in ``hamiltonian.constants``, which is NOT mapped to
      the qubit operator (electronic_energy.py docstring) -> we add ecore as an identity term.
    * ``num_particles`` must be set: get_tapered_mapper uses it to pick the HF sector.
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.hamiltonians.ElectronicEnergy.html
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.problems.ElectronicStructureProblem.html
    """
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

    ham = ElectronicEnergy.from_raw_integrals(np.asarray(P.h), np.asarray(P.g), auto_index_order=True)
    ham.nuclear_repulsion_energy = float(P.ecore)
    prob = ElectronicStructureProblem(ham)
    prob.num_spatial_orbitals = P.n_orb
    prob.num_particles = (P.n_alpha, P.n_beta)
    return prob


def nature_qubit_hamiltonian(P, taper: bool = False):
    """Jordan-Wigner (optionally Z2-tapered) qubit Hamiltonian incl. the constant ecore.

    Returns (SparsePauliOp, mapper). Tapering: ``problem.get_tapered_mapper(JordanWignerMapper())``
    finds Z2 symmetries with qiskit.quantum_info.Z2Symmetries.find_z2_symmetries and picks the
    sector of the Hartree-Fock bitstring (base_problem.py / electronic_structure_problem.py).
    https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.mappers.TaperedQubitMapper.html
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    prob = nature_problem(P)
    fop = prob.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    if taper:
        mapper = prob.get_tapered_mapper(mapper)
    qop = mapper.map(fop)  # SparsePauliOp (qiskit-nature >=0.7 has no PauliSumOp)
    qop = (qop + SparsePauliOp("I" * qop.num_qubits, coeffs=[P.ecore])).simplify(atol=1e-12)
    return qop, mapper


def ground_energy_in_sector(spo, n_orb: int, n_alpha: int, n_beta: int) -> float:
    """Exact ground energy of an UNtapered JW operator restricted to (n_alpha, n_beta)."""
    from src.pauli import sector_indices

    M = spo.to_matrix(sparse=True)
    idx = sector_indices(spo.num_qubits, n_alpha, n_beta, n_orb)
    sub = M[idx][:, idx].toarray()
    return float(np.linalg.eigvalsh(sub)[0])


def spin_penalized_sparse_pauli(P, mu: float = 0.5):
    """H + mu*S^2 (untapered JW) as SparsePauliOp, mirroring src.pipeline.prepare(P, mu): the
    penalty pins ADAPT to the singlet at long bonds. Uses src.pauli.s_squared (same block JW order).
    For the tapered version use ``our_to_sparse_pauli(pr.Hpen_t, pr.tp.n_tapered)``."""
    from src.pauli import op_add, op_scale, qubit_hamiltonian, s_squared

    H = qubit_hamiltonian(P.h, P.g, P.ecore)
    return our_to_sparse_pauli(op_add(H, op_scale(s_squared(P.n_orb), mu)), 2 * P.n_orb)
