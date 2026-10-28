# Production port. Written against Qiskit SDK 2.6.1 (https://quantum.cloud.ibm.com/docs/api/qiskit)
# and qiskit-ibm-runtime 0.51.0 fake_provider (https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/fake-provider)
# docs on 2026-10-27. Not executed in the authoring environment (no package access); run
# port/test_port.py on a laptop first.
"""Qiskit circuits from our ADAPT-VQE result + transpilation to fake IBM Heron backends.

Sign convention (checked): ``PauliEvolutionGate(op, time=t)`` implements exp(-i t op)
(qiskit/circuit/library/pauli_evolution.py: ``expm(-1j * time * spmatrix)``). Our compiled
rotations ``(x, z, angle)`` mean exp(-i angle/2 P(x,z)) (src/noise.py compile_rotations), so
we use ``time = angle / 2``. All Pauli terms of one ADAPT generator commute, so grouping them
into one PauliEvolutionGate (``group='operator'``) is exact as well.

Ordering: ``QuantumCircuit`` qubit q <-> our bit q, and the HF reference integer ``ref_bits``
sets X on every qubit whose bit is 1 (Statevector index convention is the same: bit q = qubit q).

Fake backends checked in qiskit-ibm-runtime 0.51.0 fake_provider (conf_*.json): all have basis
{cz, id, rz, sx, x}. Heron r1: FakeTorino (133 q); Heron r2: FakeFez, FakeMarrakesh, FakeKingston
(156 q); Heron r3: FakeAachen, FakeBoston (156 q); Nighthawk (square lattice): FakeMiami (120 q).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.noise import compile_rotations  # noqa: E402
from src.pauli import label  # noqa: E402

TWO_Q_NAMES = ("cz", "ecr", "cx", "rzz", "iswap")


def adapt_rotations(pr, res, k: int, tapered: bool = True) -> list[tuple[int, int, float]]:
    """(x, z, angle) list for the first k ADAPT operators at their iteration-k optimum."""
    if k == 0:
        return []
    pool = pr.pool_t if tapered else pr.pool_full
    n = pr.tp.n_tapered if tapered else 2 * pr.P.n_orb
    return compile_rotations([pool[i] for i in res.ops[:k]], res.theta_history[k - 1], n)


def adapt_operator_groups(pr, res, k: int, tapered: bool = True) -> list[list[tuple[int, int, float]]]:
    """Same rotations, grouped per ADAPT generator (each group = mutually commuting terms)."""
    if k == 0:
        return []
    pool = pr.pool_t if tapered else pr.pool_full
    n = pr.tp.n_tapered if tapered else 2 * pr.P.n_orb
    return [compile_rotations([pool[i]], [th], n) for i, th in zip(res.ops[:k], res.theta_history[k - 1])]


def build_circuit(rots_or_groups, n: int, ref_bits: int, group: str = "term", measure: bool = False):
    """HF reference (X gates) + Pauli rotations as PauliEvolutionGates.

    ``group='term'``: one gate per Pauli term, PauliEvolutionGate(SparsePauliOp(label), time=angle/2).
    ``group='operator'``: input is a list of groups; one gate per ADAPT operator,
    PauliEvolutionGate(SparsePauliOp(labels, angle_j/2), time=1.0).
    https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(n)
    for q in range(n):
        if (ref_bits >> q) & 1:
            qc.x(q)
    if group == "term":
        for x, z, ang in rots_or_groups:
            qc.append(PauliEvolutionGate(SparsePauliOp(label(x, z, n)), time=ang / 2.0), range(n))
    else:
        for grp in rots_or_groups:
            if not grp:
                continue
            op = SparsePauliOp.from_list([(label(x, z, n), ang / 2.0) for x, z, ang in grp], num_qubits=n)
            qc.append(PauliEvolutionGate(op, time=1.0), range(n))
    if measure:
        qc.measure_all()  # creates classical register "meas" -> pub_result.data.meas
    return qc


def circuit_from_adapt(pr, res, k: int, tapered: bool = True, group: str = "operator", measure: bool = False):
    """Qiskit circuit for the first k ADAPT operators (tapered 7-qubit or untapered 12-qubit)."""
    from src.pauli import hf_bitstring

    n = pr.tp.n_tapered if tapered else 2 * pr.P.n_orb
    ref = pr.tp.ref_bits if tapered else hf_bitstring(pr.P.n_orb, pr.P.n_alpha, pr.P.n_beta)
    rots = adapt_operator_groups(pr, res, k, tapered) if group == "operator" else adapt_rotations(pr, res, k, tapered)
    return build_circuit(rots, n, ref, group=group, measure=measure)


def fake_backend(name: str = "FakeFez"):
    """Instantiate a fake IBM backend by class name from qiskit_ibm_runtime.fake_provider."""
    from qiskit_ibm_runtime import fake_provider

    try:
        return getattr(fake_provider, name)()
    except AttributeError as exc:
        avail = sorted(a for a in dir(fake_provider) if a.startswith("Fake"))
        raise ValueError(f"{name} not in qiskit_ibm_runtime.fake_provider; available: {avail}") from exc


def transpile_isa(qc, backend, optimization_level: int = 3, seed: int = 7):
    """ISA circuit via generate_preset_pass_manager (Qiskit 2.x import path qiskit.transpiler).
    https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler#generate_preset_pass_manager"""
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed)
    return pm.run(qc)


def gate_report(isa) -> dict:
    """Two-qubit gate count (cz/ecr/cx/...), total ops, depth and two-qubit depth."""
    ops = {k: int(v) for k, v in isa.count_ops().items()}
    n2 = sum(v for k, v in ops.items() if k in TWO_Q_NAMES)
    return {
        "two_qubit": n2,
        "ops": ops,
        "depth": int(isa.depth()),
        "depth_2q": int(isa.depth(lambda ci: ci.operation.num_qubits == 2)),
        "num_qubits_used": len({isa.find_bit(q).index for ci in isa.data for q in ci.qubits}),
    }


def transpile_counts(qc, backend_name: str = "FakeFez", optimization_level: int = 3, seed: int = 7) -> dict:
    """Transpile to a fake backend and report hardware gate counts."""
    be = fake_backend(backend_name)
    isa = transpile_isa(qc, be, optimization_level, seed)
    rep = gate_report(isa)
    rep.update(backend=backend_name, optimization_level=optimization_level, logical_gates=len(qc.data))
    return rep


def statevector_energy(qc, H_spo) -> float:
    """<psi|H|psi> with qiskit.quantum_info.Statevector (circuit without measurements)."""
    from qiskit.quantum_info import Statevector

    return float(Statevector(qc).expectation_value(H_spo).real)
