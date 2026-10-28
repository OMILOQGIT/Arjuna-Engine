# Production port. Written against PySCF 2.14.0, Qiskit SDK 2.6.1, qiskit-nature 0.8.0,
# qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, qiskit-addon-sqd 0.14.0 docs on 2026-10-27.
# Not executed in the authoring environment (no package access); run this file on a laptop first:
#     python -m pytest port/test_port.py -v
"""Port acceptance tests. Each library-dependent test uses pytest.importorskip, so the file runs
(and the pure-NumPy tests pass) on any machine; install requirements-laptop.txt to run them all."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.n2 import n2_problems  # noqa: E402
from src.pauli import sector_indices, to_sparse  # noqa: E402
from src.pipeline import prepare, run_adapt  # noqa: E402

E_RHF_EQ = -107.495975   # N2/STO-3G RHF, R = 1.098 A, symmetric branch (RESEARCH.md)
E_CAS66_EQ = -107.622015  # CAS(6e,6o) exact at 1.098 A
E_FV108_21 = -107.448604  # FV(10e,8o) exact at 2.1 A


@pytest.fixture(scope="module")
def mock_cas66():
    """Verified NumPy prototype: CAS(6e,6o) at 1.098 A, prepared + short ADAPT run."""
    P = n2_problems(1.098)[0]["cas66"]
    pr = prepare(P)
    res = run_adapt(pr, max_ops=6)
    return P, pr, res


# ----------------------------------------------------------------------------- no external deps
def test_pure_label_converter_matches_our_matrices(mock_cas66):
    from qubit_nature import label_list_to_our, label_matrix, our_to_label_list

    _, pr, _ = mock_cas66
    n = pr.tp.n_tapered
    items = our_to_label_list(pr.Ht, n)
    M = sum(c * label_matrix(lab) for lab, c in items)
    assert np.abs(M - to_sparse(pr.Ht, n).toarray()).max() < 1e-12
    back = label_list_to_our(items)
    assert set(back) == set(pr.Ht) and max(abs(back[k] - pr.Ht[k]) for k in back) < 1e-14


def test_mock_reference_numbers(mock_cas66):
    P, pr, _ = mock_cas66
    assert abs(P.e_rhf_full - E_RHF_EQ) < 1e-5
    assert abs(pr.e_exact - E_CAS66_EQ) < 1e-6


# ----------------------------------------------------------------------------- (a) PySCF
def test_a_pyscf_rhf_sto3g_eq():
    pytest.importorskip("pyscf")
    from chem_pyscf import build_mol, run_rhf

    mf = run_rhf(build_mol(1.098, "sto-3g"))
    assert mf.converged
    assert abs(mf.e_tot - E_RHF_EQ) < 1e-5, mf.e_tot


@pytest.mark.parametrize("mode", ["rhf_irrep", "rhf_w2p"])
def test_a2_pyscf_cas66_matches_prototype(mode):
    pytest.importorskip("pyscf")
    from chem_pyscf import active_space_benchmarks, n2_problem_pyscf

    P, mf, mc = n2_problem_pyscf(1.098, "sto-3g", "cas66", mode)
    assert P.meta["active_order"] == "irrep"
    assert set(P.meta["active_irreps"][:3]) == {"Ag", "B2u", "B3u"}      # occupied: 3sg, pi_u
    assert set(P.meta["active_irreps"][3:]) == {"B1u", "B2g", "B3g"}     # virtual: 3su, pi_g
    assert (P.n_orb, P.n_alpha, P.n_beta) == (6, 3, 3)
    assert abs(mc.e_tot - E_CAS66_EQ) < 1e-6, mc.e_tot
    assert abs(active_space_benchmarks(P)["exact"] - E_CAS66_EQ) < 1e-6
    pr = prepare(P)  # our pipeline consumes the PySCF Problem unchanged
    assert abs(pr.e_exact - E_CAS66_EQ) < 1e-6
    hf = to_sparse(pr.H, 12)[0b000111000111, 0b000111000111].real  # <HF|H|HF> = RHF (canonical RHF orbitals)
    assert abs(hf - E_RHF_EQ) < 1e-6


def test_a3_pyscf_fv108_stretched():
    pytest.importorskip("pyscf")
    from chem_pyscf import active_space_benchmarks, problems_along

    # step outward like src/pipeline.problems_along (density continuation + pinned D2h occupations)
    out = problems_along([1.2, 1.35, 1.5, 1.7, 1.9, 2.1], "sto-3g", "fv108")
    P = out[2.1][0]
    b = active_space_benchmarks(P)
    assert abs(b["exact"] - E_FV108_21) < 1e-5  # singlet-constrained FCI
    assert abs(b["s2"]) < 1e-6
    assert prepare(P).tp.n_tapered == 11


# ----------------------------------------------------------------------------- (b) qiskit-nature
def test_b_nature_jw_same_ground_energy_and_coeffs(mock_cas66):
    pytest.importorskip("qiskit_nature")
    from qubit_nature import ground_energy_in_sector, nature_qubit_hamiltonian, sparse_pauli_to_our

    P, pr, _ = mock_cas66
    H, _ = nature_qubit_hamiltonian(P, taper=False)
    assert H.num_qubits == 12
    assert abs(ground_energy_in_sector(H, 6, 3, 3) - pr.e_exact) < 1e-8
    theirs, ours = sparse_pauli_to_our(H), pr.H
    diff = max(abs(theirs.get(k, 0) - ours.get(k, 0)) for k in set(theirs) | set(ours))
    assert diff < 1e-8, f"same JW convention expected; max coefficient difference {diff}"


def test_b2_nature_tapered(mock_cas66):
    pytest.importorskip("qiskit_nature")
    from qubit_nature import nature_qubit_hamiltonian

    _, pr, _ = mock_cas66
    Ht, _ = nature_qubit_hamiltonian(pr.P, taper=True)
    assert Ht.num_qubits == 7
    spec = np.linalg.eigvalsh(Ht.to_matrix())
    assert np.min(np.abs(spec - pr.e_exact)) < 1e-8


# ----------------------------------------------------------------------------- (c) converters
def test_c_sparse_pauli_roundtrip_and_ground_energy(mock_cas66):
    pytest.importorskip("qiskit")
    from qubit_nature import our_to_sparse_pauli, sparse_pauli_to_our

    _, pr, _ = mock_cas66
    n = pr.tp.n_tapered
    spo = our_to_sparse_pauli(pr.Ht, n)
    assert np.abs(spo.to_matrix() - to_sparse(pr.Ht, n).toarray()).max() < 1e-12
    back = sparse_pauli_to_our(spo)
    assert max(abs(back.get(k, 0) - pr.Ht.get(k, 0)) for k in set(back) | set(pr.Ht)) < 1e-12
    w = np.linalg.eigvalsh(spo.to_matrix())
    assert np.min(np.abs(w - pr.e_exact)) < 1e-8
    # explicit Y check: our (x=1, z=1) on one qubit is the Hermitian Pauli Y
    y = our_to_sparse_pauli({(1, 1): 1.0}, 1).to_matrix()
    assert np.allclose(y, [[0, -1j], [1j, 0]])


# ----------------------------------------------------------------------------- (d) circuits
@pytest.mark.parametrize("group", ["term", "operator"])
def test_d_circuit_statevector_equals_adapt(mock_cas66, group):
    pytest.importorskip("qiskit")
    from circuits_qiskit import circuit_from_adapt, statevector_energy
    from qubit_nature import our_to_sparse_pauli

    _, pr, res = mock_cas66
    H = our_to_sparse_pauli(pr.Ht, pr.tp.n_tapered)
    for k in (1, 3, len(res.ops)):
        qc = circuit_from_adapt(pr, res, k, tapered=True, group=group)
        assert abs(statevector_energy(qc, H) - res.energies[k]) < 1e-6


def test_d2_untapered_circuit_equals_adapt(mock_cas66):
    pytest.importorskip("qiskit")
    from circuits_qiskit import circuit_from_adapt, statevector_energy
    from qubit_nature import our_to_sparse_pauli

    _, pr, res = mock_cas66
    k = len(res.ops)
    qc = circuit_from_adapt(pr, res, k, tapered=False)
    assert abs(statevector_energy(qc, our_to_sparse_pauli(pr.H, 12)) - res.energies[k]) < 1e-6


def test_d3_transpile_and_ideal_aer_layout(mock_cas66):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    from circuits_qiskit import circuit_from_adapt, fake_backend, gate_report, transpile_isa
    from qubit_nature import our_to_sparse_pauli
    from run_aer import estimate_energy, fold_all_gates, fold_two_qubit_gates, ideal_simulator

    _, pr, res = mock_cas66
    k = 3
    qc = circuit_from_adapt(pr, res, k)
    be = fake_backend("FakeFez")
    isa = transpile_isa(qc, be)
    rep = gate_report(isa)
    assert rep["two_qubit"] > 0 and set(rep["ops"]) <= {"cz", "rz", "sx", "x", "id", "barrier"}
    H = our_to_sparse_pauli(pr.Ht, pr.tp.n_tapered)
    sim = ideal_simulator()
    assert abs(estimate_energy(isa, H, sim) - res.energies[k]) < 1e-6
    assert abs(estimate_energy(fold_two_qubit_gates(isa, 3), H, sim, layout=isa.layout) - res.energies[k]) < 1e-6
    folded = fold_all_gates(isa, 5)  # ISA-only inverses (sxdg = rz(pi) sx rz(-pi)) must be exact
    assert set(folded.count_ops()) <= {"cz", "rz", "sx", "x", "id", "barrier"}
    assert folded.count_ops().get("sx", 0) == 5 * isa.count_ops().get("sx", 0)
    assert abs(estimate_energy(folded, H, sim, layout=isa.layout) - res.energies[k]) < 1e-6


def test_d4_sampled_energy_ideal(mock_cas66):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    from circuits_qiskit import circuit_from_adapt, fake_backend
    from qubit_nature import our_to_sparse_pauli
    from run_aer import ideal_simulator, sampled_energy

    _, pr, res = mock_cas66
    k, shots = 3, 200_000
    be = fake_backend("FakeFez")
    H = our_to_sparse_pauli(pr.Ht, pr.tp.n_tapered)
    out = sampled_energy(circuit_from_adapt(pr, res, k), H, be, ideal_simulator("automatic"), shots=shots, rem=False)
    tol = 5 * sum(abs(c) for lab, c in H.to_list() if set(lab) != {"I"}) / np.sqrt(shots)
    assert abs(out["raw"] - res.energies[k]) < tol


# ----------------------------------------------------------------------------- (e) tapering
def test_e_tapered_qubits_mock(mock_cas66):
    _, pr, _ = mock_cas66
    assert pr.tp.n_tapered == 7


def test_e2_tapered_qubits_pyscf():
    pytest.importorskip("pyscf")
    from chem_pyscf import n2_problem_pyscf

    for basis, mode in (("sto-3g", "rhf_irrep"), ("sto-3g", "rhf_w2p"), ("sto-3g", "casscf_irrep")):
        P = n2_problem_pyscf(1.098, basis, "cas66", mode)[0]
        assert prepare(P).tp.n_tapered == 7, (basis, mode)


# ----------------------------------------------------------------------------- SQD ordering
def test_f_sqd_bit_ordering():
    pytest.importorskip("qiskit_addon_sqd")
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

    from sqd_ibm import bit_array_to_ints, ints_to_bit_array

    alpha, beta, n = 0b001011, 0b000111, 6  # alpha occ {0,1,3}, beta occ {0,1,2}
    full = alpha | (beta << n)
    ba = ints_to_bit_array(np.array([full]), 2 * n)
    assert ba.get_bitstrings()[0] == format(beta, "06b") + format(alpha, "06b")  # alpha on the right
    assert bit_array_to_ints(ba)[0] == full
    sa, sb = bitstring_matrix_to_ci_strs(ba.to_bool_array(), open_shell=True)
    assert list(sa) == [alpha] and list(sb) == [beta]


def test_g_sqd_end_to_end_noiseless(mock_cas66):
    pytest.importorskip("qiskit_addon_sqd")
    from sqd_ibm import ints_to_bit_array, run_sqd
    from src.sqd import untaper_bits

    P, pr, res = mock_cas66
    rng = np.random.default_rng(0)
    p = np.abs(res.state) ** 2
    t_bits = rng.choice(len(p), size=2000, p=p / p.sum())
    full = untaper_bits(pr.tp, t_bits.astype(np.int64))
    out = run_sqd(P, ints_to_bit_array(full, 12), samples_per_batch=200, num_batches=3, max_iterations=5,
                  max_frac=0.25, verbose=False)
    # variational and no worse than HF (subsampling makes "<= ADAPT energy" flaky); capped subspace
    assert pr.e_exact - 1e-8 <= out["energy"] <= res.energies[0] + 1e-8
    assert out["subspace_fraction"] <= 0.25 + 1e-12


def test_h_untaper_preserves_sector(mock_cas66):
    from src.sqd import untaper_bits

    _, pr, res = mock_cas66
    full = untaper_bits(pr.tp, np.arange(1 << pr.tp.n_tapered, dtype=np.int64))
    good = set(sector_indices(12, 3, 3, 6).tolist())
    assert int(full[pr.tp.ref_bits]) == 0b000111000111
    # every basis state in the support of the tapered ADAPT state untapers to an (3a,3b) determinant
    support = np.nonzero(np.abs(res.state) > 1e-10)[0]
    assert all(int(full[i]) in good for i in support)


# ----------------------------------------------------------------------------- honest-SQD helpers / ZNE (no deps)
def test_i_subspace_cap(mock_cas66):
    from sqd_ibm import full_dim, max_dim_for_fraction

    P, _, _ = mock_cas66
    assert full_dim(P) == 400
    md = max_dim_for_fraction(P, 0.05)
    assert md ** 2 / full_dim(P) <= 0.05 and md == 4


def test_j_exponential_extrapolation_with_asymptote():
    from run_aer import extrapolate

    a, b, c = -100.0, 0.7, 0.3
    s = [1, 3, 5]
    v = [a + b * np.exp(-c * x) for x in s]
    assert abs(extrapolate(s, v, "exponential", asymptote=a) - (a + b)) < 1e-9
