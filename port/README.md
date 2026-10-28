# QC-02 production port (PySCF + Qiskit stack)

Production port. Written on 2026-10-27 against the library versions below. **Not executed in the
authoring environment** (no package access there): run `port/test_port.py` on a laptop first.
Every API was checked against the library's **source code at its release tag** (shallow git clones
of the tags; docstrings + signatures). Of the doc URLs in the table, the qiskit-addon-sqd API /
release-notes / tutorial pages were opened directly; the others are the canonical doc locations of
the same docstrings (their sites were not reachable from the authoring sandbox).

## Laptop setup

```bash
# Python 3.12 (mitiq 1.1 needs <3.13, ffsim needs >=3.11). Linux/macOS; on Windows use WSL.
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-laptop.txt
# token only for real hardware (never commit it):
export QISKIT_IBM_TOKEN=...            # optional
export QISKIT_IBM_INSTANCE=crn:...     # optional
```

## Commands

```bash
python -m pytest port/test_port.py -v                      # 1. acceptance tests (run first)
python port/run_all.py --basis sto-3g --r 1.098 2.1        # 2. cross-check vs the NumPy prototype
python port/run_all.py --basis cc-pvdz --r 1.098 1.5 2.1 --backend FakeFez --lucj --dmrg   # 3. production
python port/chem_pyscf.py --basis cc-pvdz --r 1.098 2.1   # chemistry + classical benchmarks only
python port/run_all.py --basis sto-3g --cas fv108 --r 2.1  # SQD test in a non-trivial space (3136 dets)
python port/run_hardware.py --dry-run                      # print the hardware plan / options
python port/run_hardware.py --cas fv108 --ansatz lucj --r 2.1            # real IBM run, 16 qubits
python port/run_hardware.py --basis cc-pvdz --cas cas10e16o --ansatz lucj --r 2.1 --max-frac 0.01  # 32 qubits
```

`run_all.py` skips any stage whose library is missing (with a message) and writes
`results/port/run_<basis>_<cas>_<timestamp>.json`.

## Expected outputs (STO-3G cross-check)

| quantity | expected | test |
|---|---|---|
| RHF, R = 1.098 Å (symmetric branch) | −107.495975 Ha (±1e-5) | `test_a_pyscf_rhf_sto3g_eq` |
| CAS(6e,6o) exact singlet, 1.098 Å (RHF orbitals, chosen by irrep = prototype set) | −107.622015 Ha | `test_a2_…` |
| FV(10e,8o) exact singlet, 2.1 Å (⟨S²⟩ = 0), 16 → 11 qubits | −107.448604 Ha | `test_a3_…` |
| qubits CAS(6e,6o): JW → tapered | 12 → 7 (ours and qiskit-nature) | `test_b2_…`, `test_e*` |
| qiskit-nature JW coefficients vs ours | identical (≤1e-8) | `test_b_…` |
| Qiskit circuit (PauliEvolutionGate) statevector energy vs our ADAPT | ≤1e-6 | `test_d_…` |
| ADAPT ops to chemical accuracy (prototype) | 21 at 1.098 Å, 33–35 at 1.9–2.1 Å | `run_all` table |

In `run_all` the table shows, per geometry: exact energy, CCSD(T) error, ADAPT ops to chemical
accuracy, transpiled 2-qubit (CZ) counts tapered/untapered on the fake backend, and for the shallow
`--k-shallow` circuit: noisy, ZNE (Richardson) and readout-mitigated errors, plus the SQD error.
Prototype expectation: noisy VQE at chemical-accuracy depth is far off; SQD from a shallow noisy
circuit gets within ~1 mHa.

## Honest SQD protocol

A small active space proves nothing: CAS(6e,6o) has 400 determinants and any noisy sampler (even
uniform noise) covers it, so "SQD reached chemical accuracy" is then meaningless. Every SQD number
we report therefore comes with:

1. **Capped subspace.** `qiskit-addon-sqd` `max_dim` (per spin) is set from `--max-frac` so that
   subspace/full-space ≤ 5% by default (FV(10e,8o): 12² = 144 of 3136 determinants; cc-pVDZ
   CAS(10e,16o): 4368² ≈ 1.9·10⁷ determinants, use `--max-frac 0.01` or smaller). The fraction is
   printed next to every energy.
2. **Random-bitstring baseline**: uniformly random bitstrings, same shot count, through the *same*
   capped SQD call (`sqd_ibm.random_baseline`).
3. **Classical CISD-ranked baseline** at the same per-spin size: α/β strings ranked by marginal
   CISD weight, diagonalised with the same SCI solver (`sqd_ibm.cisd_ranked_baseline`).
4. Default problem for hardware: **FV(10e,8o)** (16 qubits) with an ffsim **LUCJ** circuit; larger
   cc-pVDZ spaces (e.g. `cas10e16o`, 32 qubits) are supported for LUCJ + SQD (no FCI reference there:
   use `chem_pyscf.dmrg_energy`).

A device result is only interesting if it beats both baselines at the same subspace fraction.
Exact references are **singlet** energies (spin-constrained FCI / `pr.e_exact` of the prototype with
penalty H + μS², μ = 0.5 Ha by default, `--mu`).

## Files

| file | content |
|---|---|
| `chem_pyscf.py` | N2 → `src.n2.Problem` (h1eff, chemist ERI, ecore); symmetric-branch RHF (D2h + pinned irrep occupations + density continuation, Newton fallback, stability report); active orbitals **selected and ordered by D2h irrep** (not by occupation); CAS modes `rhf_irrep` (STO-3G default = prototype), `rhf_w2p`, `casscf_irrep` (production), `casscf_avas`; general `cas<N>e<M>o` (e.g. `cas10e16o` in cc-pVDZ); **singlet-constrained** CASCI/CASSCF/FCI (`fix_spin_(ss=0)`); CCSD(T) in full valence / full space (in CAS(6e,6o)/D2h (T) ≡ 0); NEVPT2; block2 DMRG |
| `qubit_nature.py` | qiskit-nature JW / tapered Hamiltonians; `our_to_sparse_pauli` / `sparse_pauli_to_our` (+ pure-Python label converters); `spin_penalized_sparse_pauli` (H + μS², mirrors `src.pipeline.prepare(P, mu)`) |
| `circuits_qiskit.py` | ADAPT result → `QuantumCircuit` (HF X gates + `PauliEvolutionGate`s), fake-backend transpilation, gate report |
| `run_aer.py` | Aer noisy estimator (MPS fallback on memory errors), ZNE by folding **all** ISA gates (ISA-native inverses) + mitiq linear/Richardson/**exponential with fully-mixed asymptote c_I**, readout-mitigated sampling estimator with basis change on the **logical** circuit before transpilation, experimental client-side runtime Estimator in local mode |
| `run_hardware.py` | `QiskitRuntimeService` (env-var token); default **FV(10e,8o) + LUCJ** (cc-pVDZ `cas10e16o` supported); capped SQD + random + CISD baselines; Estimator with TREX/ZNE/PEA/truncated-PEC/DD/twirling options; client-side primitives with fallback on import *or* construction errors; dry run |
| `sqd_ibm.py` | qiskit-addon-sqd `diagonalize_fermionic_hamiltonian` **always capped** (`max_dim`), subspace fraction printed; `random_baseline`, `cisd_ranked_baseline`; BitArray ordering; untapering |
| `lucj_ffsim.py` | LUCJ circuit from active-space CCSD amplitudes (ffsim) |
| `run_all.py` | the CLI pipeline |
| `test_port.py` | acceptance tests (pytest.importorskip per library) |
| `pyscf_active_space.py` | deprecated shim → `chem_pyscf` |

## Conventions (why no permutations are needed)

* Spin orbitals: alpha `0..n-1`, beta `n..2n-1` — ours, qiskit-nature, ffsim (`alpha on the first norb qubits`).
* Qubit q ↔ bit q of every integer; Qiskit labels/bitstrings print qubit 0 rightmost.
* Our Pauli `(x, z)` = i^{popcount(x&z)} X^x Z^z ⇒ each (1,1) is a proper Y, so a term maps to the
  Qiskit label with the **same** coefficient.
* SQD addon expects `[b_N..b_0 a_N..a_0]` (alpha on the right) = our integer printed by Qiskit.
* `PauliEvolutionGate(P, time=t)` = exp(−i t P); our rotation exp(−i θ/2 P) ⇒ `time = θ/2`
  (numerically re-checked in the sandbox with a NumPy emulation: 1e-13).

## API table (what we call, version, where verified)

Source at tag = the code was read in a shallow clone of the release tag on 2026-10-27.

| library (version) | API used | verified against |
|---|---|---|
| PySCF 2.14.0 | `gto.M(atom, basis, unit, symmetry='D2h')` | https://pyscf.org/user/gto.html ; source at tag v2.14.0 |
| | `scf.RHF`, `mf.irrep_nelec`, `mf.get_irrep_nelec()`, `mf.newton()` | https://pyscf.org/user/scf.html ; `examples/scf/13-symmetry.py` |
| | `mf.stability(return_status=True)` → `(mo_i, mo_e, stable_i, stable_e)` | `pyscf/scf/stability.py` (v2.14.0) |
| | `mcscf.CASCI/CASSCF`, `.sort_mo(caslst, base=0)`, `.sort_mo_by_irrep(nocc, ncore, mo_coeff=)`, `.get_h1eff()`, `.get_h2eff()` | https://pyscf.org/user/mcscf.html ; `pyscf/mcscf/casci.py`, `mc1step_symm.py` |
| | `mcscf.avas.avas(mf, ['N 2p'], canonicalize=True)` | https://pyscf.org/user/mcscf.html ; `pyscf/mcscf/avas.py` |
| | `ao2mo.restore(1/8, eri, n)` | https://pyscf.org/user/ao2mo.html |
| | `cc.CCSD(mf, frozen=2)`, `.ccsd_t()`; custom-Hamiltonian RHF/CCSD | https://pyscf.org/user/cc.html ; `examples/cc/40-ccsd_custom_hamiltonian.py` |
| | `fci.direct_spin1.kernel(h1, g, norb, nelec, ecore=)`, `fci.direct_spin1.FCI()` + `fci.addons.fix_spin_(solver, ss=0)`, `fci.spin_op.spin_square0`, `mc.fix_spin_(ss=0)` ; `mrpt.NEVPT(mc).kernel()` | https://pyscf.org/user/ci.html ; `pyscf/fci/addons.py`, `pyscf/mcscf/casci.py` |
| | `symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo, s=)` ; `ci.CISD(mf)`, `.to_fcivec(civec, norb, nelec)` ; `fci.cistring.make_strings(orbs, nelec)` | `pyscf/symm/addons.py`, `pyscf/ci/cisd.py`, `pyscf/fci/cistring.py` (v2.14.0) |
| block2 0.5.4 | `DMRGDriver(scratch, symm_type=SymmetryTypes.SU2, n_threads)`, `initialize_system(n_sites, n_elec, spin)`, `get_qc_mpo(h1e, g2e, ecore)`, `get_random_mps(tag, bond_dim, nroots)`, `dmrg(mpo, ket, n_sweeps, bond_dims, noises, thrds)` | https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html ; `pyblock2/driver/core.py` (v0.5.4) |
| qiskit-nature 0.8.0 | `ElectronicEnergy.from_raw_integrals(h1_a, h2_aa, auto_index_order=True)`, `.nuclear_repulsion_energy` (NOT mapped), `ElectronicStructureProblem(ham)`, `.num_particles`, `.get_tapered_mapper(JordanWignerMapper())`, `mapper.map(op)` → `SparsePauliOp` | https://qiskit-community.github.io/qiskit-nature/ ; source branch stable/0.8 |
| Qiskit 2.6.1 | `SparsePauliOp.from_list(.., num_qubits=)`, `.paulis.x/.z/.phase`, `.apply_layout`, `.group_commuting(qubit_wise=True)`, `.to_matrix(sparse=)` | https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp |
| | `PauliEvolutionGate(op, time)` (= exp(−i t op)) | https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate |
| | `qiskit.transpiler.generate_preset_pass_manager(optimization_level, backend, seed_transpiler)` | https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler |
| | `Statevector(qc).expectation_value(op)` ; `BitArray.from_samples / get_int_counts / to_bool_array / get_bitstrings` | https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.primitives.BitArray |
| qiskit-aer 0.18.2 | `AerSimulator.from_backend(backend, method=)`, `qiskit_aer.primitives.EstimatorV2.from_backend(sim, options=)`, `SamplerV2.from_backend` | https://qiskit.github.io/qiskit-aer/ ; source at tag 0.18.2 |
| qiskit-ibm-runtime 0.51.0 | `fake_provider.FakeFez/FakeTorino/...` | https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/fake-provider ; `fake_provider/backends/*/conf_*.json` |
| | `QiskitRuntimeService(channel='ibm_quantum_platform', token, instance)`, `least_busy(operational, simulator, min_num_qubits)` | https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/qiskit-runtime-service |
| | `executor_estimator.Estimator` / `executor_sampler.Sampler` (new, client-side; legacy `EstimatorV2`/`SamplerV2` deprecated in 0.50), `NoiseLearnerV3` | release notes 0.50.0 https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/release-notes ; `executor_estimator/__init__.py` |
| | options: `resilience_level`, `resilience.measure_mitigation` (TREX), `resilience.zne_mitigation`, `resilience.zne.{noise_factors, extrapolator, amplifier}`, `resilience.pec_mitigation`, `resilience.pec.max_overhead`, `dynamical_decoupling.{enable, sequence_type}`, `twirling.{enable_gates, enable_measure}` | https://quantum.cloud.ibm.com/docs/guides/estimator-options ; `options_models/*.py` and `options/*.py` (0.51.0) |
| | local testing mode ignores resilience options (warning) | https://quantum.cloud.ibm.com/docs/guides/local-testing-mode ; `fake_provider/local_service.py` |
| qiskit-addon-sqd 0.14.0 | `fermion.diagonalize_fermionic_hamiltonian(h1, h2, bit_array, samples_per_batch, norb, nelec, *, num_batches, max_iterations, sci_solver, symmetrize_spin, max_dim, initial_occupancies, callback, seed)` → `SCIResult(energy, sci_state, ...)`; `fermion.solve_sci_batch(..., spin_sq, **kernel_fixed_space_kwargs)`; `fermion.bitstring_matrix_to_ci_strs`; `fermion.solve_sci((strs_a, strs_b), h1, h2, norb, nelec, spin_sq=)`; `counts.generate_bit_array_uniform(num_samples, num_bits, rand_seed=)` | https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-sqd/fermion ; tutorial https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization ; release notes https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-sqd/release-notes ; source tag 0.14.0 |
| mitiq 1.1.0 | `mitiq.zne.inference.RichardsonFactory.extrapolate(scales, values)`, `LinearFactory.extrapolate`, `ExpFactory.extrapolate(scales, values, asymptote=)`; `mitiq.zne.execute_with_zne(circ, executor, factory=, scale_noise=fold_global)` | https://mitiq.readthedocs.io/en/stable/apidoc.html ; source tag v1.1.0 |
| ffsim 0.0.85 | `UCJOpSpinBalanced.from_t_amplitudes(t2, t1=, n_reps=, interaction_pairs=(aa, ab), optimize=)`, `ffsim.qiskit.PrepareHartreeFockJW(norb, nelec)`, `ffsim.qiskit.UCJOpSpinBalancedJW(op)`, `ffsim.qiskit.generate_lucj_pass_manager(backend, norb, connectivity, interaction_pairs, **pm_kwargs)` → `(pm, pairs_ab)`, `ffsim.qiskit.PRE_INIT` | https://qiskit-community.github.io/ffsim/ ; SQD tutorial above ; source tag v0.0.85 |

## Mitigation: local vs hardware

| technique | local (fake backend) | real IBM hardware |
|---|---|---|
| ZNE | `run_aer.zne_energy` (fold **all** ISA gates ×1,3,5; mitiq linear / Richardson / exponential with asymptote = identity coefficient of H; 2q-only folding also reported for comparison); `mitiq_zne_logical` (mitiq `fold_global`) alternative | `resilience.zne_mitigation=True`, `zne.noise_factors=(1,3,5)`, `zne.extrapolator=("exponential","linear")` |
| readout | `run_aer.sampled_energy(rem=True)` (tensored symmetric REM from `target['measure']` errors) | `resilience.measure_mitigation=True` (TREX) + `twirling.enable_measure=True` |
| PEC / PEA | analytic overhead only (`src/noise.pec_overhead`) | `pec_mitigation=True` with `pec.max_overhead=100` = **truncated (biased) PEC** once the true overhead exceeds the cap; or `zne.amplifier="pea"`; client-side Estimator needs `NoiseLearnerV3` first |
| DD | not modelled (no idle noise in our flow) | `dynamical_decoupling.enable=True, sequence_type="XY4"` |
| all of the above via runtime | legacy `EstimatorV2` local mode: **ignored** (warning). Client-side `executor_estimator.Estimator(mode=FakeFez())` with `resilience_level=2`: may work locally (Executor local mode, runtime ≥0.50) — `--runtime-local`, experimental | supported |
