# QC-02 — Research draft & mock build (v0, 25 Oct 2026)

**Problem statement:** compute the ground-state energy of a strongly correlated molecular
fragment to chemical accuracy (1.6 mHa) while mitigating hardware noise, using CAS selection,
tapered Hamiltonians, ADAPT-VQE + error mitigation, benchmarked against DMRG/CCSD(T).

**What this mock is:** a complete, verified pipeline in pure NumPy/SciPy (the cloud workspace
could not install PySCF/Qiskit — pypi was blocked). Every stage was checked against published
numbers or an independent exact calculation. Molecule: N₂ bond breaking, STO-3G basis.
On laptops the same pipeline ports to PySCF + Qiskit with a bigger basis (see §6).

---

## 1. Headline results

**v2 (27 Oct 2026): revised after an expert-panel review** (quantum chemist, quantum
engineer, hackathon judge). Fixes: D₂h-symmetrised RHF so every geometry tapers to 7 qubits;
singlet pinned with a spin penalty H + 0.5·S² (the lowest state at 3.0 Å is a septet); DIIS
coupled cluster, which converges further and exposes the collapse below the exact energy;
ZNE that amplifies *all* gates (1q included); an honest fixed-budget SQD comparison against
random and classical baselines; and a corrected fault-tolerance estimate.

| Finding | Number | Figure |
|---|---|---|
| CCSD(T) misses chemical accuracy already at equilibrium (FV 10e,8o) | +2.18 mHa at 1.098 Å (CAS(6e,6o): +1.95) | `pes.png` |
| CCSD(T) drifts off, then **collapses below the exact energy** | +10.8 mHa at 1.5 Å; −64 to −280 mHa from 1.9 Å; no convergence at 2.4 / 2.7 Å | `pes.png` |
| Tapering: Z₂ symmetries (spin parities + D₂h), all 12 geometries | 12 → 7 qubits; 16 → 11 qubits (full valence) | — |
| Tapering roughly halves the circuit | 1,856 → 888 CX at equilibrium (CAS) | — |
| ADAPT-VQE (singlet) reaches chemical accuracy at **every** geometry (noiseless) | CAS: 19–40 ops, 840–2,112 CX · FV: 47–96 ops, 4.1–7.1k CX, worst 1.23 mHa | `pes.png`, `adapt_convergence.png` |
| At today's noise (r₂ = 3×10⁻³) the accurate circuit is pure noise | 1.1–1.8 Ha off after REM + ZNE (1,048 / 1,792 CX) | `noise.png` |
| Gate error needed for chemical accuracy (REM + exponential ZNE, all gates folded) | r₂ ≈ 3×10⁻⁵–10⁻⁴: 30–100× better than today | `noise.png` |
| PEC is unbiased but its cost explodes | 10¹³–10¹⁸ shots per term at r₂ = 3×10⁻³ | — |
| **Fixed-budget SQD: the noisy chip picks better configurations than chance** | FV 11 q, 2.1 Å, 192-CX circuit at r₂ = 3×10⁻³, 3,000 shots, 10 seeds, solving in 25% of the space: chip 9.3 mHa, random bitstrings 227 mHa (24×), HF-ranked classical 113 mHa (12×); chip reaches 1.6 mHa at 62% of the space | `sqd.png` |
| SQD caveat | Uncapped, ~300 random bitstrings cover this 3,136-determinant space, so the uncapped "0.30 mHa" result is not evidence of quantum value | — |
| Rough fault-tolerant estimate (no factories) | 7 logical qubits, distance 15, ~6,300 physical qubits (data + routing), ~6.9k–10.6k T gates | — |

**The story for judges:** (1) the classical gold standard drifts off and then collapses
exactly where chemistry gets interesting; (2) ADAPT-VQE fixes it in principle, with tapering
halving the cost; (3) honest noise analysis shows VQE-at-chemical-accuracy needs gates
30–100× better than today; (4) SQD changes the chip's job to choosing configurations, and at a
fixed budget a noisy chip chooses ~24× better than chance; the real payoff needs spaces too
large to enumerate (IBM: N₂ with 58 qubits).

## 2. What was verified (and how)

| Component | Check | Result |
|---|---|---|
| Gaussian integrals + RHF (`src/integrals.py`, `src/scf.py`) | H₂O/STO-3G vs Crawford project #3 | −74.942079928 (matches to 3×10⁻¹³) |
| | N₂/STO-3G vs literature | −107.495975 (lit. −107.4959) |
| CCSD / (T) (`src/ccsd.py`) | H₂O vs Crawford projects #5/#6 | match to 1×10⁻¹⁰ |
| JW qubit Hamiltonian + FCI (`src/pauli.py`) | ⟨HF|H|HF⟩ = RHF; FCI ≤ CCSD(T) | exact; FCI −75.0129802 |
| Tapering (`src/taper.py`) | tapered ⟨HF|H|HF⟩; exact energy in tapered spectrum; tapered vs untapered ADAPT | all agree to 1×10⁻⁹ |
| Noisy simulator (`src/noise.py`) | noiseless gate-level density matrix vs exact unitaries; vs ADAPT statevector | match to 1×10⁻¹⁰ / 1×10⁻⁷ |

Tests: `python3 tests/test_integrals_scf.py`, `tests/test_fci_ccsd.py`, `tests/test_taper_adapt.py` (all pass).

## 3. Method choices (and why)

- **System: N₂ triple-bond breaking.** Textbook strong static correlation; CCSD(T) is known to
  fail here, so the "beyond classical gold standard" claim is real, not staged.
- **Active space = CAS(6e,6o)** of the three bonding (πᵤ×2, 3σg) and three antibonding
  (πg×2, 3σᵤ) orbitals, embedded in the frozen-core RHF mean field. This *is* an active-space
  embedding (CASCI-in-HF). Embedding error vs full valence: ~30 mHa — motivates CASSCF orbitals,
  NEVPT2 correction and proper DMET/projection embedding in the real build.
- **Fair comparison:** CCSD(T), exact and ADAPT-VQE all act on the *identical* active-space
  Hamiltonian, so differences are method error only.
- **Symmetric RHF branch:** the lowest RHF breaks spatial symmetry at long bonds (tapering then
  drops to 2 qubits and the Hamiltonian grows 7×). We follow the symmetric branch by orbital
  continuation and rotate degenerate π pairs to pure px/py so point-group symmetries become
  tapering Z-strings.
- **ADAPT pool:** spin-conserving fermionic UCCSD singles/doubles; symmetry filtering cuts 117 → 27.
- **Noise model:** depolarising 2q (r₂) and 1q (r₁) gate errors + symmetric readout flips;
  defaults r₂=3×10⁻³, r₁=2.5×10⁻⁴, readout 1.5×10⁻². These are order-of-magnitude stand-ins —
  replace with a real IBM FakeBackend calibration on laptops.

## 4. Honest limitations of the mock

1. **STO-3G** (minimal basis). Absolute energies aren't chemically meaningful; method-vs-method
   errors are. Real build: cc-pVDZ via PySCF, CASSCF/AVAS orbitals.
2. **No DMRG yet** — exact diagonalisation replaces it at these sizes. DMRG (block2) becomes
   necessary for active spaces > ~18 orbitals.
3. **CNOT counts are naive** (one CNOT ladder per Pauli string, all-to-all connectivity). Optimised
   fermionic-excitation circuits cut this ~2–3×; heavy-hex routing adds SWAPs. Recount with the
   Qiskit transpiler on a FakeBackend.
4. **No dynamical decoupling** — needs idle-time/coherent noise, which the model lacks. Test DD via
   qiskit-ibm-runtime options on a FakeBackend.
5. **PEC** is evaluated analytically (exact noise knowledge → unbiased mean + overhead γ), not by sampling.
6. **SQD caveat:** the FV(10e,8o) sector has only 3,136 determinants, so enough shots cover it
   entirely (10k shots = exact). The meaningful signal is that *few* shots from a *shallow, noisy*
   circuit already give < 1.6 mHa while VQE on the same circuit is off by > 1 Ha. SQD's real value
   shows at 50–100+ qubits where the sampled subspace is a tiny fraction.
7. At 2.4 Å and 3.0 Å the symmetric-branch continuation slips (7 → 9–10 tapered qubits); fix by
   symmetry-labelled orbital tracking.
8. FT resource estimate is a back-of-envelope (surface-code scaling, no T-factory footprint).

## 5. What is buildable at the hackathon (verdict)

| Deliverable | Status in mock | Hackathon effort |
|---|---|---|
| CAS selection + classical baselines (HF/CCSD/CCSD(T)/exact) | ✅ done | 2–3 h to port to PySCF cc-pVDZ |
| Tapered Hamiltonian | ✅ done | 1 h (qiskit-nature tapered mapper) |
| ADAPT-VQE to chemical accuracy across PES | ✅ done | 2 h (keep own ADAPT; Qiskit for circuits) |
| Noise model + ZNE/REM/PEC ablation | ✅ done (own sim) | 3–4 h with Aer FakeBackend + mitiq |
| DD | ❌ | 1 h via runtime options |
| SQD (hardware-robust route) | ✅ teaching version | 3–4 h with qiskit-addon-sqd |
| Real IBM hardware point | ❌ | 1–2 h; free Open Plan time, SQD-style shallow circuit |
| DMRG reference for bigger CAS | ❌ | 1 h (block2) |
| FT resource estimate | ✅ rough | 1 h polish |
| Streamlit demo | ❌ | 3 h |

**Bottom line: QC-02 is fully buildable.** The core science is already working; the hackathon
work is porting to the real stack, a bigger basis, a hardware run and the demo.

## 6. Port plan to the real stack

| Mock module | Real-stack replacement |
|---|---|
| `src/integrals.py`, `src/scf.py`, `src/n2.py` | PySCF: `gto`, `scf.RHF`, `mcscf.CASSCF`, AVAS (`port/pyscf_active_space.py` is a starting point, **unverified**) |
| `src/ccsd.py` | `pyscf.cc.CCSD(...).ccsd_t()` (keep ours as a cross-check) |
| `src/pauli.py`, `src/taper.py` | qiskit-nature 0.8 `JordanWignerMapper` + `problem.get_tapered_mapper` |
| `src/adapt.py` | keep (fast, verified); export circuits as Qiskit `PauliEvolutionGate`s |
| `src/noise.py` | qiskit-aer `NoiseModel.from_backend(FakeFez())`, mitiq ZNE/PEC, runtime resilience levels |
| `src/sqd.py` | `qiskit-addon-sqd` (configuration recovery + subspace diagonalisation) |
| exact diag | block2 DMRG for larger CAS |

Install: `pip install -r requirements-laptop.txt`.

## 7. Next steps (ordered)

1. Port to PySCF cc-pVDZ, CASSCF(6,6) orbitals; re-run the baseline (expect CCSD(T) failure to persist).
2. Add NEVPT2/DMRG references; show "embedded ADAPT + NEVPT2" vs CCSD(T) on the full PES.
3. Replace the noise model with FakeFez/FakeTorino; transpile for real CX counts; add DD.
4. SQD with qiskit-addon-sqd on a shallow LUCJ/ADAPT circuit; one real-hardware run at 2.1 Å.
5. Differentiator: switch system to a biradical (p-benzyne singlet–triplet gap) or a small
   catalyst fragment once the N₂ pipeline is solid.
6. Streamlit app: PES curves, error vs 1.6 mHa line, mitigation toggles, resource panel.

## Reproduce

```
python3 tests/test_integrals_scf.py && python3 tests/test_fci_ccsd.py && python3 tests/test_taper_adapt.py
python3 scripts/run_pes.py        # ~20 min on 1 core
python3 scripts/run_noise.py      # ~5 min
python3 scripts/run_sqd.py        # ~5 min
python3 scripts/run_ft_estimate.py
python3 scripts/make_plots.py
```

## Web app (live engine)

```
./run.sh                          # or: python -m web.backend.app  ->  http://localhost:8000
python3 tests/test_web_api.py     # API reproduces results/pes.json and results/noise.json
```

Bond explorer at any bond length, live ADAPT-VQE, noise lab with sweep and fault-tolerant estimate,
fixed-budget SQD, job history and an in-browser run of every regression suite. See `web/README.md`.
