# Production port. Written against qiskit-aer 0.18.2 (https://qiskit.github.io/qiskit-aer/),
# Qiskit SDK 2.6.1, qiskit-ibm-runtime 0.51.0 and mitiq 1.1.0 (https://mitiq.readthedocs.io)
# docs/source on 2026-10-27. Not executed in the authoring environment (no package access);
# run port/test_port.py on a laptop first.
"""Noisy energy estimation on a fake IBM backend + local error mitigation.

WHAT RUNS LOCALLY (this file)
  1. Noisy expectation values: ``AerSimulator.from_backend(FakeFez(), method='density_matrix')``
     wrapped by ``qiskit_aer.primitives.EstimatorV2.from_backend(sim)``. NOTE: Aer EstimatorV2
     evaluates <H> with ``save_expectation_value`` -> gate/idle noise is included but there is
     NO readout error and NO shot noise unless ``default_precision`` > 0
     (qiskit_aer/primitives/estimator_v2.py). Aer truncates untouched qubits of 156-qubit ISA
     circuits; if memory is still a problem we fall back to method='matrix_product_state'
     (noise then sampled by trajectories: expectation values carry trajectory noise).
  2. ZNE (local cross-check): ALL ISA gates are locally folded, G -> G (G^-1 G)^n, scale 1/3/5, with
     ISA-native inverses (cz, x self-inverse; rz(t)^-1 = rz(-t); sx^-1 = rz(pi) sx rz(-pi) up to a
     global phase, checked numerically) so 1q noise is amplified too (folding only 2q gates leaves an
     un-extrapolatable 1q floor). Run WITHOUT re-transpiling (that would cancel folds). Extrapolation:
     mitiq static ``LinearFactory`` / ``RichardsonFactory`` / ``ExpFactory.extrapolate(...,
     asymptote=c_I)`` where c_I = identity coefficient of the qubit Hamiltonian = fully-mixed-state
     energy (numpy fallbacks if mitiq is absent).
  3. Readout-error mitigation (sampling path): for each qubit-wise-commuting group the basis change
     (h / sdg+h) and measurements are added to the LOGICAL circuit, which is then transpiled to ISA
     (so the basis change is in cz/rz/sx/x and its noise is simulated); readout errors are simulated
     by AerSimulator.run; REM divides each Pauli expectation by prod_q (1 - 2 p_q) with
     p_q = backend.target['measure'][(q,)].error -- a symmetric tensored (TREX-like) approximation.
  4. Optional: the NEW client-side ``qiskit_ibm_runtime.executor_estimator.Estimator`` (added in
     runtime 0.50) does TREX + gate-folding ZNE on the client and its Executor supports local mode
     with a FakeBackend (release notes 0.50.0). Tried in ``runtime_local_mitigated`` -- EXPERIMENTAL.

WHAT ONLY WORKS ON REAL IBM HARDWARE (via Runtime; see run_hardware.py for the code)
  * Legacy ``qiskit_ibm_runtime.EstimatorV2`` in local testing mode IGNORES all resilience
    options (qiskit_ibm_runtime/fake_provider/local_service.py warns "resilience_level option has
    no effect in local testing mode" and "Options {...} have no effect in local testing mode").
  * TREX (resilience.measure_mitigation), ZNE (resilience.zne_mitigation, zne.noise_factors,
    zne.extrapolator, zne.amplifier='pea'), PEC (resilience.pec_mitigation, needs learned layer
    noise), dynamical decoupling (dynamical_decoupling.enable/sequence_type) and twirling
    (twirling.enable_gates/enable_measure) are applied server-side for the legacy primitives and
    need a real backend. PEC/PEA additionally need NoiseLearnerV3 with the client-side Estimator.

Docs: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
      https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.EstimatorV2.html
      https://quantum.cloud.ibm.com/docs/guides/local-testing-mode
      https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.inference
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuits_qiskit import TWO_Q_NAMES  # noqa: E402


# ----------------------------------------------------------------------------- simulator
def noisy_simulator(backend, method: str = "density_matrix", seed: int = 11):
    """AerSimulator with the fake backend's noise model + coupling map (AerSimulator.from_backend).
    Aer truncates qubits that the circuit does not touch, so 156-qubit ISA circuits that act on
    ~10 physical qubits are cheap even with the density-matrix method."""
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(backend, method=method, seed_simulator=seed)


def ideal_simulator(method: str = "statevector"):
    from qiskit_aer import AerSimulator

    return AerSimulator(method=method)


def _run_estimator(isa, obs, sim, precision: float, run_options: dict | None = None) -> float:
    from qiskit_aer.primitives import EstimatorV2

    opts = {"default_precision": precision}
    if run_options:
        opts["run_options"] = run_options
    est = EstimatorV2.from_backend(sim, options=opts)
    return float(np.real(est.run([(isa, obs)]).result()[0].data.evs))


def estimate_energy(isa, H_spo, sim, precision: float = 0.0, layout=None, fallback_backend=None,
                    mps_shots: int = 4000) -> float:
    """<H> on the noisy simulator via qiskit_aer.primitives.EstimatorV2 (PUB = (circuit, obs)).
    ``H_spo`` is on the LOGICAL qubits; it is laid out with ``SparsePauliOp.apply_layout``
    using ``layout`` (default: ``isa.layout``; pass it explicitly for folded copies).
    If the simulation fails (e.g. memory) and ``fallback_backend`` is given, retry with
    method='matrix_product_state' (noisy MPS = trajectories; ``mps_shots`` trajectories)."""
    layout = layout if layout is not None else isa.layout
    obs = H_spo.apply_layout(layout) if layout is not None else H_spo
    try:
        return _run_estimator(isa, obs, sim, precision)
    except Exception as exc:  # noqa: BLE001 - Aer raises generic errors on memory exhaustion
        if fallback_backend is None:
            raise
        print(f"[aer] {type(exc).__name__}: {exc}; retrying with matrix_product_state")
        mps = noisy_simulator(fallback_backend, method="matrix_product_state")
        return _run_estimator(isa, obs, mps, precision, run_options={"shots": mps_shots})


# ----------------------------------------------------------------------------- ZNE
_SELF_INVERSE = {"cz", "ecr", "cx", "x", "id", "swap"}


def _isa_inverse(ci):
    """List of (operation, qubits) implementing ci.operation^-1 with ISA gates only."""
    from qiskit.circuit.library import RZGate, SXGate

    op, q = ci.operation, ci.qubits
    if op.name in _SELF_INVERSE:
        return [(op, q)]
    if op.name == "sx":  # sxdg = rz(pi) sx rz(-pi) up to global phase (time order: rz(-pi), sx, rz(pi))
        return [(RZGate(-np.pi), q), (SXGate(), q), (RZGate(np.pi), q)]
    if op.name in ("rz", "rx", "ry", "p", "rzz"):
        return [(op.inverse(), q)]
    raise ValueError(f"no ISA inverse rule for gate {op.name!r}")


def fold_all_gates(isa, scale: int):
    """Local unitary folding of EVERY gate: G -> G (G^-1 G)^((scale-1)/2). Odd ``scale`` only.
    Barriers, measurements, delays and resets are copied unfolded."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd integer >= 1")
    out = isa.copy_empty_like()
    skip = {"barrier", "measure", "delay", "reset"}
    for ci in isa.data:
        out.append(ci.operation, ci.qubits, ci.clbits)
        if ci.operation.name in skip:
            continue
        inv = _isa_inverse(ci)
        for _ in range((scale - 1) // 2):
            for op, q in inv:
                out.append(op, q)
            out.append(ci.operation, ci.qubits, ci.clbits)
    return out


def fold_two_qubit_gates(isa, scale: int):
    """Two-qubit-only folding (kept for comparison; leaves a 1q-noise floor, see module doc)."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd integer >= 1")
    out = isa.copy_empty_like()
    for ci in isa.data:
        out.append(ci.operation, ci.qubits, ci.clbits)
        if ci.operation.name in TWO_Q_NAMES:
            for _ in range((scale - 1) // 2):
                out.append(ci.operation.inverse(), ci.qubits, ci.clbits)
                out.append(ci.operation, ci.qubits, ci.clbits)
    return out


def identity_coefficient(H_spo) -> float:
    """c_I = Tr(H)/2^n = energy of the fully mixed state = infinite-noise asymptote for ZNE."""
    return float(sum(np.real(c) for lab, c in H_spo.to_list() if set(lab) == {"I"}))


def extrapolate(scales, values, kind: str = "richardson", asymptote: float | None = None) -> float:
    """Zero-noise limit with mitiq's static extrapolators (mitiq.zne.inference), numpy fallback.
    kind: 'linear' | 'richardson' | 'exponential' (needs ``asymptote``: v(s) = a + b exp(-c s))."""
    s, v = list(map(float, scales)), list(map(float, values))
    try:
        from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory

        if kind == "linear":
            return float(LinearFactory.extrapolate(s, v))
        if kind == "richardson":
            return float(RichardsonFactory.extrapolate(s, v))
        if kind == "exponential":
            return float(ExpFactory.extrapolate(s, v, asymptote=float(asymptote)))
    except ImportError:
        pass
    if kind == "linear":
        return float(np.polyval(np.polyfit(s, v, 1), 0.0))
    if kind == "richardson":
        return float(np.polyval(np.polyfit(s, v, len(s) - 1), 0.0))
    if kind == "exponential":
        d = np.asarray(v) - asymptote
        if np.any(d == 0) or not (np.all(d > 0) or np.all(d < 0)):
            return float("nan")
        slope, icpt = np.polyfit(s, np.log(np.abs(d)), 1)
        return float(asymptote + np.sign(d[0]) * np.exp(icpt))
    raise ValueError(kind)


def zne_energy(isa, H_spo, sim, scales=(1, 3, 5), fold: str = "all", fallback_backend=None) -> dict:
    """Noisy energies at each fold scale + linear / Richardson / exponential(asymptote=c_I)."""
    folder = fold_all_gates if fold == "all" else fold_two_qubit_gates
    vals = [estimate_energy(folder(isa, s), H_spo, sim, layout=isa.layout, fallback_backend=fallback_backend)
            for s in scales]
    a = identity_coefficient(H_spo)
    return {"scales": list(scales), "values": vals, "fold": fold, "asymptote_cI": a,
            "linear": extrapolate(scales, vals, "linear"),
            "richardson": extrapolate(scales, vals, "richardson"),
            "exponential": extrapolate(scales, vals, "exponential", asymptote=a)}


def mitiq_zne_logical(qc_logical, executor, scales=(1.0, 3.0, 5.0)) -> float:
    """Full mitiq path (cross-check): mitiq ``fold_global`` on the LOGICAL circuit (decompose
    PauliEvolutionGates to cx/rz/h/sx first; mitiq converts via OpenQASM 2), and ``executor``
    must transpile with optimization_level=0 so folds survive. Unverified locally: install plain
    ``mitiq`` (its [qiskit] extra pins old runtime/aer).
    https://mitiq.readthedocs.io/en/stable/guide/zne-1-intro.html"""
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import RichardsonFactory
    from mitiq.zne.scaling import fold_global

    flat = qc_logical.decompose(reps=3)
    return float(execute_with_zne(flat, executor, factory=RichardsonFactory(list(scales)),
                                  scale_noise=fold_global))


# ----------------------------------------------------------------------------- readout path
def _basis_change_logical(qc, q: int, pauli: str):
    if pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.sdg(q)
        qc.h(q)


def _readout_errors(backend, phys: list[int]) -> list[float]:
    out = []
    for q in phys:
        try:
            err = backend.target["measure"][(q,)].error
        except Exception:  # noqa: BLE001
            err = None
        out.append(float(err) if err is not None else 0.0)
    return out


def sampled_energy(qc_logical, H_spo, backend, sim, shots: int = 20000, rem: bool = True, seed: int = 5,
                   optimization_level: int = 3) -> dict:
    """Shot-based <H> with readout errors, grouped by qubit-wise commuting sets, optional REM.

    For each group the basis change + measurement is appended to the LOGICAL circuit and the whole
    circuit is transpiled to ISA (cz/rz/sx/x on FakeFez) with a fixed seed. Classical bit q holds
    logical qubit q, so parsing is layout-independent; readout errors are looked up on the physical
    qubits of each transpiled circuit (``layout.final_index_layout()``)."""
    from qiskit import ClassicalRegister

    from circuits_qiskit import transpile_isa

    n = H_spo.num_qubits
    groups = H_spo.group_commuting(qubit_wise=True)
    const = total = total_raw = 0.0
    n2 = []
    for grp in groups:
        labels = grp.paulis.to_labels()
        basis = ["I"] * n
        for lab in labels:
            for q in range(n):
                if lab[n - 1 - q] != "I":
                    basis[q] = lab[n - 1 - q]
        circ = qc_logical.remove_final_measurements(inplace=False) if qc_logical.num_clbits else qc_logical.copy()
        creg = ClassicalRegister(n, "m")
        circ.add_register(creg)
        for q in range(n):
            _basis_change_logical(circ, q, basis[q])
        circ.measure(range(n), creg)
        isa = transpile_isa(circ, backend, optimization_level, seed)
        n2.append(sum(v for k, v in isa.count_ops().items() if k in TWO_Q_NAMES))
        phys = isa.layout.final_index_layout() if isa.layout is not None else list(range(n))
        p_ro = _readout_errors(backend, phys)
        counts = sim.run(isa, shots=shots, seed_simulator=seed).result().get_counts()
        ints = {int(k.replace(" ", ""), 2): v for k, v in counts.items()}
        for lab, c in zip(labels, grp.coeffs):
            mask = sum(1 << q for q in range(n) if lab[n - 1 - q] != "I")
            if mask == 0:
                const += float(np.real(c))
                continue
            ev = sum(v * (1 - 2 * (bin(b & mask).count("1") & 1)) for b, v in ints.items()) / shots
            damp = np.prod([1 - 2 * p_ro[q] for q in range(n) if (mask >> q) & 1])
            total_raw += float(np.real(c)) * ev
            total += float(np.real(c)) * (ev / damp if rem else ev)
    return {"raw": const + total_raw, "rem": const + total, "shots_per_group": shots,
            "n_groups": len(groups), "two_qubit_per_group": n2}


# ----------------------------------------------------------------------------- runtime local (experimental)
def runtime_local_mitigated(isa, H_spo, backend, resilience_level: int = 2, shots: int = 20000):
    """Client-side runtime Estimator (qiskit_ibm_runtime >= 0.50) in local mode with a FakeBackend:
    TREX + gate-folding ZNE run on the client. EXPERIMENTAL -- returns None on any failure.
    https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/executor-estimator"""
    try:
        from qiskit_ibm_runtime.executor_estimator import Estimator
    except ImportError:
        print("[runtime-local] qiskit_ibm_runtime.executor_estimator not available (<0.50) -> skipped")
        return None
    try:
        est = Estimator(mode=backend)
        est.options.resilience_level = resilience_level
        est.options.default_shots = shots
        obs = H_spo.apply_layout(isa.layout)
        res = est.run([(isa, obs)]).result()
        return float(np.real(res[0].data.evs))
    except Exception as exc:  # noqa: BLE001
        print(f"[runtime-local] client-side Estimator failed locally: {exc!r}")
        return None


def noisy_report(qc_logical, H_spo, backend, opt_level: int = 3, seed: int = 7, shots: int = 20000,
                 zne_scales=(1, 3, 5), try_runtime_local: bool = False) -> dict:
    """Transpile once; ideal / noisy / ZNE / readout-mitigated energies."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from circuits_qiskit import gate_report, statevector_energy, transpile_isa

    isa = transpile_isa(qc_logical, backend, opt_level, seed)
    sim = noisy_simulator(backend)
    out = {"gates": gate_report(isa), "ideal": statevector_energy(qc_logical, H_spo)}
    out["noisy"] = estimate_energy(isa, H_spo, sim, fallback_backend=backend)
    out["zne"] = zne_energy(isa, H_spo, sim, zne_scales, fold="all", fallback_backend=backend)
    out["zne_2q_only"] = zne_energy(isa, H_spo, sim, zne_scales, fold="2q", fallback_backend=backend)
    sim_shots = noisy_simulator(backend, method="automatic")
    out["sampled"] = sampled_energy(qc_logical, H_spo, backend, sim_shots, shots=shots, seed=seed)
    if try_runtime_local:
        out["runtime_local_rl2"] = runtime_local_mitigated(isa, H_spo, backend)
    return out
