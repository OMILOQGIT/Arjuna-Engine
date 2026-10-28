# Production port. Written against qiskit-ibm-runtime 0.51.0 (source tag 0.51.0;
# https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime) docs on 2026-10-27. Not executed in
# the authoring environment (no package access); run port/test_port.py on a laptop first.
"""Real IBM hardware runs: SamplerV2 bitstrings for SQD and EstimatorV2 with resilience options.

Credentials: NEVER hard-code a token. Either
  export QISKIT_IBM_TOKEN=...            # API key from quantum.cloud.ibm.com
  export QISKIT_IBM_INSTANCE=crn:...     # optional instance CRN
or save once with QiskitRuntimeService.save_account(...). Channel is "ibm_quantum_platform"
(0.51 accepts "ibm_quantum_platform", "ibm_cloud", "local"; the old "ibm_quantum" channel is gone).
With no credentials this script only prints the plan (dry run).

PRIMITIVE CLASSES (version note): qiskit-ibm-runtime 0.50 DEPRECATED the server-side
``qiskit_ibm_runtime.EstimatorV2`` / ``SamplerV2`` in favour of the client-side
``qiskit_ibm_runtime.executor_estimator.Estimator`` / ``executor_sampler.Sampler`` (same interface,
mitigation pre/post-processing on the client; PEC/PEA then need an explicit NoiseLearnerV3 step).
We use the client-side classes when importable and fall back to the legacy V2 classes otherwise.
Release notes: https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/release-notes (0.50.0)

Options reference: https://quantum.cloud.ibm.com/docs/guides/estimator-options ,
https://quantum.cloud.ibm.com/docs/guides/error-mitigation-and-suppression-techniques
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CHANNEL = "ibm_quantum_platform"


# ----------------------------------------------------------------------------- options (exact code)
def estimator_options(level: str = "zne") -> dict:
    """Resilience options as a dict accepted by ``Estimator(mode=backend, options=...)``.

    level='trex' : TREX readout mitigation (+ measurement twirling) only
    level='zne'  : TREX + gate-folding ZNE (1,3,5; exponential then linear) + DD (XY4) + gate twirling
    level='pea'  : TREX + ZNE with probabilistic error amplification (needs layer noise learning)
    level='pec'  : TRUNCATED (biased) PEC: pec.max_overhead=100 caps the sampling overhead gamma^2, so
                   the estimate is no longer unbiased once the true overhead exceeds the cap (our
                   prototype estimates gamma ~ 1e3-1e5 for chemically accurate circuits). Needs layer
                   noise learning.
    Field names checked in qiskit_ibm_runtime/options_models/{estimator,resilience,zne,twirling,
    dynamical_decoupling}.py (0.51.0) and identical in the legacy options/ package.
    """
    opts: dict = {
        "default_shots": 20_000,
        "resilience_level": 1,
        "resilience": {"measure_mitigation": True},          # TREX
        "twirling": {"enable_gates": True, "enable_measure": True},
        "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
    }
    if level in ("zne", "pea"):
        opts["resilience_level"] = 2
        opts["resilience"].update({
            "zne_mitigation": True,
            "zne": {"noise_factors": (1, 3, 5), "extrapolator": ("exponential", "linear"),
                    "amplifier": "gate_folding" if level == "zne" else "pea"},
        })
    if level == "pec":
        opts["resilience"].update({"pec_mitigation": True, "pec": {"max_overhead": 100}})
    return opts


def sampler_options() -> dict:
    """Sampler supports error *suppression* only (DD, twirling) -- no resilience options."""
    return {"default_shots": 100_000,
            "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
            "twirling": {"enable_gates": False, "enable_measure": True}}


# ----------------------------------------------------------------------------- service / primitives
def get_service():
    """QiskitRuntimeService from env vars or a saved account; None if no credentials."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[hw] qiskit-ibm-runtime not installed")
        return None
    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    try:
        if token:
            return QiskitRuntimeService(channel=CHANNEL, token=token, instance=instance)
        return QiskitRuntimeService()  # saved default account, if any
    except Exception as exc:  # noqa: BLE001
        print(f"[hw] no usable IBM Quantum credentials ({type(exc).__name__}) -> dry run")
        return None


def pick_backend(service, name: str | None = None, min_qubits: int = 12):
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)


def make_estimator(mode, options: dict):
    """Client-side Estimator (runtime >= 0.50); on ImportError OR any construction error (e.g. an
    option it rejects) fall back to the legacy server-side EstimatorV2."""
    try:
        from qiskit_ibm_runtime.executor_estimator import Estimator  # >= 0.50, client-side

        return Estimator(mode=mode, options=options)
    except Exception as exc:  # noqa: BLE001
        print(f"[hw] client-side Estimator unavailable ({exc!r}) -> legacy EstimatorV2")
        from qiskit_ibm_runtime import EstimatorV2

        return EstimatorV2(mode=mode, options=options)


def make_sampler(mode, options: dict):
    """Client-side Sampler (runtime >= 0.50) with fallback to legacy SamplerV2 on any error."""
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler  # >= 0.50, client-side

        return Sampler(mode=mode, options=options)
    except Exception as exc:  # noqa: BLE001
        print(f"[hw] client-side Sampler unavailable ({exc!r}) -> legacy SamplerV2")
        from qiskit_ibm_runtime import SamplerV2

        return SamplerV2(mode=mode, options=options)


def pec_with_noise_learning(estimator, pubs, backend):
    """Client-side PEC/PEA needs explicit layer-noise learning (runtime >= 0.50 docs, executor_estimator):
    layers = est.find_unique_layers(pubs); NoiseLearnerV3(mode=backend).run(layers).result()
    .to_pauli_lindblad_maps(); est.options.resilience.layer_noise_model = zip(layers, maps)."""
    from qiskit_ibm_runtime.noise_learner_v3 import NoiseLearnerV3

    layers = estimator.find_unique_layers(pubs)
    nl = NoiseLearnerV3(mode=backend).run(layers).result()
    estimator.options.resilience.layer_noise_model = list(zip(layers, nl.to_pauli_lindblad_maps()))
    return estimator


# ----------------------------------------------------------------------------- jobs
def run_estimator(backend, qc_logical, H_spo, level: str = "zne") -> dict:
    from circuits_qiskit import gate_report, transpile_isa

    isa = transpile_isa(qc_logical, backend, optimization_level=3)
    est = make_estimator(backend, estimator_options(level))
    pubs = [(isa, H_spo.apply_layout(isa.layout))]
    if level in ("pec", "pea") and hasattr(est, "find_unique_layers"):
        est = pec_with_noise_learning(est, pubs, backend)
    job = est.run(pubs)
    print(f"[hw] estimator job {job.job_id()} submitted on {backend.name}")
    res = job.result()[0]
    return {"job_id": job.job_id(), "backend": backend.name, "level": level, "gates": gate_report(isa),
            "evs": float(res.data.evs), "stds": float(res.data.stds)}


def run_sampler(backend, qc_logical_measured, shots: int = 100_000):
    """Returns (BitArray, job_id). Circuit must contain measure_all() -> data.meas."""
    from circuits_qiskit import transpile_isa

    return run_sampler_isa(backend, transpile_isa(qc_logical_measured, backend, optimization_level=3), shots)


def run_sampler_isa(backend, isa, shots: int = 100_000):
    sampler = make_sampler(backend, sampler_options())
    job = sampler.run([isa], shots=shots)
    print(f"[hw] sampler job {job.job_id()} submitted on {backend.name}")
    return job.result()[0].data.meas, job.job_id()


def plan(args) -> dict:
    return {
        "channel": CHANNEL,
        "credentials": "QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE env vars or saved account",
        "backend": args.backend or "least_busy(operational=True, simulator=False)",
        "steps": [
            f"chemistry: N2 {args.basis} R={args.r} active space {args.cas} (UNtapered for SQD)",
            f"ansatz: {args.ansatz} ({'ffsim LUCJ from active-space CCSD, n_reps=1' if args.ansatz == 'lucj' else f'first k={args.k} ADAPT ops'})",
            f"Sampler: {args.shots} shots -> qiskit-addon-sqd capped at subspace fraction <= {args.max_frac}",
            "baselines through the SAME capped SQD call: uniform random bitstrings (same shots), "
            "CISD-ranked determinants (same per-spin size)",
            "Estimator (ADAPT ansatz only) with options "
            + json.dumps(estimator_options(args.level)),
        ],
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="IBM hardware runs (dry run without credentials)")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--cas", default="fv108",
                    help="fv108 (default, 16 qubits, 3136 dets) or e.g. cas10e16o with --basis cc-pvdz (32 qubits)")
    ap.add_argument("--ansatz", default="lucj", choices=["lucj", "adapt"])
    ap.add_argument("--r", type=float, default=2.1)
    ap.add_argument("--k", type=int, default=4, help="number of ADAPT operators (ansatz=adapt)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--level", default="zne", choices=["trex", "zne", "pea", "pec"])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--max-frac", type=float, default=0.05, help="cap on SQD subspace / full space")
    ap.add_argument("--samples-per-batch", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cas == "cas66":
        print("[hw] WARNING: CAS(6e,6o) has only 400 determinants; any noisy device saturates it. "
              "Use --cas fv108 or larger.")

    service = None if args.dry_run else get_service()
    if service is None:
        print("[hw] DRY RUN -- plan:")
        print(json.dumps(plan(args), indent=1))
        return

    from run_all import build_problem
    from sqd_ibm import cisd_ranked_baseline, honest_report, max_dim_for_fraction, random_baseline, run_sqd

    P = build_problem(args.r, args.basis, args.cas)
    backend = pick_backend(service, args.backend, min_qubits=2 * P.n_orb)
    max_dim = max_dim_for_fraction(P, args.max_frac)
    out = {"plan": plan(args), "max_dim_per_spin": max_dim}
    e_ref = None
    pr = res = None
    if P.n_orb <= 8:
        from src.pipeline import prepare, run_adapt

        pr = prepare(P)
        e_ref = pr.e_exact
        out["exact_singlet"] = e_ref
    if args.ansatz == "adapt" and pr is None:
        raise SystemExit("ansatz=adapt needs <= 8 active orbitals (exact diag + ADAPT); use --ansatz lucj")
    if args.ansatz == "lucj":
        from lucj_ffsim import lucj_circuit

        got = lucj_circuit(P, n_reps=1, backend=backend)
        if got is None:
            raise SystemExit("ffsim missing")
        _, isa, P_sqd, pairs = got
        bits, jid = run_sampler_isa(backend, isa, args.shots)
        out["pairs"] = pairs
    else:
        from circuits_qiskit import circuit_from_adapt

        res = run_adapt(pr, max_ops=max(args.k, 1))
        P_sqd = P
        bits, jid = run_sampler(backend, circuit_from_adapt(pr, res, args.k, tapered=False, measure=True), args.shots)
    kw = dict(samples_per_batch=args.samples_per_batch, max_dim=max_dim)
    out["sqd"] = run_sqd(P_sqd, bits, **kw)
    out["sqd"]["job_id"] = jid
    out["random"] = random_baseline(P_sqd, int(bits.num_shots), **kw)
    try:
        out["cisd_ranked"] = cisd_ranked_baseline(P_sqd, max_dim)
    except Exception as exc:  # noqa: BLE001
        out["cisd_ranked"] = {"status": f"failed: {exc!r}"}
    print("\n[hw] HONEST SQD SUMMARY (same cap for all):")
    for name in ("sqd", "random", "cisd_ranked"):
        if "energy" in out[name]:
            honest_report(name, out[name], e_ref)
    if args.ansatz == "adapt":
        from circuits_qiskit import circuit_from_adapt
        from qubit_nature import our_to_sparse_pauli

        qc_t = circuit_from_adapt(pr, res, args.k, tapered=True)
        out["estimator"] = run_estimator(backend, qc_t, our_to_sparse_pauli(pr.Ht, pr.tp.n_tapered), args.level)
    dest = _ROOT / "results" / "port"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"hardware_{backend.name}_{args.cas}_R{args.r}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "plan"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
