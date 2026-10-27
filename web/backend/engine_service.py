"""Compute services behind the web API: every number comes from the verified engine in src/.

Each public `job_*` function takes a context (progress / log / partial results / cancellation) and
validated parameters, and returns a JSON-serialisable dict. Expensive intermediate objects
(SCF problems, tapered Hamiltonians, ADAPT runs) are cached in memory and shared between jobs.
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ccsd import ccsd_t  # noqa: E402
from src.n2 import n2_problems  # noqa: E402
from src.noise import DensitySim, Estimator, NoiseModel, count_gates, pec_overhead, zne  # noqa: E402
from src.pauli import exact_ground, hf_bitstring, op_add, op_scale, popcount, s_squared, sector_indices, to_sparse  # noqa: E402
from src.pipeline import EQ, SPIN_PENALTY, circuit_for, first_chem_acc, prepare, run_adapt, untapered_cx  # noqa: E402
from src.sqd import sample, untaper_bits  # noqa: E402

CHEM = 1.6e-3
SPACES = {"cas66": "CAS(6e,6o) · 12 → 7 qubits", "fv108": "FV(10e,8o) · 16 → 11 qubits"}
SCALES = [1, 3, 5]
R_MIN, R_MAX = 0.80, 3.20
STEP = 0.15  # max bond-length step when following the symmetric RHF branch
GRID = [0.90, 1.00, 1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70, 3.00]  # published study geometries


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------------------------
class _LRU:
    def __init__(self, size):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.size:
                self.d.popitem(last=False)

    def __len__(self):
        return len(self.d)


_branch = {}                 # r -> (problems, C) along the symmetric RHF branch (small, kept)
_branch_lock = threading.Lock()
_prepared = _LRU(16)         # (r, space) -> Prepared
_adapt = _LRU(32)            # (r, space) -> (res, max_ops)
_sparse = _LRU(8)            # (r, space) -> full-space sparse H (for SQD)
_key_locks: dict = {}
_key_locks_lock = threading.Lock()


def _lock_for(key):
    with _key_locks_lock:
        return _key_locks.setdefault(key, threading.Lock())


def cache_stats():
    return {"geometries_solved": len(_branch), "prepared": len(_prepared), "adapt_runs": len(_adapt)}


def norm_r(r: float) -> float:
    return round(float(r), 3)


def problems(r: float, ctx=None):
    """SCF + active-space integrals at bond length r, following the symmetric RHF branch from
    equilibrium in steps of <= STEP so every geometry lands on the same (D2h-symmetric) solution."""
    r = norm_r(r)
    with _branch_lock:
        if r in _branch:
            return _branch[r][0]
        if EQ not in _branch:
            if ctx:
                ctx.log(f"SCF at equilibrium {EQ} Å (branch anchor)")
            _branch[EQ] = n2_problems(EQ)
        r0 = min((x for x in _branch if x in GRID), key=lambda x: abs(x - r))
        # walk through the reference grid (same path as the published study, so results at grid
        # points are bit-for-bit identical), never stepping more than STEP between grid points
        lo, hi = sorted((r0, r))
        stops = sorted({g for g in GRID if lo < g < hi} | {r}, reverse=r < r0)
        path, prev = [], r0
        for s in stops:
            n_steps = 1
            path += [round(prev + (s - prev) * (i + 1) / n_steps, 4) for i in range(n_steps)]
            prev = s
        C = _branch[r0][1]
        for x in path:
            if ctx:
                ctx.check()
                ctx.log(f"SCF at {x:.3f} Å (guess from {r0:.3f} Å)")
            key = norm_r(x)
            if key not in _branch:
                _branch[key] = n2_problems(x, C)
            C, r0 = _branch[key][1], x
        return _branch[r][0]


def warm_branch(log=print):
    """Solve SCF along the whole reference grid once (about a minute), so later jobs start instantly."""
    t = time.time()
    for g in GRID:
        problems(g)
    log(f"SCF branch warmed: {len(GRID)} geometries in {time.time() - t:.0f} s")


def prepared(r: float, space: str, ctx=None):
    key = (norm_r(r), space)
    with _lock_for(("prep",) + key):
        pr = _prepared.get(key)
        if pr is None:
            P = problems(r, ctx)[space]
            if ctx:
                ctx.log(f"Jordan–Wigner + tapering + exact diagonalisation ({SPACES[space]})")
            pr = prepare(P)
            _prepared.put(key, pr)
        return pr


def adapt_result(r: float, space: str, max_ops: int, ctx=None, on_iter=None):
    """ADAPT-VQE (singlet-pinned). A cached run with >= max_ops operators (or one that converged
    earlier) is reused: ADAPT is deterministic, so a shorter run is a prefix of a longer one."""
    key = (norm_r(r), space)
    with _lock_for(("adapt",) + key):
        hit = _adapt.get(key)
        if hit and (hit[1] >= max_ops or len(hit[0].ops) < hit[1]):
            return hit[0], True
        pr = prepared(r, space, ctx)

        def cb(it, e_pen, label, grad):
            if ctx:
                ctx.check()
            if on_iter:
                on_iter(it, e_pen, label, grad, pr)

        res = run_adapt(pr, max_ops=max_ops, callback=cb)
        _adapt.put(key, (res, max_ops))
        return res, False


def full_sparse(r, space, ctx=None):
    key = (norm_r(r), space)
    Hs = _sparse.get(key)
    if Hs is None:
        pr = prepared(r, space, ctx)
        Hs = to_sparse(pr.H, 2 * pr.P.n_orb).tocsr()
        _sparse.put(key, Hs)
    return Hs


def mha(e, ref):
    return None if e is None or not np.isfinite(e) else 1e3 * (e - ref)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def bits_to_occ(det: int, n_orb: int) -> dict:
    return {"alpha": [(det >> q) & 1 for q in range(n_orb)], "beta": [(det >> (q + n_orb)) & 1 for q in range(n_orb)]}


def wf_summary(pr, top=8):
    P = pr.P
    nq = 2 * P.n_orb
    idx = sector_indices(nq, P.n_alpha, P.n_beta, P.n_orb)
    Hpen = op_add(pr.H, op_scale(s_squared(P.n_orb), SPIN_PENALTY))
    _, psi = exact_ground(to_sparse(Hpen, nq), idx)
    w = np.abs(psi) ** 2
    hf = hf_bitstring(P.n_orb, P.n_alpha, P.n_beta)
    order = np.argsort(-w)[:top]
    occ = [float(sum(w[b] for b in idx if (b >> q) & 1) + sum(w[b] for b in idx if (b >> (q + P.n_orb)) & 1))
           for q in range(P.n_orb)]
    return {"hf_weight": float(w[hf]), "n_over_1pct": int((w > 0.01).sum()), "sector_dim": int(len(idx)),
            "natural_occupations": [round(o, 5) for o in occ],
            "top": [{"det": int(b), "weight": float(w[b]), **bits_to_occ(int(b), P.n_orb), "is_hf": int(b) == hf}
                    for b in order]}


def gates_payload(gates):
    out = []
    for g in gates:
        if g[0] == "cx":
            out.append(["cx", int(g[1]), int(g[2])])
        else:
            out.append(["1q", int(g[1])])
    return out


# --------------------------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------------------------
def job_energies(ctx, r, space):
    t0 = time.time()
    ctx.progress(0.05, "Hartree–Fock and integrals")
    P = problems(r, ctx)[space]
    ctx.progress(0.35, "Qubit Hamiltonian, tapering, exact answer")
    pr = prepared(r, space, ctx)
    ctx.progress(0.7, "Coupled cluster CCSD(T)")
    cc = ccsd_t(P.h, P.g, P.ecore, P.n_alpha, P.n_beta)
    ctx.progress(0.9, "Wavefunction analysis")
    ex = pr.e_exact
    conv = bool(cc["converged"])
    out = {
        "r": norm_r(r), "space": space, "space_label": SPACES[space],
        "qubits": 2 * P.n_orb, "qubits_tapered": pr.tp.n_tapered,
        "pauli_terms": len(pr.H), "pauli_terms_tapered": len(pr.Ht), "pool_size": len(pr.pool_t),
        "e_rhf_full": P.e_rhf_full, "exact": ex, "exact_lowest_any_spin": pr.e_lowest_any,
        "s2_lowest_any_spin": pr.s2_lowest_any,
        "hf": cc["hf"], "ccsd": cc["ccsd"] if conv else None, "ccsd_t": cc["ccsd_t"] if conv else None,
        "e_t": cc.get("e_t") if conv else None, "cc_converged": conv,
        "err_mHa": {"hf": mha(cc["hf"], ex), "ccsd": mha(cc["ccsd"], ex) if conv else None,
                    "ccsd_t": mha(cc["ccsd_t"], ex) if conv else None},
        "wavefunction": wf_summary(pr),
        "seconds": round(time.time() - t0, 2),
    }
    if conv and out["err_mHa"]["ccsd_t"] < -CHEM * 1e3:
        out["cc_verdict"] = "collapsed below the exact energy"
    elif not conv:
        out["cc_verdict"] = "did not converge"
    elif abs(out["err_mHa"]["ccsd_t"]) <= CHEM * 1e3:
        out["cc_verdict"] = "within chemical accuracy"
    else:
        out["cc_verdict"] = "outside chemical accuracy"
    return out


def job_adapt(ctx, r, space, max_ops):
    t0 = time.time()
    ctx.progress(0.02, "Preparing Hamiltonian")
    pr = prepared(r, space, ctx)
    ex = pr.e_exact
    trace = []

    def on_iter(it, e_pen, label, grad, _pr):
        trace.append({"iter": it, "err_mHa": 1e3 * (e_pen - ex), "op": label, "grad": grad})
        ctx.partial({"exact": ex, "trace": trace[-200:]})
        ctx.progress(min(0.95, 0.05 + 0.9 * it / max_ops), f"ADAPT iteration {it}: operator {label}")

    res, cached = adapt_result(r, space, max_ops, ctx, on_iter)
    n_ops = min(len(res.ops), max_ops)
    energies = res.energies[:n_ops + 1]
    k = first_chem_acc(res, ex)
    k = k if k is not None and k <= n_ops else None
    k_half = first_chem_acc(res, ex, thr=0.8e-3)
    k_half = k_half if k_half is not None and k_half <= n_ops else None
    circ = None
    if k:
        g = circuit_for(pr, res, k)
        circ = {**count_gates(g), "cx_untapered": untapered_cx(pr, res, k), "ops": k}
    ctx.progress(0.98, "Compiling circuit")
    return {
        "r": norm_r(r), "space": space, "space_label": SPACES[space], "exact": ex, "hf": res.energies[0],
        "qubits_tapered": pr.tp.n_tapered, "pool_size": len(pr.pool_t), "max_ops": max_ops,
        "iterations": [{"k": i, "energy": e, "err_mHa": 1e3 * (e - ex), "s2": res.s2[i],
                        "op": (res.labels[i - 1] if i > 0 else "HF"),
                        "max_grad": (res.max_grads[i - 1] if 0 < i <= len(res.max_grads) else None)}
                       for i, e in enumerate(energies)],
        "final_err_mHa": 1e3 * (energies[-1] - ex), "k_chem_acc": k, "k_half": k_half, "circuit_at_chem_acc": circ,
        "converged": n_ops < max_ops, "reused_cached_run": cached, "seconds": round(time.time() - t0, 2),
    }


def _depth(res, ex, depth):
    if depth == "half":
        k = first_chem_acc(res, ex, thr=0.8e-3)
    elif depth == "chem":
        k = first_chem_acc(res, ex)
    else:
        k = min(int(depth), len(res.ops))
    return k


def _mitigated(sim, est, gates, ref, nm):
    rhos = {s: sim.run(gates, ref, nm, fold=s) for s in SCALES}
    rem = [est.energy(rhos[s], p_ro=nm.p_ro, rem=True) for s in SCALES]
    vals = {"raw": est.energy(rhos[1], p_ro=nm.p_ro), "rem": rem[0],
            "rem+zne_lin": zne(SCALES[:2], rem[:2], "linear"), "rem+zne_rich": zne(SCALES, rem, "richardson"),
            "rem+zne_exp": zne(SCALES, rem, "exponential"),
            "rem+zne_exp_asym": zne(SCALES, rem, "exponential_asym", asymptote=est.const)}
    return vals, rhos[1]


def _noise_setup(ctx, r, space, depth):
    pr = prepared(r, space, ctx)
    ctx.progress(0.1, "ADAPT circuit (cached after first run)")
    res, _ = adapt_result(r, space, 80 if space == "cas66" else 120, ctx)
    k = _depth(res, pr.e_exact, depth)
    if not k:
        raise ValueError("ADAPT never reached that accuracy; choose a fixed number of bricks")
    gates = circuit_for(pr, res, k)
    n = pr.tp.n_tapered
    return pr, res, k, gates, DensitySim(n), Estimator(pr.Ht, n)


def job_noise(ctx, r, space, depth, r2, r1, p_ro):
    t0 = time.time()
    pr, res, k, gates, sim, est = _noise_setup(ctx, r, space, depth)
    ex = pr.e_exact
    ctx.progress(0.3, f"Density-matrix simulation of {count_gates(gates)['cx']} CNOTs at noise ×1, ×3, ×5")
    ideal = est.energy(sim.run(gates, pr.tp.ref_bits, None))
    nm = NoiseModel(r2=r2, r1=r1 if r1 is not None else r2 / 12, p_ro=p_ro)
    vals, rho1 = _mitigated(sim, est, gates, pr.tp.ref_bits, nm)
    gamma = pec_overhead(gates, nm)
    ev = est.pauli_expectations(rho1)
    sigma1 = float(np.sqrt(np.sum(est.coefs ** 2 * np.clip(1 - ev ** 2, 0, None))))
    budget = CHEM - abs(ideal - ex)
    best = min((abs(1e3 * (v - ex)), kname) for kname, v in vals.items() if np.isfinite(v))
    return {
        "r": norm_r(r), "space": space, "k": k, "depth": depth, "gates": count_gates(gates), "exact": ex,
        "noise_model": vars(nm), "noiseless_err_mHa": 1e3 * (ideal - ex),
        "ablation": {kname: {"energy": v if np.isfinite(v) else None, "err_mHa": mha(v, ex)} for kname, v in vals.items()},
        "best": {"method": best[1], "abs_err_mHa": best[0], "within_target": best[0] <= CHEM * 1e3},
        "pec": {"gamma": gamma, "shots_per_term": (gamma * sigma1 / budget) ** 2 if budget > 0 else None,
                "shots_per_term_unmitigated": (sigma1 / budget) ** 2 if budget > 0 else None},
        "seconds": round(time.time() - t0, 2),
    }


P_PHYS, P_TH, EPS_ROT = 1e-3, 1e-2, 1e-4


def job_noise_sweep(ctx, r, space, depth, r2_list, p_ro):
    t0 = time.time()
    pr, res, k, gates, sim, est = _noise_setup(ctx, r, space, depth)
    ex = pr.e_exact
    ideal = est.energy(sim.run(gates, pr.tp.ref_bits, None))
    rows = []
    for i, r2 in enumerate(sorted(r2_list)):
        ctx.check()
        ctx.progress(0.15 + 0.8 * i / len(r2_list), f"Gate error {r2:.0e}")
        vals, _ = _mitigated(sim, est, gates, pr.tp.ref_bits, NoiseModel(r2=r2, r1=r2 / 12, p_ro=p_ro))
        row = {"r2": r2, **{kname: mha(v, ex) for kname, v in vals.items()},
               "pec_gamma": pec_overhead(gates, NoiseModel(r2=r2, r1=r2 / 12, p_ro=p_ro))}
        row["best_zne"] = min((abs(row[kn]) for kn in ("rem+zne_lin", "rem+zne_rich", "rem+zne_exp") if row[kn] is not None),
                              default=None)
        rows.append(row)
        ctx.partial({"sweep": rows})
    ok = [row["r2"] for row in rows if row["best_zne"] is not None and row["best_zne"] <= CHEM * 1e3]
    # fault-tolerant estimate (same assumptions as scripts/run_ft_estimate.py)
    ansatz = abs(1e3 * (ideal - ex))
    s0 = rows[0]
    slope = (abs(s0["rem"]) - ansatz) / s0["r2"] if s0["rem"] is not None else None
    ft = None
    if slope and slope > 0 and 1.6 - ansatz > 0:
        target = (1.6 - ansatz) / slope
        d = next((d for d in range(3, 61, 2) if 3 * d * 0.1 * (P_PHYS / P_TH) ** ((d + 1) / 2) <= target), None)
        n_rz = sum(1 for g in gates if g[0] == "1q" and abs(g[2][0, 1]) < 1e-12 and abs(g[2][1, 0]) < 1e-12)
        n_log = pr.tp.n_tapered
        ft = {"logical_qubits": n_log, "logical_r2_target": target, "code_distance": d,
              "physical_qubits_data_and_routing": n_log * 2 * d * d * 2 if d else None,
              "t_gates": n_rz * math.ceil(3 * math.log2(1 / EPS_ROT)),
              "assumptions": "p=1e-3, p_th=1e-2, per-gate 3·d·p_L, 2d² per patch ×2 routing, factories excluded"}
    return {"r": norm_r(r), "space": space, "k": k, "gates": count_gates(gates), "exact": ex,
            "noiseless_err_mHa": 1e3 * (ideal - ex), "p_ro": p_ro, "sweep": rows,
            "max_r2_within_target": max(ok) if ok else None, "fault_tolerant": ft,
            "seconds": round(time.time() - t0, 2)}


def _recover(strings, n_orb, n_el, occ):
    out = []
    for s in strings:
        s = int(s)
        bits = [(s >> q) & 1 for q in range(n_orb)]
        cnt = sum(bits)
        while cnt != n_el:
            if cnt > n_el:
                q = min((q for q in range(n_orb) if bits[q]), key=lambda q: occ[q])
                bits[q], cnt = 0, cnt - 1
            else:
                q = max((q for q in range(n_orb) if not bits[q]), key=lambda q: occ[q])
                bits[q], cnt = 1, cnt + 1
        out.append(sum(b << q for q, b in enumerate(bits)))
    return out


def rank_strings(full_bits, n_orb, n_a, n_b):
    """Orbital (alpha/beta) strings ranked by how often they appear after configuration recovery."""
    amask = (1 << n_orb) - 1
    A = [int(v) & amask for v in full_bits]
    B = [(int(v) >> n_orb) & amask for v in full_bits]
    good = [popcount(a) == n_a and popcount(b) == n_b for a, b in zip(A, B)]
    ng = sum(good)
    if ng:
        occ_a = [np.mean([(a >> q) & 1 for a, g in zip(A, good) if g]) for q in range(n_orb)]
        occ_b = [np.mean([(b >> q) & 1 for b, g in zip(B, good) if g]) for q in range(n_orb)]
        A, B = _recover(A, n_orb, n_a, occ_a), _recover(B, n_orb, n_b, occ_b)
    counts: dict = {}
    for s in A + B:
        counts[s] = counts.get(s, 0) + 1
    ranked = sorted(counts, key=lambda s: (-counts[s], s))
    return ranked, counts, (ng / len(full_bits) if len(full_bits) else 0.0)


def subspace_energy(Hs, strings, n_orb):
    dets = np.array(sorted(a | (b << n_orb) for a in strings for b in strings), dtype=np.int64)
    sub = Hs[dets][:, dets]
    if len(dets) <= 900:
        return float(np.linalg.eigvalsh(sub.toarray())[0]), len(dets)
    from scipy.sparse.linalg import eigsh
    return float(eigsh(sub.tocsc(), k=1, which="SA", tol=1e-10)[0][0]), len(dets)


def job_sqd(ctx, r, space, bricks, r2, p_ro, shots, seed, budgets):
    t0 = time.time()
    pr = prepared(r, space, ctx)
    P, ex = pr.P, pr.e_exact
    ctx.progress(0.08, f"ADAPT circuit with {bricks} bricks")
    res, _ = adapt_result(r, space, bricks, ctx)
    k = min(bricks, len(res.ops))
    gates = circuit_for(pr, res, k)
    n = pr.tp.n_tapered
    ctx.progress(0.15, f"Noisy density-matrix simulation ({n} qubits, {count_gates(gates)['cx']} CNOTs)")
    sim = DensitySim(n)
    nm = NoiseModel(r2=r2, r1=r2 / 12, p_ro=p_ro)
    rho = sim.run(gates, pr.tp.ref_bits, nm)
    est = Estimator(pr.Ht, n)
    e_vqe = est.energy(rho, p_ro=p_ro, rem=True)
    rng = np.random.default_rng(seed)
    ctx.progress(0.3, f"Sampling {shots:,} shots, plus {shots:,} random bitstrings")
    chip = untaper_bits(pr.tp, sample(rho, shots, n, p_ro, rng))
    rnd = untaper_bits(pr.tp, rng.integers(0, 1 << n, size=shots, dtype=np.int64))
    ranked_chip, counts_chip, fg_chip = rank_strings(chip, P.n_orb, P.n_alpha, P.n_beta)
    ranked_rnd, _, fg_rnd = rank_strings(rnd, P.n_orb, P.n_alpha, P.n_beta)
    hf = (1 << P.n_alpha) - 1
    all_strings = [s for s in range(1 << P.n_orb) if popcount(s) == P.n_alpha]
    ranked_cls = sorted(all_strings, key=lambda s: (popcount(s ^ hf), s))
    n_str = len(all_strings)
    Ks = sorted({int(K) for K in budgets if 1 <= K <= n_str} | {n_str})
    Hs = full_sparse(r, space, ctx)
    rows = []
    for i, K in enumerate(Ks):
        ctx.check()
        ctx.progress(0.35 + 0.6 * i / len(Ks), f"Exact solve in {K * K:,} of {n_str ** 2:,} determinants")
        row = {"K": K, "dets": K * K, "frac": K * K / n_str ** 2}
        for name, ranked in (("chip", ranked_chip), ("random", ranked_rnd), ("classical", ranked_cls)):
            if K <= len(ranked):
                e, _ = subspace_energy(Hs, ranked[:K], P.n_orb)
                row[name] = 1e3 * (e - ex)
            else:
                row[name] = None
        rows.append(row)
        ctx.partial({"rows": rows})
    e_all, dim_all = subspace_energy(Hs, ranked_chip, P.n_orb)
    top = sorted(((int(b), int(c)) for b, c in zip(*np.unique(chip, return_counts=True))), key=lambda t: -t[1])[:12]
    return {
        "r": norm_r(r), "space": space, "space_label": SPACES[space], "exact": ex, "bricks": k, "gates": count_gates(gates),
        "qubits_tapered": n, "noise_model": vars(nm), "shots": shots, "seed": seed,
        "n_strings": n_str, "space_dets": n_str ** 2,
        "distinct_strings": {"chip": len(ranked_chip), "random": len(ranked_rnd)},
        "frac_right_electron_count": {"chip": fg_chip, "random": fg_rnd},
        "noisy_vqe_err_mHa": 1e3 * (e_vqe - ex), "ideal_vqe_err_mHa": 1e3 * (res.energies[k] - ex),
        "uncapped": {"err_mHa": 1e3 * (e_all - ex), "dets": dim_all},
        "budget_rows": rows,
        "top_samples": [{"det": b, "count": c, **bits_to_occ(b, P.n_orb),
                         "right_count": popcount(b & ((1 << P.n_orb) - 1)) == P.n_alpha and popcount(b >> P.n_orb) == P.n_beta}
                        for b, c in top],
        "note": "Each method may solve exactly only inside the K² determinants built from its K highest-ranked "
                "orbital strings. Uncapped, enough random bitstrings eventually cover this small space.",
        "seconds": round(time.time() - t0, 2),
    }


def job_pes_scan(ctx, space, r_list, with_adapt):
    rows = []
    for i, r in enumerate(sorted(r_list)):
        ctx.check()
        ctx.progress(i / len(r_list), f"{r:.3f} Å ({i + 1}/{len(r_list)})")
        sub = _SubCtx(ctx)
        row = job_energies(sub, r, space)
        if with_adapt:
            a = job_adapt(sub, r, space, 80 if space == "cas66" else 120)
            row["adapt"] = {"final_err_mHa": a["final_err_mHa"], "k_chem_acc": a["k_chem_acc"],
                            "cx": (a["circuit_at_chem_acc"] or {}).get("cx")}
        row.pop("wavefunction", None)
        rows.append(row)
        ctx.partial({"rows": rows})
    return {"space": space, "rows": rows}


def job_tests(ctx):
    """Run the regression suites (Python engine, and the browser engine through Node when present)."""
    suites = [("python", f, [sys.executable, "-W", "ignore", str(ROOT / "tests" / f)])
              for f in ("test_integrals_scf.py", "test_fci_ccsd.py", "test_taper_adapt.py", "test_web_api.py")]
    node = _which("node")
    if node and (ROOT / "results" / "lab_data.json").exists():
        suites.append(("browser engine", "lab/test_engine.js", [node, str(ROOT / "lab" / "test_engine.js"),
                                                                 str(ROOT / "results" / "lab_data.json")]))
    out = []
    for i, (kind, name, cmd) in enumerate(suites):
        ctx.check()
        ctx.progress(i / len(suites), f"Running {name}")
        t = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
        passed = sum(l.count("PASS") for l in lines)
        failed = sum(l.count("FAIL") for l in lines)
        out.append({"suite": name, "kind": kind, "exit_code": p.returncode, "pass_marks": passed, "fail_marks": failed,
                    "ok": p.returncode == 0 and failed == 0, "seconds": round(time.time() - t, 1), "output": lines[-60:]})
        ctx.partial({"suites": out})
    return {"suites": out, "all_ok": all(s["ok"] for s in out)}


def _which(name):
    from shutil import which
    return which(name)


class _SubCtx:
    """Context for a nested job: shares logs and cancellation, ignores progress/partials."""

    def __init__(self, parent):
        self.parent = parent

    def progress(self, *_a, **_k):
        pass

    def partial(self, *_a, **_k):
        pass

    def log(self, msg):
        self.parent.log(msg)

    def check(self):
        self.parent.check()


JOBS = {
    "energies": job_energies,
    "adapt": job_adapt,
    "noise": job_noise,
    "noise_sweep": job_noise_sweep,
    "sqd": job_sqd,
    "pes_scan": job_pes_scan,
    "tests": job_tests,
}
