# Production port. Written against qiskit-addon-sqd 0.14.0 (source tag 0.14.0;
# https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/fermion) and Qiskit SDK 2.6.1 BitArray
# docs on 2026-10-27. Not executed in the authoring environment (no package access); run
# port/test_port.py on a laptop first.
"""Sample-based quantum diagonalization with IBM's qiskit-addon-sqd on our sampled bitstrings.

CURRENT API (0.12 -> 0.14): the single entry point is
    qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian(
        one_body_tensor, two_body_tensor, bit_array, samples_per_batch, norb, nelec, *,
        num_batches=1, energy_tol=1e-8, occupancies_tol=1e-5, max_iterations=100,
        sci_solver=None, [policy=None  (0.14+)], symmetrize_spin=False, max_dim=None,
        include_configurations=None, initial_occupancies=None, carryover_threshold=...,
        callback=None, seed=None) -> SCIResult
with ``sci_solver = functools.partial(solve_sci_batch, spin_sq=0.0, max_cycle=200)``.
``SCIResult.energy`` is the ELECTRONIC energy: add ``Problem.ecore`` (frozen core + nuclear).
``solve_fermion`` / ``postselect_and_subsample`` / ``recover_configurations`` are the older
building blocks (the latter two deprecated in 0.12); we do not use them.
``bit_array`` is a qiskit.primitives.BitArray (0.14 also accepts a 2-D bool array; we always pass a
BitArray so 0.12/0.13 work too).

BITSTRING ORDERING (verified, all consistent -- no reversal needed):
  * addon docstring: "alpha part concatenated on the right-hand side: [b_N, ..., b_0, a_N, ..., a_0]".
  * Qiskit bitstrings are little-endian printed: clbit 0 is the RIGHTMOST character, and
    ``measure_all`` measures qubit q into clbit q; ``BitArray.from_samples(ints, num_bits)`` takes
    integers whose bit q is clbit q.
  * Our block ordering puts alpha spin-orbital p on qubit p (p < n) and beta p on qubit n+p, i.e.
    our full integer bit q = spin orbital q -> printed string = [b_{n-1}..b_0 a_{n-1}..a_0]. Exactly
    the layout the addon expects (and ffsim's, which puts alpha on the first norb qubits).
  * Within a half, orbital 0 is the least-significant bit = PySCF CI-string convention.

HONEST PROTOCOL (panel): always cap the subspace (max_dim) so subspace/full << 1, print the subspace
fraction with every energy, and compare against (i) uniformly random bitstrings through the SAME SQD
call (same shots, same cap) and (ii) a classical CISD-ranked determinant selection of the same size.

TAPERED SAMPLES: a 7-qubit tapered circuit is measured in the Z basis on the kept qubits. Kept-qubit
Z operators commute with the tapering Clifford U = prod (X_q + Z^s)/sqrt(2), so those outcomes are
exactly the kept bits of the 12-qubit determinant; each dropped pivot bit q is fixed by the symmetry
parity <Z^s> = eigenvalue. ``src.sqd.untaper_bits`` implements this; ``untaper_bit_array`` wraps it.
Alternative: build the circuit on the UNtapered 12 qubits (``circuit_from_adapt(..., tapered=False)``)
and feed its BitArray directly (more 2q gates, no reconstruction step).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sqd import untaper_bits  # noqa: E402


# ----------------------------------------------------------------------------- bit plumbing
def bit_array_to_ints(bit_array) -> np.ndarray:
    """BitArray (one shot per entry) -> int64 samples, bit q = clbit q (= qubit q for measure_all).
    Uses BitArray.get_int_counts() (https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.primitives.BitArray)."""
    counts = bit_array.get_int_counts()
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    reps = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    return np.repeat(keys, reps)


def ints_to_bit_array(samples: np.ndarray, num_bits: int):
    """int samples -> BitArray.from_samples(samples, num_bits) (bit q -> clbit q)."""
    from qiskit.primitives import BitArray

    return BitArray.from_samples([int(s) for s in samples], num_bits=num_bits)


def untaper_bit_array(tp, bit_array_tapered):
    """7-qubit tapered BitArray -> 12-qubit determinant BitArray (see module docstring)."""
    full = untaper_bits(tp, bit_array_to_ints(bit_array_tapered))
    return ints_to_bit_array(full, tp.n)


def hf_occupancies(P) -> tuple[np.ndarray, np.ndarray]:
    """Hartree-Fock (alpha, beta) orbital occupancies used as ``initial_occupancies``."""
    a = np.zeros(P.n_orb)
    b = np.zeros(P.n_orb)
    a[: P.n_alpha] = 1.0
    b[: P.n_beta] = 1.0
    return a, b


# ----------------------------------------------------------------------------- SQD (honest protocol)
def full_dim(P) -> int:
    """Dimension of the (n_alpha, n_beta) determinant space = C(norb, na) * C(norb, nb)."""
    from math import comb

    return comb(P.n_orb, P.n_alpha) * comb(P.n_orb, P.n_beta)


def max_dim_for_fraction(P, max_frac: float = 0.05) -> int:
    """Per-spin cap so that (max_dim)^2 / full_dim <= max_frac (symmetrize_spin -> one shared cap)."""
    from math import comb

    per_spin = min(comb(P.n_orb, P.n_alpha), comb(P.n_orb, P.n_beta))
    return max(1, min(per_spin, int(np.floor(np.sqrt(max_frac * full_dim(P))))))


def run_sqd(P, bit_array, samples_per_batch: int = 300, num_batches: int = 5, max_iterations: int = 10,
            max_dim: int | None = None, max_frac: float = 0.05, spin_sq: float = 0.0, seed: int = 24,
            verbose: bool = True, tag: str = "sqd") -> dict:
    """qiskit-addon-sqd configuration recovery + SCI on our Problem's integrals, ALWAYS capped.

    The subspace is capped with ``max_dim`` (per spin sector; default from ``max_frac`` so that
    dim/full_dim <= max_frac) -- otherwise a noisy device simply saturates a small space and a
    'success' is meaningless. Every printed energy carries the subspace fraction.
    ``bit_array`` must be the UNtapered 2*n_orb-bit BitArray in our block ordering.
    https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/fermion#diagonalize_fermionic_hamiltonian
    """
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

    if max_dim is None:
        max_dim = max_dim_for_fraction(P, max_frac)
    D = full_dim(P)
    history = []

    def callback(results):  # list[SCIResult], one per batch
        best = min(results, key=lambda r: r.energy)
        dim = int(np.prod(best.sci_state.amplitudes.shape)) if best.sci_state is not None else None
        history.append({"energy": float(best.energy + P.ecore), "subspace_dim": dim,
                        "subspace_fraction": None if dim is None else dim / D})
        if verbose:
            frac = "n/a" if dim is None else f"{dim / D:.2e}"
            print(f"   [{tag}] iter {len(history)}: E = {best.energy + P.ecore:.8f}  dim = {dim}/{D} "
                  f"(fraction {frac})", flush=True)

    sci_solver = partial(solve_sci_batch, spin_sq=spin_sq, max_cycle=200)
    res = diagonalize_fermionic_hamiltonian(
        np.asarray(P.h), np.asarray(P.g), bit_array, samples_per_batch=samples_per_batch, norb=P.n_orb,
        nelec=(P.n_alpha, P.n_beta), num_batches=num_batches, max_iterations=max_iterations,
        sci_solver=sci_solver, symmetrize_spin=(P.n_alpha == P.n_beta), max_dim=max_dim,
        initial_occupancies=hf_occupancies(P), callback=callback, seed=np.random.default_rng(seed))
    dim = int(np.prod(res.sci_state.amplitudes.shape)) if res.sci_state is not None else None
    out = {"energy": float(res.energy + P.ecore), "subspace_dim": dim, "full_dim": D,
           "subspace_fraction": None if dim is None else dim / D, "max_dim_per_spin": max_dim,
           "history": history, "num_shots": int(bit_array.num_shots),
           "samples_per_batch": samples_per_batch, "num_batches": num_batches}
    if verbose:
        frac = "n/a" if dim is None else f"{dim / D:.2e}"
        print(f"   [{tag}] final E = {out['energy']:.8f}  subspace {dim}/{D} = {frac}")
    return out


def random_baseline(P, num_shots: int, seed: int = 99, **sqd_kwargs) -> dict:
    """Same SQD call on UNIFORMLY RANDOM bitstrings (same shot count, same cap). If the device run is
    not clearly better than this, the quantum samples carried no information."""
    from qiskit_addon_sqd.counts import generate_bit_array_uniform

    bits = generate_bit_array_uniform(num_shots, 2 * P.n_orb, rand_seed=seed)
    return run_sqd(P, bits, tag="random", **sqd_kwargs)


def cisd_ranked_baseline(P, max_dim: int, spin_sq: float = 0.0) -> dict:
    """Classical baseline at the SAME per-spin size: CISD on the active-space model Hamiltonian
    (PySCF ci.CISD + to_fcivec), rank alpha / beta strings by marginal CISD weight, keep the top
    ``max_dim`` of each (shared list, singlet), diagonalise with qiskit_addon_sqd.fermion.solve_sci.
    Works in the model-RHF basis (chem_pyscf.rotate_problem), as CISD's determinants live there."""
    from pyscf import ci
    from pyscf.fci import cistring
    from qiskit_addon_sqd.fermion import solve_sci

    from chem_pyscf import model_mf, rotate_problem

    mf = model_mf(P)
    P_mo = rotate_problem(P, mf.mo_coeff)
    myci = ci.CISD(mf)
    myci.kernel()
    nelec = (P.n_alpha, P.n_beta)
    fcivec = myci.to_fcivec(myci.ci, P.n_orb, nelec)
    w_a = (np.abs(fcivec) ** 2).sum(axis=1)
    w_b = (np.abs(fcivec) ** 2).sum(axis=0)
    sa = cistring.make_strings(range(P.n_orb), P.n_alpha)
    sb = cistring.make_strings(range(P.n_orb), P.n_beta)
    weight = {}
    for s_, w in list(zip(sa, w_a)) + list(zip(sb, w_b)):
        weight[int(s_)] = weight.get(int(s_), 0.0) + float(w)
    keep = np.array(sorted(sorted(weight, key=lambda k: -weight[k])[:max_dim]), dtype=np.int64)
    r = solve_sci((keep, keep), np.asarray(P_mo.h), np.asarray(P_mo.g), norb=P.n_orb, nelec=nelec, spin_sq=spin_sq)
    dim = len(keep) ** 2
    return {"energy": float(r.energy + P_mo.ecore), "subspace_dim": dim, "full_dim": full_dim(P),
            "subspace_fraction": dim / full_dim(P), "e_cisd": float(myci.e_tot)}


def honest_report(name: str, out: dict, e_ref: float | None):
    err = "" if e_ref is None else f"  err {1e3 * (out['energy'] - e_ref):8.3f} mHa"
    frac = out.get("subspace_fraction")
    frac = "n/a" if frac is None else f"{frac:.2e}"
    print(f"   {name:<14} E = {out['energy']:.8f}{err}  subspace {out['subspace_dim']}/{out['full_dim']} = {frac}")


def run_sqd_teaching(P, full_ints: np.ndarray) -> dict:
    """Fallback when qiskit-addon-sqd is missing: our teaching SQD (src/sqd.py)."""
    from src.pauli import qubit_hamiltonian, to_sparse
    from src.sqd import sqd_energy

    Hs = to_sparse(qubit_hamiltonian(P.h, P.g, P.ecore), 2 * P.n_orb).tocsr()
    e, dim, fg = sqd_energy(Hs, np.asarray(full_ints, dtype=np.int64), P.n_orb, P.n_alpha, P.n_beta)
    return {"energy": e, "subspace_dim": dim, "frac_good": fg, "solver": "src.sqd (teaching)"}
