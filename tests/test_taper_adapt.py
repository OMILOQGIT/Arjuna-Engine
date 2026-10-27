"""Tapering + ADAPT-VQE checks on N2 CAS(6e,6o) at equilibrium and stretched."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adapt import Adapt, uccsd_pool  # noqa: E402
from src.n2 import n2_problems  # noqa: E402
from src.pauli import exact_ground, hf_bitstring, qubit_hamiltonian, sector_indices, to_sparse  # noqa: E402
from src.taper import Tapering  # noqa: E402

ok = True
# follow the symmetric RHF branch outward from equilibrium
C = None
probs_at = {}
for r in (1.098, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1):
    probs, C = n2_problems(r, C)
    probs_at[r] = probs
for r in (1.098, 2.1):
    P = probs_at[r]["cas66"]
    nq = 2 * P.n_orb
    H = qubit_hamiltonian(P.h, P.g, P.ecore)
    Hs = to_sparse(H, nq)
    e_ex, _ = exact_ground(Hs, sector_indices(nq, P.n_alpha, P.n_beta, P.n_orb))
    hf = hf_bitstring(P.n_orb, P.n_alpha, P.n_beta)
    tp = Tapering(H, nq, hf)
    Ht = tp.taper(H)
    Hts = to_sparse(Ht, tp.n_tapered)
    e_hf_t = Hts[tp.ref_bits, tp.ref_bits].real
    good = abs(e_hf_t - Hs[hf, hf].real) < 1e-9
    ok &= good
    spec = np.linalg.eigvalsh(Hts.toarray())
    in_spec = np.min(np.abs(spec - e_ex)) < 1e-8
    ok &= in_spec
    print(f"R={r}: {nq} -> {tp.n_tapered} qubits ({len(H)} -> {len(Ht)} terms); "
          f"{'PASS' if good else 'FAIL'} tapered <HF|H|HF>; {'PASS' if in_spec else 'FAIL'} exact E in tapered spectrum")

    pool = uccsd_pool(P.n_orb, P.n_alpha, P.n_beta)
    keep = [(lab, A) for lab, A in pool if tp.commutes_with_symmetries(A)]
    labels = [lab for lab, _ in keep]
    tpool = [tp.taper(A) for _, A in keep]
    t = time.time()
    ad = Adapt(Ht, tpool, tp.n_tapered, tp.ref_bits)
    res = ad.run(e_exact=e_ex, grad_tol=1e-5, max_ops=80, labels=labels)
    err = 1e3 * (res.energies[-1] - e_ex)
    good = err < 1.6
    ok &= good
    first_ca = next((i for i, e in enumerate(res.energies) if 1e3 * (e - e_ex) < 1.6), None)
    print(f"  pool {len(pool)} -> {len(keep)} symmetry-allowed; ADAPT {len(res.ops)} ops, final err {err:.5f} mHa; "
          f"chem. accuracy after {first_ca} ops; {'PASS' if good else 'FAIL'} ({time.time() - t:.1f}s)")

    # untapered ADAPT must give the same energy (tapering is exact)
    ad0 = Adapt(H, [A for _, A in keep], nq, hf)
    res0 = ad0.run(e_exact=e_ex, grad_tol=1e-5, max_ops=80)
    good = abs(res0.energies[-1] - res.energies[-1]) < 1e-6
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} tapered vs untapered ADAPT agree: {res.energies[-1]:.9f} vs {res0.energies[-1]:.9f}")
sys.exit(0 if ok else 1)
