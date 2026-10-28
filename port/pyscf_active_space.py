# Production port. Written against PySCF 2.14.0 docs (https://pyscf.org/user.html) on 2026-10-27.
# Not executed in the authoring environment (no package access); run port/test_port.py on a laptop first.
"""DEPRECATED shim: the earlier unverified draft lived here. Use ``port/chem_pyscf.py``.

Kept so old imports keep working; everything is re-exported from chem_pyscf.
"""
from chem_pyscf import (  # noqa: F401
    active_space_benchmarks,
    build_mol,
    cas66_casscf,
    dmrg_energy,
    full_space_cc,
    model_mf,
    n2_problem_pyscf,
    problems_along,
    run_rhf,
)

GEOMETRIES = [0.90, 1.00, 1.098, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50]  # Angstrom
