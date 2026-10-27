"""Pre-compute the demo runs into the result cache so the live demo answers instantly.

    python -m web.backend.warm            # standard demo set (~20-40 min on a laptop)
    python -m web.backend.warm --quick    # equilibrium and 2.1 A only (~5 min)
"""
import sys
import time

from .jobs import JobManager

GRID = [0.90, 1.00, 1.098, 1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70, 3.00]


def plan(quick):
    geoms = [1.098, 2.10] if quick else GRID
    jobs = []
    for r in geoms:
        jobs += [("energies", {"r": r, "space": "fv108"}), ("energies", {"r": r, "space": "cas66"}),
                 ("adapt", {"r": r, "space": "cas66"})]
    for r in (1.098, 2.10):
        jobs += [("noise", {"r": r}), ("noise_sweep", {"r": r}), ("sqd", {"r": r, "space": "fv108"}),
                 ("sqd", {"r": r, "space": "cas66"})]
    if not quick:
        jobs += [("adapt", {"r": 1.098, "space": "fv108"}), ("adapt", {"r": 2.10, "space": "fv108"})]
    return jobs


def main():
    quick = "--quick" in sys.argv
    jm = JobManager(workers=1)
    t0 = time.time()
    for i, (t, p) in enumerate(plan(quick), 1):
        j = jm.submit(t, p)
        while j.status in ("queued", "running"):
            time.sleep(0.5)
        print(f"[{i:2d}] {t:12s} {p} -> {j.status}{' (cached)' if j.cached else ''} "
              f"{'' if j.error is None else j.error} [{time.time() - t0:.0f} s]", flush=True)
    jm.pool.shutdown()


if __name__ == "__main__":
    main()
