# Arjuna web app

The Arjuna quantum-chemistry engine as a website: a Python backend (Flask + the verified NumPy/SciPy
engine in `src/`) and a browser frontend that runs every calculation live through the API.

## Run it

```bash
./run.sh                 # macOS / Linux: creates .venv, installs numpy, scipy, flask, starts the server
run.bat                  # Windows
# or, with the packages already installed:
python -m web.backend.app
```

Open **http://localhost:8000**. Settings (environment variables): `PORT` (8000), `HOST` (127.0.0.1;
use 0.0.0.0 to serve your network), `WORKERS` (2 parallel jobs), `WARM` (1 = solve the SCF branch at startup).

Docker: `docker build -t arjuna . && docker run -p 8000:8000 arjuna`

## What the site does

| Station | What runs on the engine | Typical time |
|---|---|---|
| 1 Bond explorer | Hartree–Fock at **any** bond length 0.80–3.20 Å, qubit Hamiltonian, tapering, exact answer, CCSD(T), exact wavefunction; range scans | 2–30 s per point |
| 2 ADAPT-VQE | Singlet-pinned ADAPT with live convergence, circuit compilation, CNOT counts with and without tapering | 10–40 s (7 qubits), minutes (11 qubits) |
| 3 Noise lab | Gate-level density-matrix simulation, readout fix, ZNE (all gates folded; linear / Richardson / exponential), PEC cost; gate-error sweep with fault-tolerant estimate | 10 s; sweep ~1 min |
| 4 SQD rescue | Noisy sampling, configuration recovery, fixed-budget exact solves for chip / random / classical picks | 30–60 s |
| 5 Runs | Every job with parameters, status, timing; full JSON | – |
| 6 Proof | Runs all regression suites (Python engine, web API, browser engine) | 3–5 min |

Finished results are cached in `web/backend/cache/`, so a repeated run is instant. The zip ships a
pre-computed demo cache; `python -m web.backend.warm` rebuilds it (`--quick` for a short set).
`/offline-lab` serves the self-contained browser lab, which works with no server at all.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | engine status, versions, cache, running jobs |
| GET | `/api/meta` | active spaces, bond-length range, job types |
| GET | `/api/reference` | published study results (PES, noise, SQD, fault tolerance) |
| POST | `/api/jobs` | `{"type": ..., "params": {...}}` → job (202) |
| GET | `/api/jobs` | recent jobs |
| GET | `/api/jobs/<id>` | status, progress, log, partial results, result |
| DELETE | `/api/jobs/<id>` | cancel |
| DELETE | `/api/cache` | clear cached results |

Job types and parameters (defaults in brackets):

- `energies`: `r` [1.098], `space` `cas66` | `fv108` [cas66]
- `adapt`: `r`, `space`, `max_ops` [80 / 120]
- `noise`: `r`, `depth` `half` | `chem` | integer [half], `r2` [3e-3], `r1` [r2/12], `p_ro` [0.015]
- `noise_sweep`: `r`, `depth`, `r2_list`, `p_ro`
- `sqd`: `r` [2.1], `space` [fv108], `bricks` [4], `r2`, `p_ro`, `shots` [3000], `seed` [101], `budgets`
- `pes_scan`: `space`, `r_list`, `with_adapt`
- `tests`: no parameters

Example:

```bash
curl -s -X POST localhost:8000/api/jobs -H 'Content-Type: application/json' \
     -d '{"type":"energies","params":{"r":2.1,"space":"fv108"}}'
curl -s localhost:8000/api/jobs/<id>
```

## Layout

```
web/backend/app.py             Flask app: API routes + static frontend
web/backend/jobs.py            job manager: validation, worker pool, progress, cancellation, disk cache
web/backend/engine_service.py  job implementations on top of src/ (SCF branch following, caches)
web/backend/warm.py            pre-computes the demo cache
web/frontend/                  index.html, styles.css, app.js, plot.js (no build step, no CDN needed)
tests/test_web_api.py          API checks against the published study numbers
```

The SCF for a new bond length follows the symmetric Hartree–Fock branch from equilibrium through
the published grid, so results at grid points are identical to `results/pes.json` and
`results/noise.json` (checked by `tests/test_web_api.py`).

Limits: minimal STO-3G basis, so exact answers are still classical; simulated noise
(depolarising, all-to-all); the Flask development server is meant for a laptop demo, not public hosting.
