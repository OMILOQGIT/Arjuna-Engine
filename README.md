# Arjuna · N≡N quantum chemistry engine (Team Arjuna, QC-02, Chakravyuh)

**Start the website:** `./run.sh` (macOS/Linux) or `run.bat` (Windows), then open http://localhost:8000.
Needs Python 3.10+; the script installs numpy, scipy and flask into `.venv`. Docker: `docker build -t arjuna . && docker run -p 8000:8000 arjuna`.

| Folder | What it is |
|---|---|
| `web/` | the website: Flask API backend + frontend ([web/README.md](web/README.md)) |
| `src/` | the verified engine: integrals, Hartree–Fock, CCSD(T), qubit mapping, tapering, ADAPT-VQE, noise, SQD |
| `tests/` | regression suites (engine and web API) |
| `scripts/`, `results/` | the full study and its published numbers |
| `lab/` | self-contained offline browser lab (open `lab/arjuna-lab.html`, no server needed) |
| `port/` | the same pipeline on IBM's stack (PySCF, Qiskit, qiskit-addon-sqd), not yet run |
| `deck/` | 12-slide presentation and pitch script |

Research write-up: [RESEARCH.md](RESEARCH.md).
