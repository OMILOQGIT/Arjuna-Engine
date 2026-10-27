"""Web API checks: validation, job lifecycle, cancellation, caching, and that the API reproduces the
published study numbers (results/pes.json, results/noise.json) through the same engine.

    python tests/test_web_api.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.backend import jobs as J  # noqa: E402

J.CACHE_DIR = Path(tempfile.mkdtemp(prefix="arjuna-cache-"))  # isolated cache for the test
from web.backend.app import app  # noqa: E402

c = app.test_client()
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def run(t, p, timeout=1200):
    j = c.post("/api/jobs", json={"type": t, "params": p}).get_json()
    t0 = time.time()
    while j["status"] in ("queued", "running") and time.time() - t0 < timeout:
        time.sleep(0.3)
        j = c.get(f"/api/jobs/{j['id']}").get_json()
    return j


h = c.get("/api/health").get_json()
check("health", h["status"] == "ok", h["engine"])
check("frontend served", c.get("/").status_code == 200 and b"Arjuna" in c.get("/").data)
check("unknown job type -> 400", c.post("/api/jobs", json={"type": "nope"}).status_code == 400)
check("bond length out of range -> 400", c.post("/api/jobs", json={"type": "energies", "params": {"r": 9}}).status_code == 400)
check("bad space -> 400", c.post("/api/jobs", json={"type": "energies", "params": {"space": "x"}}).status_code == 400)
check("missing job -> 404", c.get("/api/jobs/doesnotexist").status_code == 404)

pes = {round(r["r"], 3): r for r in json.loads((ROOT / "results" / "pes.json").read_text())}
noise = json.loads((ROOT / "results" / "noise.json").read_text())

j = run("energies", {"r": 2.1, "space": "cas66"})
d = j["result"]
ref = pes[2.1]["cas66"]
check("energies job done", j["status"] == "done", f"{j['seconds']} s")
check("exact energy = published", abs(d["exact"] - ref["exact"]) < 1e-8, f"{d['exact']:.9f}")
check("CCSD(T) = published (collapse)", abs(d["ccsd_t"] - ref["ccsd_t"]) < 1e-7 and d["cc_verdict"].startswith("collapsed"),
      f"{d['err_mHa']['ccsd_t']:.2f} mHa")
check("tapering 12 -> 7", d["qubits"] == 12 and d["qubits_tapered"] == 7)

j2 = run("energies", {"r": 2.1, "space": "cas66"})
check("second identical run served from cache", j2["cached"] and j2["result"]["exact"] == d["exact"])

j = run("energies", {"r": 1.77, "space": "cas66"})
check("arbitrary bond length 1.77 A", j["status"] == "done" and j["result"]["qubits_tapered"] == 7,
      f"CCSD(T) {j['result']['err_mHa']['ccsd_t']:+.2f} mHa")

j = run("adapt", {"r": 2.1, "space": "cas66"})
a = j["result"]
cref = pes[2.1]["cas66"]["adapt"]
check("ADAPT reaches chemical accuracy", a["k_chem_acc"] == cref["k_chem_acc"], f"k={a['k_chem_acc']}")
check("ADAPT circuit = published", a["circuit_at_chem_acc"]["cx"] == cref["circuit_at_chem_acc"]["cx"],
      f"{a['circuit_at_chem_acc']['cx']} CX")
check("ADAPT singlet", a["iterations"][-1]["s2"] < 1e-3)

j = run("noise", {"r": 2.1, "r2": 3e-3})
n = j["result"]
nref = noise["2.1"]["ablation"]
check("noise ablation = published", all(abs(n["ablation"][k]["err_mHa"] - nref[k]["err_mHa"]) < 1e-6 for k in nref),
      f"REM {n['ablation']['rem']['err_mHa']:.2f} mHa")

j = run("sqd", {"r": 2.1, "space": "cas66", "shots": 1000, "budgets": [4, 8, 12]})
s = j["result"]
check("SQD job (7 qubits)", j["status"] == "done" and len(s["budget_rows"]) == 4 and abs(s["budget_rows"][-1]["chip"]) < 1e-6,
      f"full space exact; K=8 chip {s['budget_rows'][1]['chip']:.1f} vs random {s['budget_rows'][1]['random']}")

jid = c.post("/api/jobs", json={"type": "adapt", "params": {"r": 2.95, "space": "fv108"}}).get_json()["id"]
time.sleep(1.0)
c.delete(f"/api/jobs/{jid}")
t0 = time.time()
while c.get(f"/api/jobs/{jid}").get_json()["status"] in ("queued", "running") and time.time() - t0 < 300:
    time.sleep(0.3)
check("cancel a running job", c.get(f"/api/jobs/{jid}").get_json()["status"] == "cancelled")
check("job list", len(c.get("/api/jobs").get_json()) >= 6)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
