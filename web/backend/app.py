"""Arjuna web app: Flask API + static frontend.

Run:  python -m web.backend.app            (from the repository root)  ->  http://localhost:8000
      PORT=9000 WORKERS=3 python -m web.backend.app
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from flask import Flask, abort, jsonify, request, send_from_directory

from . import engine_service as S
from .jobs import ENGINE_VERSION, BadRequest, JobManager

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / "web" / "frontend"
RESULTS = ROOT / "results"
STARTED = time.time()

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False
jobs = JobManager(workers=int(os.environ.get("WORKERS", "2")))


def _read(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def reference():
    """Published (pre-computed) results of the full study, for overlays and comparisons."""
    pes = _read("pes.json") or []
    curve = {}
    for sp in S.SPACES:
        rows = []
        for r in pes:
            d = r.get(sp)
            if not d:
                continue
            ad = d.get("adapt") or {}
            rows.append({"r": r["r"], "exact": d["exact"], "hf": d["hf"],
                         "ccsd": d["ccsd"] if d["cc_converged"] else None,
                         "ccsd_t": d["ccsd_t"] if d["cc_converged"] else None,
                         "adapt": ad.get("final"), "k_chem_acc": ad.get("k_chem_acc"),
                         "cx": (ad.get("circuit_at_chem_acc") or {}).get("cx"),
                         "cx_untapered": (ad.get("circuit_at_chem_acc") or {}).get("cx_untapered")})
        curve[sp] = rows
    return {"pes": curve, "noise": _read("noise.json"), "sqd_capped": _read("sqd_capped.json"),
            "fault_tolerant": _read("ft_estimate.json")}


_REF = None


@app.after_request
def headers(resp):
    resp.headers["Cache-Control"] = "no-store" if request.path.startswith("/api/") else "no-cache"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.errorhandler(BadRequest)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return "Not found", 404


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "engine": ENGINE_VERSION, "uptime_s": round(time.time() - STARTED, 1),
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "workers": jobs.pool._max_workers, "cache": S.cache_stats(),
                    "active_jobs": sum(1 for j in jobs.list() if j["status"] in ("queued", "running"))})


@app.get("/api/reference")
def get_reference():
    global _REF
    if _REF is None:
        _REF = reference()
    return jsonify(_REF)


@app.get("/api/meta")
def meta():
    return jsonify({"spaces": S.SPACES, "r_range": [S.R_MIN, S.R_MAX], "chemical_accuracy_mHa": 1.6,
                    "job_types": sorted(S.JOBS), "reference_geometries": [0.9, 1.0, 1.098, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0]})


@app.post("/api/jobs")
def create_job():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "type" not in body:
        raise BadRequest("body must be JSON: {\"type\": ..., \"params\": {...}}")
    job = jobs.submit(body["type"], body.get("params") or {}, use_cache=body.get("use_cache", True) is not False)
    return jsonify(job.to_dict()), 202


@app.get("/api/jobs")
def list_jobs():
    return jsonify(jobs.list())


@app.get("/api/jobs/<jid>")
def get_job(jid):
    job = jobs.get(jid)
    if not job:
        abort(404)
    return jsonify(job.to_dict())


@app.delete("/api/jobs/<jid>")
def cancel_job(jid):
    job = jobs.cancel(jid)
    if not job:
        abort(404)
    return jsonify(job.to_dict(full=False))


@app.delete("/api/cache")
def clear_cache():
    return jsonify({"removed": jobs.clear_cache()})


# ----------------------------------------------------------------------------- frontend
@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.get("/offline-lab")
def offline_lab():
    return send_from_directory(ROOT / "lab", "arjuna-lab.html")


@app.get("/<path:path>")
def static_files(path):
    if path.startswith("api/"):
        abort(404)
    return send_from_directory(FRONT, path)


def main():
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Arjuna web app on http://{host}:{port}  (engine {ENGINE_VERSION}, {jobs.pool._max_workers} workers)")
    if os.environ.get("WARM", "1") != "0":
        import threading
        threading.Thread(target=S.warm_branch, daemon=True, name="warm").start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
