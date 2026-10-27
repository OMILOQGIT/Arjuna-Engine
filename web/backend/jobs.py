"""Background job manager: validation, a small worker pool, progress, cancellation and a disk cache.

A job is created by POST /api/jobs and runs on a worker thread. Clients poll GET /api/jobs/<id>
for status, progress, log lines, partial results (live charts) and the final result. Finished
results are cached on disk by (job type, parameters, engine version), so repeating a run is instant.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import engine_service as S

ENGINE_VERSION = "arjuna-engine-2.0"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
NO_CACHE = {"tests"}


class BadRequest(ValueError):
    pass


# ----------------------------------------------------------------------------- validation
def _num(p, key, default, lo, hi, cast=float):
    v = p.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number")
    if isinstance(v, float) and not math.isfinite(v):
        raise BadRequest(f"'{key}' must be finite")
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return v


def _space(p, default="cas66"):
    s = p.get("space", default)
    if s not in S.SPACES:
        raise BadRequest(f"'space' must be one of {sorted(S.SPACES)}")
    return s


def _r(p, default=1.098):
    return S.norm_r(_num(p, "r", default, S.R_MIN, S.R_MAX))


def _depth(p):
    d = p.get("depth", "half")
    if d in ("half", "chem"):
        return d
    return _num(p, "depth", 0, 1, 150, int)


def validate(job_type: str, p: dict) -> dict:
    if not isinstance(p, dict):
        raise BadRequest("'params' must be an object")
    if job_type == "energies":
        return {"r": _r(p), "space": _space(p)}
    if job_type == "adapt":
        sp = _space(p)
        return {"r": _r(p), "space": sp, "max_ops": _num(p, "max_ops", 80 if sp == "cas66" else 120, 1, 150, int)}
    if job_type == "noise":
        r2 = _num(p, "r2", 3e-3, 1e-7, 3e-2)
        r1 = p.get("r1")
        return {"r": _r(p), "space": "cas66", "depth": _depth(p), "r2": r2,
                "r1": None if r1 is None else _num(p, "r1", r2 / 12, 0, 3e-2), "p_ro": _num(p, "p_ro", 1.5e-2, 0, 0.2)}
    if job_type == "noise_sweep":
        lst = p.get("r2_list", [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
        if not isinstance(lst, list) or not 2 <= len(lst) <= 16:
            raise BadRequest("'r2_list' must be a list of 2–16 numbers")
        lst = sorted({_num({"x": x}, "x", 0, 1e-7, 3e-2) for x in lst})
        return {"r": _r(p), "space": "cas66", "depth": _depth(p), "r2_list": lst, "p_ro": _num(p, "p_ro", 1.5e-2, 0, 0.2)}
    if job_type == "sqd":
        sp = _space(p, "fv108")
        n_str = 56 if sp == "fv108" else 20
        budgets = p.get("budgets") or ([8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52] if sp == "fv108" else [4, 6, 8, 10, 12, 14, 16, 18])
        if not isinstance(budgets, list) or not 1 <= len(budgets) <= 30:
            raise BadRequest("'budgets' must be a list of 1–30 integers")
        budgets = sorted({_num({"x": b}, "x", 1, 1, n_str, int) for b in budgets})
        return {"r": _r(p, 2.1), "space": sp, "bricks": _num(p, "bricks", 4, 1, 40, int),
                "r2": _num(p, "r2", 3e-3, 0, 3e-2), "p_ro": _num(p, "p_ro", 1.5e-2, 0, 0.2),
                "shots": _num(p, "shots", 3000, 50, 50000, int), "seed": _num(p, "seed", 101, 0, 2 ** 31 - 1, int),
                "budgets": budgets}
    if job_type == "pes_scan":
        lst = p.get("r_list")
        if not isinstance(lst, list) or not 1 <= len(lst) <= 40:
            raise BadRequest("'r_list' must be a list of 1–40 bond lengths")
        lst = sorted({S.norm_r(_num({"x": x}, "x", 1.0, S.R_MIN, S.R_MAX)) for x in lst})
        return {"space": _space(p), "r_list": lst, "with_adapt": bool(p.get("with_adapt", False))}
    if job_type == "tests":
        return {}
    raise BadRequest(f"unknown job type '{job_type}'; known: {sorted(S.JOBS)}")


# ----------------------------------------------------------------------------- jobs
def _clean(o):
    """Make results strict-JSON safe (NaN/inf -> null, numpy scalars -> Python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (ValueError, AttributeError):
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class Job:
    def __init__(self, job_type, params):
        self.id = uuid.uuid4().hex[:12]
        self.type, self.params = job_type, params
        self.status = "queued"
        self.progress_value, self.message = 0.0, "Queued"
        self.logs: list = []
        self.partial_data = None
        self.result = self.error = None
        self.cached = False
        self.created, self.started, self.finished = time.time(), None, None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # context API used by engine_service
    def progress(self, frac, msg=None):
        with self._lock:
            self.progress_value = max(self.progress_value, min(1.0, float(frac)))
            if msg:
                self.message = msg
                self._log(msg)

    def log(self, msg):
        with self._lock:
            self._log(msg)

    def _log(self, msg):
        t = 0.0 if self.started is None else time.time() - self.started
        if not self.logs or self.logs[-1][1] != msg:
            self.logs.append([round(t, 2), msg])
            del self.logs[:-300]

    def partial(self, data):
        with self._lock:
            self.partial_data = _clean(data)

    def check(self):
        if self._cancel.is_set():
            raise S.Cancelled()

    def to_dict(self, full=True):
        with self._lock:
            d = {"id": self.id, "type": self.type, "params": self.params, "status": self.status,
                 "progress": round(self.progress_value, 4), "message": self.message, "cached": self.cached,
                 "created": self.created, "started": self.started, "finished": self.finished,
                 "seconds": None if self.started is None else round((self.finished or time.time()) - self.started, 2)}
            if full:
                d.update({"logs": self.logs[-80:], "partial": self.partial_data, "result": self.result, "error": self.error})
            return d


class JobManager:
    def __init__(self, workers=2):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arjuna-job")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    @staticmethod
    def cache_key(job_type, params):
        blob = json.dumps({"v": ENGINE_VERSION, "t": job_type, "p": params}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:20]

    def submit(self, job_type, raw_params, use_cache=True):
        params = validate(job_type, raw_params or {})
        job = Job(job_type, params)
        with self.lock:
            self.jobs[job.id] = job
            for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-200]:
                if old.status in ("done", "error", "cancelled"):
                    self.jobs.pop(old.id, None)
        path = CACHE_DIR / f"{job_type}-{self.cache_key(job_type, params)}.json"
        if use_cache and job_type not in NO_CACHE and path.exists():
            try:
                job.result = json.loads(path.read_text())
                job.status, job.cached, job.progress_value, job.message = "done", True, 1.0, "Loaded from cache"
                job.started = job.finished = time.time()
                return job
            except (OSError, ValueError):
                pass
        self.pool.submit(self._run, job, path)
        return job

    def _run(self, job, path):
        if job._cancel.is_set():
            job.status, job.finished = "cancelled", time.time()
            return
        job.status, job.started = "running", time.time()
        job.log(f"Started {job.type}")
        try:
            res = _clean(S.JOBS[job.type](job, **job.params))
            job.result, job.status, job.progress_value, job.message = res, "done", 1.0, "Done"
            if job.type not in NO_CACHE:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(res))
                tmp.replace(path)
        except S.Cancelled:
            job.status, job.message = "cancelled", "Cancelled"
        except Exception as e:  # report to the client; keep the server alive
            job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "Failed"
            job.log(traceback.format_exc(limit=4))
        finally:
            job.finished = time.time()
            job.log(f"{job.status} in {job.finished - job.started:.1f} s")

    def get(self, jid):
        return self.jobs.get(jid)

    def cancel(self, jid):
        job = self.jobs.get(jid)
        if job and job.status in ("queued", "running"):
            job._cancel.set()
            if job.status == "queued":
                job.status, job.finished, job.message = "cancelled", time.time(), "Cancelled"
        return job

    def list(self):
        with self.lock:
            return [j.to_dict(full=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def clear_cache(self):
        n = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            n += 1
        return n
