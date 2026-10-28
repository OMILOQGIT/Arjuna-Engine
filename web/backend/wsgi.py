"""Production entry point (gunicorn / Render):

    gunicorn web.backend.wsgi:app --workers 1 --threads 8 --bind 0.0.0.0:$PORT

Use exactly ONE worker process: jobs live in that process's memory (threads are fine).
Set WARM=1 to solve the Hartree–Fock branch in the background at startup.
"""
import os
import threading

from . import engine_service as S
from .app import app  # noqa: F401  (the WSGI callable)

if os.environ.get("WARM", "0") == "1":
    threading.Thread(target=S.warm_branch, daemon=True, name="warm").start()
