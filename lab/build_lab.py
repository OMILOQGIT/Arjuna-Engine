"""Assemble lab/arjuna-lab.html from the template, the engine and the exported data."""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "lab/lab_template.html").read_text(encoding="utf-8")
eng = (ROOT / "lab/engine.js").read_text(encoding="utf-8")
data = (ROOT / "results/lab_data.json").read_text()
noise = json.loads((ROOT / "results/noise.json").read_text())
extra = {"noise": {k: {"sweep": v["sweep"], "ablation": v["ablation"]} for k, v in noise.items()}}
assert "</script" not in eng and "</script" not in data
out = t.replace("__ENGINE__", eng).replace("__DATA__", data).replace("__EXTRA__", json.dumps(extra, separators=(",", ":"))).replace("__GENERATED__", time.strftime("%Y-%m-%d"))
(ROOT / "lab/arjuna-lab.html").write_text(out, encoding="utf-8")
print("wrote lab/arjuna-lab.html", round(len(out.encode()) / 1e6, 2), "MB")
