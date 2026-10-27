"""Figures for the research draft (reads results/*.json)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CHEM = 1.6
INK, GRID = "#1f2328", "#d0d7de"
COL = {"exact": INK, "hf": "#8c959f", "ccsd": "#bf8700", "ccsd_t": "#cf222e", "adapt": "#0969da", "sqd": "#1a7f37"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def pes():
    """Left: energy curves (valence space). Right: |error| per method; CC collapse (below exact) marked."""
    rows = json.loads((RES / "pes.json").read_text())
    r = np.array([x["r"] for x in rows])
    fv = [x["fv108"] for x in rows]
    cas = [x["cas66"] for x in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(r, [d["exact"] for d in fv], "-", color=COL["exact"], lw=2, label="Exact (singlet)")
    a1.plot(r, [d["hf"] for d in fv], "--", color=COL["hf"], label="RHF")
    cc = [(x, d["ccsd_t"]) for x, d in zip(r, fv) if d["cc_converged"]]
    a1.plot([c[0] for c in cc], [c[1] for c in cc], "o", ms=5, color=COL["ccsd_t"], label="CCSD(T)")
    a1.plot(r, [d["adapt"]["final"] for d in fv], "s", ms=7, mfc="none", color=COL["adapt"], label="ADAPT-VQE (11 qubits)")
    a1.set_title("N$_2$/STO-3G, valence space (10e, 8o)"); a1.set_xlabel("N–N distance (Å)"); a1.set_ylabel("Energy (Ha)")
    a1.legend(frameon=False, fontsize=8.5)
    for d, x in zip(fv, r):
        if not d["cc_converged"]:
            a2.plot(x, 400, "x", color="#cf222e", ms=9)
            continue
        e = 1e3 * (d["ccsd_t"] - d["exact"])
        a2.plot(x, abs(e), "v" if e < -1.6 else "o", color="#cf222e" if e < -1.6 else COL["ccsd_t"], ms=7)
    a2.semilogy(r, [max(1e3 * (d["adapt"]["final"] - d["exact"]), 1e-4) for d in fv], "s-", color=COL["adapt"], label="ADAPT-VQE, 11 qubits")
    a2.semilogy(r, [max(1e3 * (d["adapt"]["final"] - d["exact"]), 1e-4) for d in cas], "s--", color=COL["adapt"], alpha=.5, label="ADAPT-VQE, 7 qubits")
    a2.axhline(CHEM, color=INK, lw=1, ls=":"); a2.text(r[0], CHEM * 1.3, "1.6 mHa", fontsize=8)
    a2.plot([], [], "o", color=COL["ccsd_t"], label="CCSD(T) above exact"); a2.plot([], [], "v", color="#cf222e", label="CCSD(T) collapsed below exact")
    a2.plot([], [], "x", color="#cf222e", label="CCSD(T) not converged")
    a2.set_ylim(1e-4, 1e3); a2.set_xlabel("N–N distance (Å)"); a2.set_ylabel("|E − E$_{exact}$| (mHa)")
    a2.set_title("Errors against exact diagonalisation"); a2.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "pes.png", dpi=160)


def adapt_conv():
    rows = json.loads((RES / "pes.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        for key, ls in [("cas66", "-"), ("fv108", "--")]:
            d = row[key]
            if "adapt" not in d or row["r"] not in (1.098, 1.7, 2.1):
                continue
            e = 1e3 * (np.array(d["adapt"]["energies"]) - d["exact"])
            ax.semilogy(np.arange(len(e)), np.maximum(e, 1e-6), ls,
                        label=f"{'CAS(6e,6o) 7q' if key == 'cas66' else 'FV(10e,8o) 11q'}  R={row['r']} Å")
    ax.axhline(CHEM, color=INK, lw=1, ls=":")
    ax.set_xlabel("ADAPT iterations (operators in ansatz)")
    ax.set_ylabel("Energy error (mHa)")
    ax.set_title("ADAPT-VQE convergence: stretched bonds need deeper circuits")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "adapt_convergence.png", dpi=160)


def noise():
    d = json.loads((RES / "noise.json").read_text())
    keys = ["raw", "rem", "rem+zne_lin", "rem+zne_rich", "rem+zne_exp"]
    names = ["Raw", "REM", "REM+ZNE\nlinear", "REM+ZNE\nRichardson", "REM+ZNE\nexponential"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    w = 0.38
    for i, (r, v) in enumerate(d.items()):
        errs = [abs(v["ablation"][k]["err_mHa"]) for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 0.5) * w, errs, w, label=f"R={r} Å ({v['gates']['cx']} CX)")
    ax.set_yscale("log")
    ax.axhline(CHEM, color=INK, lw=1, ls=":")
    ax.set_xticks(range(len(keys)), names, fontsize=8)
    nm = next(iter(d.values()))["noise_model"]
    ax.set_title(f"Mitigation ablation (r2={nm['r2']:.0e}, r1={nm['r1']:.1e}, ro={nm['p_ro']:.1e})", fontsize=9)
    ax.set_ylabel("|Energy error| (mHa)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for r, v in d.items():
        sw = v["sweep"]
        x = [s["r2"] for s in sw]
        ax.loglog(x, [abs(s["rem"]) for s in sw], "o--", ms=4, label=f"REM, R={r}")
        ax.loglog(x, [abs(s["rem+zne_rich"]) for s in sw], "s-", ms=4, label=f"REM+ZNE(Rich.), R={r}")
    ax.axhline(CHEM, color=INK, lw=1, ls=":")
    ax.set_xlabel("Two-qubit gate error r2")
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title("What gate error does chemical accuracy need?", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "noise.png", dpi=160)


def sqd():
    p = RES / "sqd_capped.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, r in zip(axes, (2.1, 1.098)):
        rows = [x for x in d["rows"] if abs(x["r"] - r) < 1e-6]
        f = [x["frac"] for x in rows]
        for key, col, lab in (("noisy", COL["sqd"], "noisy chip samples"), ("clean", COL["adapt"], "noise-free chip samples"), ("random", COL["ccsd_t"], "random bitstrings")):
            ax.semilogy(f, [max(x[key]["median"], 1e-4) if x[key]["median"] is not None else np.nan for x in rows], "o-", color=col, label=lab, ms=4)
        ax.semilogy(f, [max(x["hf_ranked"], 1e-4) for x in rows], "s--", color=COL["hf"], label="classical ranking (fewest excitations)", ms=4)
        ax.axhline(CHEM, color=INK, lw=1, ls=":")
        ax.set_title(f"Capped SQD, FV(10e,8o), R = {r} Å (median of 10 seeds)", fontsize=9.5)
        ax.set_xlabel("share of the 3,136 determinants solved exactly")
    axes[0].set_ylabel("Energy error (mHa)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sqd.png", dpi=160)


for fn in (pes, adapt_conv, noise, sqd):
    try:
        fn()
        print("ok", fn.__name__)
    except FileNotFoundError as e:
        print("skip", fn.__name__, e, file=sys.stderr)
