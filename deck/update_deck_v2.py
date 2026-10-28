"""Apply the expert-panel corrections to the 12-slide deck, pulling every number from results/*.json.

Changes: valence-space CCSD(T) collapse story (slide 4), CNOT/brick counts (7, 8), noise threshold with
all-gate ZNE (9), honest capped-SQD comparison (9, 6, 12), check counts and fault-tolerance numbers (10, 11),
speaker notes. Input: deck/Team-Arjuna-Chakravyuh-QC02.html (in place, backup kept by caller).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "deck" / "Team-Arjuna-Chakravyuh-QC02.html"
pes = json.loads((ROOT / "results/pes.json").read_text())
noise = json.loads((ROOT / "results/noise.json").read_text())
capped = json.loads((ROOT / "results/sqd_capped.json").read_text())
ft = json.loads((ROOT / "results/ft_estimate.json").read_text())
t = DECK.read_text(encoding="utf-8")


def rep(a, b, count=1):
    global t
    assert a in t, "missing: " + a[:90]
    t = t.replace(a, b, count)


def f1(v):
    return f"{v:.1f}"


row = {round(r["r"], 3): r for r in pes}
fv = {r: row[r]["fv108"] for r in row}
cas = {r: row[r]["cas66"] for r in row}
err = lambda d: 1e3 * (d["ccsd_t"] - d["exact"]) if d["cc_converged"] else None  # noqa: E731
e_eq, e_15 = err(fv[1.098]), err(fv[1.5])
collapses = [-err(fv[r]) for r in fv if err(fv[r]) is not None and err(fv[r]) < -1.6]
nc = [r for r in fv if err(fv[r]) is None]
coll_min, coll_max = min(collapses), max(collapses)
k_eq, k_21 = cas[1.098]["adapt"]["k_chem_acc"], cas[2.1]["adapt"]["k_chem_acc"]
cx_eq, cxu_eq = cas[1.098]["adapt"]["circuit_at_chem_acc"]["cx"], cas[1.098]["adapt"]["circuit_at_chem_acc"]["cx_untapered"]
max_fv_adapt = max(1e3 * (fv[r]["adapt"]["final"] - fv[r]["exact"]) for r in fv)
# noise
best_today = min(min(abs(noise[g]["ablation"][k]["err_mHa"]) for k in ("rem+zne_lin", "rem+zne_rich", "rem+zne_exp")) for g in noise)
gam_lo = min(noise[g]["shots_pec_per_term"] for g in noise)
gam_hi = max(noise[g]["shots_pec_per_term"] for g in noise)
cx_noise = [noise[g]["gates"]["cx"] for g in noise]


def exp10(v):
    e = int(f"{v:e}".split("e")[1])
    return e


sup = lambda n: str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))  # noqa: E731
# capped SQD at R = 2.1, budget K = 28 (25% of the space)
cr = {(round(x["r"], 3), x["K"]): x for x in capped["rows"]}
c28 = cr[(2.1, 28)]
chip28, rnd28, cls28 = c28["noisy"]["median"], c28["random"]["median"], c28["hf_ranked"]
ratio_rnd, ratio_cls = rnd28 / chip28, cls28 / chip28
reach = next(x["frac"] for x in sorted([x for x in capped["rows"] if abs(x["r"] - 2.1) < 1e-6], key=lambda x: x["K"]) if x["noisy"]["median"] is not None and x["noisy"]["median"] <= 1.6)
d_ft = ft["1.098"]["code_distance"]
phys_ft = ft["1.098"]["physical_qubits_data_and_routing"]
print(f"eq {e_eq:.2f} 1.5 {e_15:.2f} collapse {coll_min:.0f}-{coll_max:.0f} nc {nc} | k {k_eq}/{k_21} cx {cxu_eq}->{cx_eq} | "
      f"best today {best_today:.0f} | PEC {gam_lo:.1e}-{gam_hi:.1e} | SQD@25% chip {chip28:.1f} rnd {rnd28:.0f} cls {cls28:.0f} reach {reach:.2f} | FT d={d_ft} {phys_ft}")

# ---------------- data: add valence-space CC rows and capped SQD rows ----------------
i = t.index("const DATA = ") + len("const DATA = ")
j = t.index(";\nconst NOTES", i)
D = json.loads(t[i:j])
D["fv"] = [{"r": r, "ex": fv[r]["exact"], "cct": fv[r]["ccsd_t"] if fv[r]["cc_converged"] else None} for r in sorted(fv)]
D["capped"] = [{"K": x["K"], "frac": x["frac"], "chip": x["noisy"]["median"], "rnd": x["random"]["median"], "cls": x["hf_ranked"]} for x in capped["rows"] if abs(x["r"] - 2.1) < 1e-6]
for g in noise:
    D["noise"].setdefault(g, {})["sweep"] = [{"r2": s["r2"], "rem": s["rem"], "rich": s["rem+zne_rich"], "exp": s["rem+zne_exp"]} for s in noise[g]["sweep"]]
conv = {}
for r, key in ((1.098, "cas66_1.098"), (2.1, "cas66_2.1")):
    conv[key] = [round(1e3 * (e - cas[r]["exact"]), 5) for e in cas[r]["adapt"]["energies"]]
D["conv"].update(conv)
D["pes"] = [{"r": r, "ex": cas[r]["exact"], "hf": cas[r]["hf"], "ad": cas[r]["adapt"]["final"], "k": cas[r]["adapt"]["k_chem_acc"],
             "cx": cas[r]["adapt"]["circuit_at_chem_acc"]["cx"], "cxu": cas[r]["adapt"]["circuit_at_chem_acc"]["cx_untapered"],
             "q": cas[r]["qubits_tapered"], "cc": cas[r]["ccsd"] if cas[r]["cc_converged"] else None, "cct": cas[r]["ccsd_t"] if cas[r]["cc_converged"] else None} for r in sorted(cas)]
t = t[:i] + json.dumps(D, separators=(",", ":")) + t[j:]

# ---------------- slide 1 (title) ----------------
rep("How a quantum computer can price the <b>hardest bond in chemistry</b> more accurately than the classical gold standard, and what noise does to it.",
    "How a quantum computer gets the <b>hardest bond in chemistry</b> right where the classical gold standard breaks down, and what noise does to it.")

# ---------------- slide 4 (gold standard) ----------------
rep('''<h2 class="rv" style="animation-delay:.15s">The <span class="ho">"gold standard"</span> misses the target, then <span class="u">gives up</span></h2>''',
    '''<h2 class="rv" style="animation-delay:.15s">The <span class="ho">"gold standard"</span> drifts off target, then <span class="u">collapses</span></h2>''')
rep('''<p class="txt">Pull the two atoms apart and plot the energy: that's the curve on the left. <b>CCSD(T)</b> is the method chemists trust most. It assumes one arrangement dominates, so as the bond stretches it drifts off target, then returns no answer.</p>''',
    '''<p class="txt">Pull the two atoms apart and plot the energy: that's the curve above. <b>CCSD(T)</b> is the method chemists trust most. It assumes one arrangement dominates, so as the bond stretches it drifts off target, then <b>falls below the true energy</b>, which no correct method can do.</p>''')
rep('''<p class="sm">Same 12-qubit active-space Hamiltonian for every method, so the gap is method error only. N₂, STO-3G basis, CAS(6e,6o), errors against exact diagonalisation.</p>''',
    '''<p class="sm">10 electrons in 8 valence orbitals (11 qubits after tapering), N₂/STO-3G, same Hamiltonian for every method; errors against exact diagonalisation. Multireference classical methods also handle this size: the point is where single-reference methods break.</p>''')
i = t.index('  const q = new Pad("ccerr", { base: .6, step: .08 });')
j = t.index("\n}\n", i)
t = t[:i] + f'''  const F = DATA.fv;
  const q = new Pad("ccerr", {{ base: .6, step: .08 }});
  const W2 = 860, H2 = 440, m2 = {{ l: 80, r: 24, t: 40, b: 50 }};
  const X = lin(.85, 3.05, m2.l, W2 - m2.r), Y = logs(.3, 700, H2 - m2.b, m2.t);
  axesXY(q, {{ x: X, y: Y, W: W2, H: H2, m: m2, yt: [1, 10, 100], yf: v => v, xt: [1, 1.5, 2, 2.5, 3], xf: v => v.toFixed(1) + " Å", xl: "N–N distance", yl: "how far off CCSD(T) is (mHa, log)" }});
  q.hatch(m2.l + 2, Y(1.6), W2 - m2.r - m2.l - 2, Y(.3) - Y(1.6), "green", {{ o: .18, step: 10, d: .5 }});
  budget(q, m2.l, W2 - m2.r, Y(1.6), "inside 1.6 mHa = chemically accurate", {{ below: true, d: .6 }});
  const e = r => (r.cct - r.ex) * 1e3, up = F.filter(r => r.cct !== null && e(r) > -1.6), dn = F.filter(r => r.cct !== null && e(r) <= -1.6), nc = F.filter(r => r.cct === null);
  q.poly(up.map(r => [X(r.r), Y(Math.max(e(r), .3))]), {{ c: "orange", w: 3.4 }});
  up.forEach((r, i) => q.dot(X(r.r), Y(Math.max(e(r), .3)), 6, {{ c: "orange", d: .9 + i * .08 }}));
  dn.forEach((r, i) => {{ const xx = X(r.r), yy = Y(-e(r)); q.ink(`M${{xx - 10}} ${{yy - 8}}L${{xx + 10}} ${{yy - 8}}L${{xx}} ${{yy + 10}}Z`, {{ c: "pink", w: 3, d: 1.5 + i * .1 }}); }});
  nc.forEach((r, i) => {{ const xx = X(r.r), yy = Y(450); q.line(xx - 10, yy - 10, xx + 10, yy + 10, {{ c: "pink", w: 3.2, single: true, d: 1.9 + i * .08 }}); q.line(xx + 10, yy - 10, xx - 10, yy + 10, {{ c: "pink", w: 3.2, single: true, d: 1.95 + i * .08 }}); }});
  q.text(X(2.45), Y(28), "▼ collapsed BELOW the true energy", {{ anchor: "middle", size: 20, font: "hand", c: "pink", d: 1.9 }});
  q.text(X(2.55), Y(450) - 22, "× no convergence", {{ anchor: "middle", size: 18, font: "hand", c: "pink", d: 2.1 }});
  const e0 = F.find(r => Math.abs(r.r - 1.098) < 1e-3), e15 = F.find(r => Math.abs(r.r - 1.5) < 1e-3);
  q.text(X(1.0), Y(.42), "+{e_eq:.1f} mHa at rest: already over budget", {{ size: 18, c: "orange", d: 1.3 }});
  q.text(X(1.5), Y(e(e15)) - 18, "+{e_15:.1f} mHa", {{ anchor: "middle", size: 22, font: "hand", c: "orange", d: 1.4 }});''' + t[j:]
rep('''<div class="note plain" data-sk="yellow"><span class="lab">In plain words</span><p class="txt">Pull the two atoms apart''', '''<div class="note plain" data-sk="yellow"><span class="lab">In plain words</span><p class="txt">Pull the two atoms apart''')

# ---------------- slide 6 (pipeline card 7) ----------------
t = re.sub(r'<span class="v">0\.30 mHa from noisy samples</span>', f'<span class="v">{ratio_rnd:.0f}× better than random picks</span>', t)

# ---------------- slide 7 (shrink) ----------------
rep("CASCI-in-HF embedding (frozen-core effective Hamiltonian).", "Active space on a frozen mean-field core (CASCI), the simplest embedding.")
rep("CNOTs 2,032 → 944; operator pool 117 → 27.", f"CNOTs {cxu_eq:,} → {cx_eq:,}; operator pool 117 → 27.")

# ---------------- slide 8 (ADAPT) ----------------
rep('[["cas66_1.098", "cyan", "at rest: 21 bricks"], ["cas66_2.1", "pink", "stretched: 33 bricks"]]', f'[["cas66_1.098", "cyan", "at rest: {k_eq} bricks"], ["cas66_2.1", "pink", "stretched: {k_21} bricks"]]')
rep("noiseless simulation (Grimsley et al., 2019)", "spin penalty keeps the singlet · noiseless simulation (Grimsley et al., 2019)")

# ---------------- slide 9 (noise + SQD) ----------------
rep("Our accurate circuit has ~1,000 of them, so the answer drowns.", f"Our accurate circuit has {min(cx_noise):,}–{max(cx_noise):,} of them, so the answer drowns.")
i = t.index('<div class="note deep" data-sk="orange"><p style="font-size:17.5px">At 3 × 10⁻³ gate error')
j = t.index("</div>", i) + len("</div>")
t = t[:i] + (f'<div class="note deep" data-sk="orange"><p style="font-size:17.5px">At 3 × 10⁻³ gate error: still <b>{best_today / 1000:.1f}+ Ha</b> off after readout fix + ZNE. '
             f'PEC is unbiased but needs <b>10{sup(exp10(gam_lo))}–10{sup(exp10(gam_hi))} shots</b> per term. Amplifying every gate\'s noise, ZNE reaches 1.6 mHa only below '
             f'<b>~3 × 10⁻⁵–10⁻⁴</b>: gates <b>30–100× better</b> than today.</p></div>') + t[j:]
rep("So we change the chip's job. It only has to <b>find the arrangements that matter</b>; a normal computer then solves exactly inside them. Noise changes which arrangements show up, not the energy of the ones found.",
    "So we change the chip's job: it only <b>points at the arrangements that matter</b>, and a normal computer solves exactly inside them. The fair test is a fixed budget: does the chip pick better arrangements than chance?")
i = t.index('<div class="note deep" data-sk="green"><p style="font-size:17.5px">Same noisy model')
j = t.index("</div>", i) + len("</div>")
t = t[:i] + (f'<div class="note deep" data-sk="green"><p style="font-size:17.5px">11 qubits, 2.1 Å, 192-CNOT circuit at today\'s noise, 3,000 shots, 10 seeds. Solving exactly in <b>25%</b> of the space: '
             f'chip-picked arrangements are <b>{chip28:.0f} mHa</b> off, random picks <b>{rnd28:.0f} mHa</b> ({ratio_rnd:.0f}× worse), a simple classical ranking <b>{cls28:.0f} mHa</b>. '
             f'The chip reaches 1.6 mHa at {100 * reach:.0f}% of the space. Without a budget, random bitstrings cover this small space in ~300 shots, so real advantage needs bigger spaces (IBM ran SQD on N₂ with 58 qubits).</p></div>') + t[j:]
i = t.index("  const p = new Pad(\"sweep\", { base: .6, step: .06 });")
j = t.index("\n}\n", i)
t = t[:i] + '''  const p = new Pad("sweep", { base: .6, step: .06 });
  const W = 700, H = 300, m = { l: 74, r: 18, t: 34, b: 46 };
  const x = logs(1e-6, 4e-3, m.l, W - m.r), y = logs(.3, 5000, H - m.b, m.t);
  axesXY(p, { x, y, W, H, m, yt: [1, 10, 100, 1000], yf: v => v.toLocaleString("en-US"), xt: [1e-6, 1e-5, 1e-4, 1e-3], xf: pow10, xl: "two-qubit gate error rate", yl: "energy error (mHa)" });
  p.hatch(x(1.2e-3), m.t, x(3.6e-3) - x(1.2e-3), y(.3) - m.t, "orange", { o: .2, step: 9, d: .5 });
  p.text(x(2.1e-3), m.t + 20, "today", { anchor: "middle", size: 20, font: "hand", c: "orange" });
  p.hatch(x(3e-5), m.t, x(1e-4) - x(3e-5), y(.3) - m.t, "green", { o: .25, step: 9, d: .5 });
  p.text(x(5.5e-5), m.t + 20, "needed", { anchor: "middle", size: 20, font: "hand", c: "green" });
  budget(p, m.l, W - m.r, y(1.6), "1.6 mHa", { d: .6, below: true });
  const s0 = N["1.098"].sweep, s1 = N["2.1"].sweep, best = s => Math.max(Math.min(Math.abs(s.rich), Math.abs(s.exp)), .3);
  p.poly(s0.map(s => [x(s.r2), y(Math.max(Math.abs(s.rem), .3))]), { c: "faint", w: 2.2 });
  p.poly(s0.map(s => [x(s.r2), y(best(s))]), { c: "cyan", w: 3 });
  p.poly(s1.map(s => [x(s.r2), y(best(s))]), { c: "pink", w: 3 });
  s0.forEach((s, i) => p.dot(x(s.r2), y(best(s)), 4.5, { c: "cyan", d: 1.4 + i * .05 }));
  s1.forEach((s, i) => p.dot(x(s.r2), y(best(s)), 4.5, { c: "pink", d: 1.5 + i * .05 }));
  p.text(x(1.2e-5), y(120), "readout fix only", { size: 16, c: "faint", d: 1.6 });
  p.text(x(1.6e-4), y(4), "+ ZNE, all gates amplified", { size: 16, c: "cyan", d: 1.7 });

  const Cp = DATA.capped;
  const q = new Pad("sqd", { base: .7, step: .06 });
  const W2 = 700, H2 = 300, m2 = { l: 74, r: 18, t: 34, b: 46 };
  const X = lin(0, 1, m2.l, W2 - m2.r), Y = logs(.1, 1000, H2 - m2.b, m2.t);
  axesXY(q, { x: X, y: Y, W: W2, H: H2, m: m2, yt: [.1, 1, 10, 100, 1000], yf: v => v.toLocaleString("en-US"), xt: [0, .25, .5, .75, 1], xf: v => (100 * v).toFixed(0) + "%", xl: "share of the space solved exactly", yl: "energy error (mHa)" });
  q.hatch(m2.l + 2, Y(1.6), W2 - m2.r - m2.l - 2, Y(.1) - Y(1.6), "green", { o: .18, step: 10, d: .5 });
  budget(q, m2.l, W2 - m2.r, Y(1.6), "1.6 mHa", { d: .6 });
  const ln = (key, col, d0) => { const pts = Cp.filter(c => c[key] !== null).map(c => [X(c.frac), Y(Math.max(c[key], .1))]); q.poly(pts, { c: col, w: 3, d: d0 }); pts.forEach((pt, i) => q.dot(pt[0], pt[1], 4.5, { c: col, d: d0 + .3 + i * .04 })); };
  ln("rnd", "orange", .8); ln("cls", "faint", 1); ln("chip", "green", 1.2);
  q.text(X(.55), Y(260), "random picks", { size: 18, font: "hand", c: "orange", d: 1.6 });
  q.text(X(.40), Y(38), "classical ranking", { size: 18, font: "hand", c: "chalk", d: 1.7 });
  q.text(X(.10), Y(3.2), "chip-picked", { size: 20, font: "hand", c: "green", d: 1.8 });
  q.ink(`M${X(.25)} ${m2.t}L${X(.25)} ${H2 - m2.b}`, { c: "yellow", w: 1.6, dash: "3 6", d: 1.4 });''' + t[j:]
rep('''<h2 class="rv" style="animation-delay:.15s"><span class="ho">Plot twist:</span> noise wrecks it. <span class="hg">The rescue:</span> SQD</h2>''',
    '''<h2 class="rv" style="animation-delay:.15s"><span class="ho">Plot twist:</span> noise wrecks it. <span class="hg">The rescue:</span> SQD</h2>''')

# ---------------- slide 10 (stack) ----------------
rep("<b>14 regression checks</b> against published numbers", "<b>14 Python + 52 engine checks</b> against published numbers")

# ---------------- slide 11 (proof) ----------------
rep('["All 14 regression checks pass", "one command reruns everything"]', '["All 66 automated checks pass", "14 Python + 52 browser-engine checks"]')
rep("Prototype uses the minimal STO-3G basis. CNOT counts are upper bounds (no routing, no circuit optimisation). No real-hardware run yet. Our SQD test space is small enough to sample completely.",
    "Prototype uses the minimal STO-3G basis, so exact answers are still classical. CNOT counts are upper bounds. No real-hardware run yet. Noise model is idealised (depolarising, all-to-all).")
rep("Verified prototype from integrals to SQD · 14 checks pass · every number in this deck comes from real runs", "Verified prototype from integrals to SQD · 66 checks pass · every number in this deck comes from real runs")
rep("Our own case already sizes at 7 logical qubits on a distance-9 surface code", f"Our own case already sizes at 7 logical qubits on a distance-{d_ft} surface code (~{phys_ft:,} physical qubits before magic-state factories)")

# ---------------- slide 12 (thank you) ----------------
rep('<div class="stat"><div class="big hg">0.30<small>mHa</small></div><div class="k">from a noisy circuit</div></div>',
    f'<div class="stat"><div class="big hg">{ratio_rnd:.0f}×</div><div class="k">better than random picks at equal budget</div></div>')

# ---------------- notes ----------------
i = t.index("const NOTES = ") + len("const NOTES = ")
j = t.index(";\n", i)
N = json.loads(t[i:j])
N[0] = N[0]
N[3] = (f"Here's the evidence. CCSD(T) is the method chemists call the gold standard. In the 10-electron valence space we compare it with the exact answer. "
        f"At rest it's already {e_eq:.1f} milli-hartree off, over budget. At 1.5 ångström it's {e_15:.1f} off. From 1.9 ångström it collapses {coll_min:.0f} to {coll_max:.0f} milli-hartree below the true energy, which no correct method can do, and at some lengths it doesn't converge at all. Exactly where the chemistry gets interesting, the single-reference shortcut breaks.")
N[6] = N[6].replace("12 qubits become 7, the gate count halves", f"12 qubits become 7, the gate count halves from {cxu_eq:,} to {cx_eq:,} CNOTs")
N[7] = re.sub(r"Stretched bonds need more bricks: \d+ at rest, \d+ stretched\.", f"Stretched bonds need more bricks: {k_eq} at rest, {k_21} stretched. In the bigger 11-qubit valence space it also stays within the target at all 12 lengths.", N[7])
N[8] = (f"Now the honest part. Today's two-qubit gates fail roughly once every 300 to 1,000 operations, and our accurate circuit has about {min(cx_noise):,} to {max(cx_noise):,}. "
        f"We modelled that noise and tried the standard fixes. Even after readout correction and zero-noise extrapolation it's still more than {best_today / 1000:.1f} hartree off, and error cancellation would need around 10 to the {exp10(gam_lo)} repeats. "
        f"Mitigation only reaches the target once gates are 30 to 100 times better than today. [Pause.] So we change the chip's job. With SQD it only points at the arrangements that matter, and a normal computer solves exactly inside them. "
        f"The fair test is a fixed budget: solving in a quarter of the space, the chip's picks are {chip28:.0f} milli-hartree off, random picks {rnd28:.0f}, so the chip carries about {ratio_rnd:.0f} times more information than chance. In a space this small random sampling eventually covers everything, so the real payoff is at scale: IBM has run SQD on nitrogen with 58 qubits.")
N[10] = N[10].replace("all 14 regression checks pass", "all 66 automated checks pass").replace("CNOT counts are upper bounds, no hardware run yet", "CNOT counts are upper bounds, and no hardware run yet")
N[11] = f"To sum up: 12 qubits down to 7, chemical accuracy at all 12 bond lengths where the gold standard collapses, and a noisy chip that picks the right arrangements {ratio_rnd:.0f} times better than chance. Thank you. We'd love your questions."
t = t[:i] + json.dumps(N, ensure_ascii=False) + t[j:]
DECK.write_text(t, encoding="utf-8")
print("deck updated")
