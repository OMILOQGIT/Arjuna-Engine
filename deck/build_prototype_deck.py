"""Build the 5-slide prototype walkthrough deck by reusing the main deck's design system
(chalk toolkit, atom-burst transitions, chrome) with new slides. Output: deck/Arjuna-Prototype-Walkthrough.html"""
from pathlib import Path

D = Path(__file__).resolve().parent
src = (D / "Team-Arjuna-Chakravyuh-QC02.html").read_text(encoding="utf-8")


def between(a, b, start=0):
    i = src.index(a, start)
    j = src.index(b, i)
    return src[i:j]


head = src[:src.index("<!-- 1 · H · TITLE -->")]
head = head.replace("<title>Breaking N≡N · Team Arjuna</title>", "<title>Arjuna Engine Walkthrough</title>")
head = head.replace('aria-label="Breaking N≡N, Team Arjuna"', 'aria-label="The Arjuna engine, prototype walkthrough"')
chrome = between('<canvas id="fx"', "<script>")
js_tools = between("const NS = ", "function s1() {")
js_tail = between("function overlayFor(container)", "function s7() {") + between("function wrapText(", "function s13() {") \
    + between("/* sketched borders around HTML boxes */", "function build() {")

SLIDES = r'''
<!-- 1 · H · WHAT IT IS -->
<section class="slide" data-el="1" data-sym="H" data-name="Hydrogen" data-tag="The prototype">
  <div class="kicker rv" style="animation-delay:.1s">Team Arjuna · QC-02 <b>· prototype walkthrough · Chakravyuh hackathon</b></div>
  <div class="row" style="align-items:center;gap:24px">
    <div class="col" style="width:780px;gap:26px">
      <h1 class="rv" style="font-size:128px;animation-delay:.25s">The <span class="hl">Arjuna</span> engine</h1>
      <p class="txt rv" style="font-size:29px;line-height:1.38;animation-delay:.45s">A quantum-chemistry engine you drive from a browser. Pick any N≡N bond length and watch it <b>solve the molecule</b>, <b>build the quantum circuit</b>, run it on a <b>simulated noisy chip</b> and <b>rescue the answer</b>, live.</p>
      <div class="rv" style="display:flex;gap:48px;margin-top:12px;animation-delay:.7s">
        <div class="stat"><div class="big hl">any</div><div class="k">bond length from<br>0.80 to 3.20 Å</div></div>
        <div class="stat"><div class="big hp">6</div><div class="k">live stations,<br>one engine</div></div>
        <div class="stat"><div class="big hg">87</div><div class="k">automated checks,<br>all passing</div></div>
      </div>
    </div>
    <svg class="d" id="w-art" width="700" height="560" viewBox="0 0 700 560" aria-label="Sketch of a browser window running the engine on a nitrogen molecule"></svg>
  </div>
  <p class="sm rv" style="animation-delay:1s">Press → or click to move · N for speaker notes · F for fullscreen</p>
</section>

<!-- 2 · He · ARCHITECTURE -->
<section class="slide" data-el="2" data-sym="He" data-name="Helium" data-tag="How it fits together">
  <div class="kicker rv">How it works <b>· the architecture</b></div>
  <h2 class="rv" style="animation-delay:.15s;font-size:58px">Your click becomes a <span class="hy">job</span>; the <span class="hl">real engine</span> answers it</h2>
  <svg class="d" id="arch" width="1316" height="395" viewBox="0 0 1432 430" style="align-self:center" aria-label="Architecture: browser, API, job manager, engine, result cache"></svg>
  <div class="row" style="gap:36px;flex:none">
    <div class="note plain" data-sk="yellow" style="flex:1"><span class="lab">In plain words</span><p class="txt" style="font-size:20px">The website never makes up a number. Each button sends a request; the server checks it, runs our verified code on a worker and <b>streams progress back so the charts fill in while it computes</b>. Finished answers are saved, so asking again is instant.</p></div>
    <div class="note deep" data-sk="cyan" style="flex:1"><span class="lab">Under the hood</span><p>Flask API: <span class="mono">POST /api/jobs</span> returns a job id; <span class="mono">GET /api/jobs/&lt;id&gt;</span> gives status, progress, log and partial results; <span class="mono">DELETE</span> cancels. Two worker threads. Results cached on disk by job type + parameters + engine version. Plain Python 3 with NumPy, SciPy and Flask, no build step.</p></div>
  </div>
</section>

<!-- 3 · Li · ONE CLICK -->
<section class="slide" data-el="3" data-sym="Li" data-name="Lithium" data-tag="One click, end to end">
  <div class="kicker rv">How it works <b>· following one request</b></div>
  <h2 class="rv" style="animation-delay:.15s">One click, end to end: <span class="hp">"solve at 2.10 Å"</span></h2>
  <div class="row" style="gap:40px">
    <svg class="d" id="trace" width="930" height="600" viewBox="0 0 930 600" aria-label="Timeline of one request through the engine"></svg>
    <div class="col" style="flex:1;gap:22px">
      <div class="rv" style="display:flex;gap:40px;animation-delay:.5s">
        <div class="stat"><div class="big hy">22<small>s</small></div><div class="k">first time</div></div>
        <div class="stat"><div class="big hg">0<small>s</small></div><div class="k">every repeat (cached)</div></div>
      </div>
      <div class="note deep" data-sk="cyan"><span class="lab">What comes back</span><p class="mono" style="font-size:15px;line-height:1.55;color:var(--dim);display:block">{ "exact": −107.448604,<br>&nbsp;&nbsp;"ccsd_t": −107.592661,<br>&nbsp;&nbsp;"err_mHa": { "ccsd_t": −144.06 },<br>&nbsp;&nbsp;"cc_verdict": "collapsed below<br>&nbsp;&nbsp;&nbsp;&nbsp;the exact energy",<br>&nbsp;&nbsp;"qubits": 16, "qubits_tapered": 11,<br>&nbsp;&nbsp;"wavefunction": { "hf_weight": 0.192, … } }</p></div>
      <div class="note plain" data-sk="yellow"><span class="lab">Why it's trustworthy</span><p class="txt" style="font-size:19px">At the 12 bond lengths in our study, the website returns <b>exactly the published numbers</b>: energies, circuit sizes and noise results. A test checks this every run.</p></div>
    </div>
  </div>
</section>

<!-- 4 · Be · STATIONS -->
<section class="slide" data-el="4" data-sym="Be" data-name="Beryllium" data-tag="Six stations">
  <div class="kicker rv">What you can do <b>· the six stations</b></div>
  <h2 class="rv" style="animation-delay:.15s">Six stations, <span class="hl">one engine</span>, every number live</h2>
  <div class="cards" style="grid-template-columns:repeat(3,1fr);grid-template-rows:1fr 1fr;flex:1;gap:30px 40px">
    <div class="card" data-sk="chalk"><span class="n">1</span><h3>Bond explorer</h3><p class="m">Pick any bond length. See the exact energy, the gold standard's error and who is in the electron crowd.</p><span class="t">Hartree–Fock · Jordan–Wigner · tapering · exact · CCSD(T)</span><span class="v">CCSD(T) collapses −144 mHa at 2.1 Å</span></div>
    <div class="card" data-sk="cyan"><span class="n">2</span><h3>ADAPT-VQE</h3><p class="m">Watch the circuit grow one brick at a time until it lands within 1.6 mHa.</p><span class="t">singlet-pinned ADAPT · BFGS · circuit compiler</span><span class="v">35 bricks, 1,728 CNOTs at 2.1 Å</span></div>
    <div class="card" data-sk="orange"><span class="n">3</span><h3>Noise lab</h3><p class="m">Turn the chip's error up or down, try every fix, sweep to find the gate quality chemistry needs.</p><span class="t">density matrix · readout fix · ZNE · PEC · surface code</span><span class="v">needs gates 30–100× better</span></div>
    <div class="card" data-sk="green"><span class="n">4</span><h3>SQD rescue</h3><p class="m">The noisy chip picks arrangements; chip vs random vs classical at the same budget.</p><span class="t">sampling · configuration recovery · exact subspace solve</span><span class="v">9.6 mHa vs 160 random, at 25%</span></div>
    <div class="card" data-sk="chalk"><span class="n">5</span><h3>Runs</h3><p class="m">Every job with its inputs, progress log, timing and full result.</p><span class="t">job history · JSON · result cache</span><span class="v">repeat runs are instant</span></div>
    <div class="card" data-sk="green"><span class="n">6</span><h3>Proof</h3><p class="m">One button runs every test suite on the server and shows each result.</p><span class="t">Python engine · web API · browser engine</span><span class="v">87 checks pass</span></div>
  </div>
</section>

<!-- 5 · B · PROOF + RUN -->
<section class="slide" data-el="5" data-sym="B" data-name="Boron" data-tag="Proven, and how to run it">
  <div class="kicker rv">The build <b>· proof, and how to run it</b></div>
  <h2 class="rv" style="animation-delay:.15s">What's <span class="hg">proven</span>, and how to <span class="hl">run it yourself</span></h2>
  <div class="row" style="gap:44px">
    <svg class="d" id="checks5" width="640" height="470" viewBox="0 0 640 470" aria-label="Checklist of passing test suites"></svg>
    <div class="col" style="flex:1;gap:26px">
      <div class="note deep" data-sk="cyan"><span class="lab">Run it</span>
        <p style="font-size:19px">Unzip, then one command: <span class="mono">./run.sh</span> (Windows: <span class="mono">run.bat</span>) → open <span class="mono">localhost:8000</span>. Python 3.10+; installs NumPy, SciPy and Flask itself.</p>
        <p style="font-size:19px;margin-top:8px">Docker: <span class="mono">docker run -p 8000:8000 arjuna</span> · no server at all: <span class="mono">lab/arjuna-lab.html</span> runs offline in any browser.</p></div>
      <div class="note deep" data-sk="orange"><span class="lab" style="color:var(--orange)">Limits we state up front</span><p style="font-size:19px">Minimal STO-3G basis, so exact answers are still classical. Chip noise is simulated (depolarising, all-to-all). Built for a laptop demo, not public hosting. The port to IBM's tools is written but not yet run.</p></div>
      <div class="note plain" data-sk="yellow"><span class="lab">Next</span><p class="txt" style="font-size:20px">Realistic basis with PySCF, IBM noise models, IBM's SQD, one real IBM run, all behind <b>the same website</b>.</p></div>
    </div>
  </div>
</section>
'''

JS = r'''
const NOTES = [
  "This is what we built. The Arjuna engine is a quantum-chemistry engine you drive from a web browser. You pick a bond length for nitrogen, anywhere from 0.8 to 3.2 ångström, and it solves the molecule, builds the quantum circuit, runs it on a simulated noisy chip and rescues the answer, all live. Six stations, one engine, and 87 automated checks that all pass.",
  "Here's how it fits together. Every button in the browser sends a request to our server. The server checks it, turns it into a job and runs our verified Python engine on a worker thread. While it computes, the browser asks for progress about twice a second, so charts fill in live. When it's done, the answer is saved, so the same question is instant next time. The website never makes up a number: everything comes from the engine.",
  "Let's follow one click: solve at 2.1 ångström. The request goes to the server. Hartree–Fock walks out from equilibrium one step at a time, so it stays on the right solution. We zoom into ten electrons in eight orbitals, translate to qubits and use symmetry to go from 16 qubits to 11. Then the exact answer from 3,136 arrangements, and the gold standard CCSD(T). It comes back 144 milli-hartree below the exact energy: collapsed. 22 seconds the first time, instant after that. And at our 12 study bond lengths the website returns exactly the published numbers; a test checks that.",
  "Six stations. Bond explorer: any bond length, exact energy versus the gold standard. ADAPT-VQE: watch the circuit grow brick by brick, 35 bricks at 2.1 ångström. Noise lab: turn the chip's error up and down and see that today's chips need to be 30 to 100 times better for this circuit. SQD rescue: the noisy chip picks arrangements, compared fairly against random and classical picks at the same budget. Runs: every job with its full result. Proof: one button runs every test on the server.",
  "What's proven: integrals and Hartree–Fock against textbook values, coupled cluster against published references, tapering and ADAPT, the web API against our published study, and the browser engine against the Python engine. 87 checks, all passing. To run it: unzip, one command, open localhost 8000; or Docker; or the offline lab with no server at all. Limits up front: minimal basis, simulated noise, laptop server, and the IBM port isn't run yet. Next: a realistic basis, IBM noise models and one real IBM run, behind the same website. Thank you."
];

function w1() {
  const p = new Pad("w-art", { base: .5, step: .05 });
  p.rect(30, 40, 640, 440, { c: "chalk", w: 2.6 });
  p.line(30, 86, 670, 86, { c: "faint", w: 2, single: true });
  [58, 82, 106].forEach((x, i) => p.ell(x, 63, 8, 8, { c: ["pink", "yellow", "green"][i], w: 2, single: true }));
  p.rect(140, 52, 380, 24, { c: "faint", w: 1.6, single: true });
  p.text(156, 70, "localhost:8000", { size: 17, font: "mono", c: "dim" });
  drawN2(p, 190, 210, 190, 50);
  p.text(190, 300, "R = 2.10 Å", { anchor: "middle", size: 26, font: "hand", c: "pink", d: 1.3 });
  // mini error chart
  const X = lin(.9, 3.0, 350, 640), Y = logs(.3, 400, 330, 120);
  p.line(350, 330, 645, 330, { c: "faint", w: 1.8, single: true }); p.line(350, 330, 350, 112, { c: "faint", w: 1.8, single: true });
  p.ink(`M350 ${Y(1.6)}L645 ${Y(1.6)}`, { c: "green", w: 2, dash: "3 7", d: 1 });
  const cc = [[.9, .66], [1.098, 2.2], [1.35, 7.4], [1.5, 10.8], [1.7, .8]];
  p.poly(cc.map(([x, y]) => [X(x), Y(y)]), { c: "orange", w: 3, d: 1.2 });
  [[1.9, 64], [2.1, 144], [3.0, 280]].forEach(([x, y], i) => { const xx = X(x), yy = Y(y); p.ink(`M${xx - 8} ${yy - 7}L${xx + 8} ${yy - 7}L${xx} ${yy + 8}Z`, { c: "pink", w: 2.6, d: 1.6 + i * .1 }); });
  const ad = [[.9, .1], [1.35, 1.2], [1.7, .9], [2.1, .3], [2.7, .06], [3.0, .03]];
  p.poly(ad.map(([x, y]) => [X(x), Y(Math.max(y, .32))]), { c: "cyan", w: 3, d: 1.5 });
  p.text(512, 246, "▼ collapsed", { size: 18, font: "hand", c: "pink", d: 1.9 });
  // progress bar + tabs
  p.text(60, 380, "ADAPT-VQE · iteration 35 · 1,728 CNOTs", { size: 18, font: "mono", c: "cyan", d: 1.7 });
  p.rect(60, 398, 580, 18, { c: "faint", w: 1.6, single: true, d: 1.7 });
  p.hatch(62, 400, 470, 14, "cyan", { d: 1.9, o: .7 });
  ["explore", "ADAPT", "noise", "SQD", "runs", "proof"].forEach((t, i) => { p.rect(60 + i * 98, 432, 86, 32, { c: i === 1 ? "cyan" : "faint", w: 1.6, single: true, d: 2 + i * .05 }); p.text(103 + i * 98, 454, t, { anchor: "middle", size: 16, c: i === 1 ? "cyan" : "faint", d: 2.05 + i * .05 }); });
  p.arrow(350, 540, 350, 488, { c: "yellow", w: 2.6, d: 2.3 });
  p.text(370, 536, "the real engine answers every click", { size: 22, font: "hand", c: "yellow", d: 2.4 });
  orbitDots(p.svg, 190, 210, 150, 52, -15, 3, "pink", 7);
}

function w2() {
  const p = new Pad("arch", { base: .45, step: .05 });
  const B = (x, y, w, h, c, title, lines, d) => {
    p.rect(x, y, w, h, { c, w: 2.4, d });
    p.text(x + 18, y + 40, title, { size: 32, font: "hand", c, d: d + .1 });
    lines.forEach((l, i) => p.text(x + 18, y + 72 + i * 25, l, { size: 17, c: "dim", d: d + .15 + i * .04 }));
  };
  B(10, 60, 220, 180, "chalk", "Browser", ["6 stations", "live charts", "progress bar · cancel"], .5);
  B(320, 60, 235, 180, "yellow", "API", ["Flask web server", "POST /api/jobs", "GET /api/jobs/<id>"], .8);
  B(645, 60, 255, 180, "orange", "Job manager", ["checks every input", "2 worker threads", "progress · cancel"], 1.1);
  p.rect(990, 20, 430, 400, { c: "cyan", w: 2.6, d: 1.4 });
  p.text(1012, 62, "The engine (src/)", { size: 34, font: "hand", c: "cyan", d: 1.5 });
  const steps = ["Hartree–Fock on the right branch", "active space (6 or 8 orbitals)", "Jordan–Wigner → qubits", "Z₂ tapering: 12 → 7, 16 → 11", "ADAPT-VQE circuit", "noisy-chip simulation + fixes", "SQD: sample, recover, solve"];
  steps.forEach((s, i) => { const y = 92 + i * 45; p.rect(1012, y, 386, 36, { c: "faint", w: 1.5, single: true, d: 1.6 + i * .08 }); p.text(1026, y + 25, `${i + 1}  ${s}`, { size: 18, c: "chalk", d: 1.65 + i * .08 }); });
  B(645, 300, 255, 120, "green", "Result cache", ["same request → instant", "saved as JSON"], 1.9);
  B(320, 300, 235, 120, "faint", "Published study", ["results/*.json", "reference curves"], 2.1);
  p.arrow(234, 120, 316, 120, { c: "chalk", w: 2.4, label: "request", lc: "dim", ly: -12, d: 2.3 });
  p.arrow(559, 120, 641, 120, { c: "yellow", w: 2.4, label: "job", lc: "dim", ly: -12, d: 2.4 });
  p.arrow(904, 120, 986, 120, { c: "orange", w: 2.4, label: "run", lc: "dim", ly: -12, d: 2.5 });
  p.arrow(986, 200, 904, 200, { c: "cyan", w: 2.2, label: "progress", lc: "cyan", ly: 22, d: 2.6 });
  p.arrow(316, 200, 234, 200, { c: "yellow", w: 2.2, label: "updates", lc: "yellow", ly: 22, d: 2.7, dash: "6 6" });
  p.arrow(772, 242, 772, 296, { c: "green", w: 2.2, d: 2.8 });
  p.arrow(437, 296, 437, 242, { c: "faint", w: 2, d: 2.9 });
}

function w3() {
  const p = new Pad("trace", { base: .45, step: .05 });
  p.ink("M46 20C58 120 34 220 46 320S58 480 46 580", { c: "pink", w: 3, dur: 1.6 });
  const steps = [
    ["You click", "POST /api/jobs  {type: energies, r: 2.1, space: fv108}", "chalk"],
    ["Hartree–Fock walks out", "1.098 → 1.20 → 1.35 → 1.50 → 1.70 → 1.90 → 2.10 Å, staying on the symmetric solution", "orange"],
    ["Zoom + translate + shrink", "10 electrons in 8 orbitals → 16 qubits → symmetry removes 5 → 11 qubits, 815 Pauli terms", "cyan"],
    ["Exact answer + gold standard", "exact singlet from 3,136 arrangements · CCSD(T) with DIIS", "yellow"],
    ["Charts fill in", "CCSD(T) −144 mHa: collapsed below the exact energy · best arrangement holds only 19%", "pink"],
    ["Saved", "result cached by its inputs: the next identical click answers instantly", "green"],
  ];
  steps.forEach(([t, s, c], i) => {
    const y = 34 + i * 96;
    p.dot(46, y, 10, { c, glow: true, d: .7 + i * .22 });
    p.text(78, y + 9, t, { size: 31, font: "hand", c, d: .75 + i * .22 });
    wrapText(p, 78, y + 38, s, 820, { size: 18, c: "dim", d: .85 + i * .22 });
  });
}

function w5() {
  const p = new Pad("checks5", { base: .5, step: .12 });
  const items = [["Integrals + Hartree–Fock", "water and N₂ against textbook values, to 3 × 10⁻¹³ Ha · 4 checks"], ["Coupled cluster + qubit Hamiltonian", "Crawford references to 10⁻¹⁰ Ha, exact energies · 4 checks"], ["Tapering + ADAPT-VQE", "12- and 7-qubit runs agree to 10⁻⁹ Ha · 8 checks"], ["Web API", "same energies, circuits and noise as the published study · 19 checks"], ["Browser engine", "offline lab matches the Python engine · 52 checks"], ["All 87 checks pass", "run them from the Proof tab with one click"]];
  items.forEach(([a, b], i) => {
    const y = 14 + i * 74;
    p.rect(8, y, 32, 32, { c: "chalk", w: 2, single: true });
    p.ink(`M13 ${y + 16}L23 ${y + 28}L46 ${y - 6}`, { c: "green", w: 3.4 });
    p.text(64, y + 20, a, { size: 27, font: "hand", c: i === 5 ? "green" : "chalk" });
    p.text(64, y + 46, b, { size: 17, c: "faint" });
  });
}

function build() {
  R = mkRng(42);
  sketchBoxes();
  [w1, w2, w3, w5].forEach(f => { try { f(); } catch (err) { console.error(f.name, err); } });
  dust();
  const m = /^#s(\d+)$/.exec(location.hash);
  go(m ? +m[1] - 1 : 0);
}
let built = false;
const once = () => { if (!built) { built = true; build(); } };
(document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(once);
setTimeout(once, 2500);
'''

out = head + SLIDES + "\n" + chrome + "<script>\n" + js_tools + JS + js_tail + "</script>\n</body>\n</html>\n"
# js_tail ends right before the original build(); our JS defines build after helpers, so order: tools, slides JS, tail.
# The tail must come before build() is invoked; build() runs async (fonts.ready), so definition order is fine.
(D / "Arjuna-Prototype-Walkthrough.html").write_text(out, encoding="utf-8")
print("wrote", len(out), "bytes")
