// Arjuna web app: talks to the engine API (web/backend), renders live progress and results.
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const CHEM = 1.6;
  const R2_STEPS = [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2];
  const SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const supNum = n => (n < 0 ? "⁻" : "") + String(Math.abs(n)).split("").map(d => SUPS[d]).join("");
  const sci = (v, d = 1) => { if (v === null || v === undefined || !isFinite(v)) return "–"; if (v === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e; return (Math.abs(e) < 3 ? v.toPrecision(d + 1) : `${m.toFixed(d)} × 10${supNum(e)}`); };
  const sciExp = v => { const e = Math.floor(Math.log10(v)); return `10${supNum(e)}`; };
  const mha = (v, d) => { if (v === null || v === undefined || !isFinite(v)) return "–"; const a = Math.abs(v); const dd = d ?? (a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3); return (v > 0 ? "+" : v < 0 ? "−" : "") + a.toLocaleString("en-US", { minimumFractionDigits: dd, maximumFractionDigits: dd }); };
  const absm = (v, d) => mha(Math.abs(v), d).replace("+", "");
  const ang = r => `${(+r).toFixed(3)} Å`;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const store = { get(k, d) { try { const v = localStorage.getItem("arjuna." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem("arjuna." + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } } };

  // ------------------------------------------------------------------ API + jobs
  async function api(path, method = "GET", body) {
    const res = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    let data = null; try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
    return data;
  }

  class JobView {
    constructor(id, buttons) { this.el = $(id); this.buttons = buttons.map($); this.el.innerHTML = `<div class="msg"><span class="m">–</span><span class="t"></span></div><div class="bar"><i></i></div><button class="btn ghost cancel" type="button">Cancel</button><details><summary>Engine log</summary><pre></pre></details>`; this.el.querySelector(".cancel").onclick = () => this.jobId && api(`/api/jobs/${this.jobId}`, "DELETE").catch(() => {}); }
    start() { this.el.classList.add("on"); this.el.classList.remove("err"); this.buttons.forEach(b => b && (b.disabled = true)); this.el.querySelector(".cancel").hidden = false; this.set({ progress: 0, message: "Sending to engine…", seconds: 0 }); }
    set(j) {
      this.el.querySelector(".m").textContent = j.message || "";
      this.el.querySelector(".t").textContent = j.cached ? "cached" : (j.seconds !== null && j.seconds !== undefined ? `${(+j.seconds).toFixed(1)} s` : "");
      this.el.querySelector(".bar i").style.width = `${Math.round(100 * (j.progress || 0))}%`;
      if (j.logs) this.el.querySelector("pre").textContent = j.logs.map(l => `${String(l[0].toFixed(1)).padStart(6)} s  ${l[1]}`).join("\n");
    }
    end(ok, msg) { this.buttons.forEach(b => b && (b.disabled = false)); this.el.querySelector(".cancel").hidden = true; if (!ok) { this.el.classList.add("err"); this.el.querySelector(".m").textContent = msg; } }
  }

  async function runJob(type, params, view, onPartial) {
    view.start();
    try {
      let j = await api("/api/jobs", "POST", { type, params });
      view.jobId = j.id;
      while (j.status === "queued" || j.status === "running") {
        view.set(j);
        if (onPartial && j.partial) { try { onPartial(j.partial); } catch (e) { console.error(e); } }
        await new Promise(r => setTimeout(r, 450));
        j = await api(`/api/jobs/${j.id}`);
      }
      view.set(j);
      if (j.status !== "done") { view.end(false, j.status === "cancelled" ? "Cancelled." : `Engine error: ${j.error}`); return null; }
      view.end(true);
      refreshRuns();
      return j.result;
    } catch (e) { view.end(false, `Could not reach the engine: ${e.message}`); return null; }
  }

  // ------------------------------------------------------------------ chrome: tabs, status, text size
  const TABS = ["explore", "adapt", "noise", "sqd", "runs", "proof"];
  function show(tab) {
    if (!TABS.includes(tab)) tab = "explore";
    TABS.forEach(t => { $(t).hidden = t !== tab; $("tab-" + t).setAttribute("aria-selected", String(t === tab)); });
    store.set("tab", tab);
    if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
    if (tab === "runs") refreshRuns();
    redraw[tab] && redraw[tab]();
  }
  TABS.forEach(t => { $("tab-" + t).onclick = () => show(t); });
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  $("bigtext").onclick = () => { const on = document.documentElement.classList.toggle("big"); $("bigtext").setAttribute("aria-pressed", String(on)); store.set("big", on); };
  if (store.get("big", false)) { document.documentElement.classList.add("big"); $("bigtext").setAttribute("aria-pressed", "true"); }

  async function health() {
    const s = $("status");
    try {
      const h = await api("/api/health");
      s.className = "status ok";
      $("status-t").textContent = `Engine online · ${h.workers} workers · ${h.active_jobs ? h.active_jobs + " running" : "idle"} · ${h.cache.geometries_solved} geometries ready`;
    } catch (e) { s.className = "status down"; $("status-t").textContent = "Engine offline: start it with run.sh"; }
  }
  health(); setInterval(health, 8000);

  function seg(id, onChange) {
    const el = $(id);
    el.addEventListener("click", ev => { const b = ev.target.closest("button"); if (!b) return; el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); onChange && onChange(b.dataset.v); });
    return () => el.querySelector('[aria-pressed="true"]').dataset.v;
  }
  function slider(id, outId, fmt) { const el = $(id), o = $(outId); const f = () => { o.textContent = fmt(+el.value); }; el.addEventListener("input", f); f(); return () => +el.value; }

  let REF = null;
  const redraw = {};

  // ------------------------------------------------------------------ 1. explorer
  const exSpace = seg("ex-space", () => { redraw.explore(); });
  const exR = slider("ex-r", "ex-rv", ang);
  const computed = { cas66: {}, fv108: {} };
  let exLast = null;
  const exView = new JobView("ex-job", ["ex-run", "sc-run"]);
  const GRID = [0.9, 1.0, 1.098, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0];
  $("ex-chips").innerHTML = GRID.map(g => `<button type="button" data-r="${g}">${g.toFixed(g === 1.098 ? 3 : 2)}</button>`).join("");
  $("ex-chips").onclick = ev => { const b = ev.target.closest("button"); if (!b) return; $("ex-r").value = b.dataset.r; $("ex-r").dispatchEvent(new Event("input")); exSolve(); };
  $("ex-r").addEventListener("input", () => { redraw.explore(); });
  $("ex-form").onsubmit = ev => { ev.preventDefault(); exSolve(); };

  async function exSolve() {
    const space = exSpace(), r = exR();
    const res = await runJob("energies", { r, space }, exView);
    if (!res) return;
    computed[space][res.r] = res; exLast = res; renderEx(res); redraw.explore();
  }
  $("sc-run").onclick = async () => {
    const a = +$("sc-from").value, b = +$("sc-to").value, n = Math.max(2, Math.min(40, +$("sc-n").value || 2));
    const list = Array.from({ length: n }, (_, i) => +(a + (b - a) * i / (n - 1)).toFixed(3));
    const space = exSpace();
    const res = await runJob("pes_scan", { space, r_list: list, with_adapt: $("sc-adapt").checked }, exView, p => { (p.rows || []).forEach(row => { computed[space][row.r] = Object.assign(computed[space][row.r] || {}, row); }); redraw.explore(); });
    if (res) { res.rows.forEach(row => { computed[space][row.r] = Object.assign(computed[space][row.r] || {}, row); }); redraw.explore(); }
  };

  function renderEx(d) {
    $("ex-exact").textContent = d.exact.toFixed(6);
    const e = d.err_mHa.ccsd_t, el = $("ex-cc");
    el.className = "v " + (!d.cc_converged ? "bad" : Math.abs(e) <= CHEM ? "good" : e < 0 ? "bad" : "warn");
    el.innerHTML = d.cc_converged ? `${mha(e)}<small>mHa</small>` : "no answer";
    $("ex-cc-k").innerHTML = `CCSD(T) error · ${esc(d.cc_verdict)}`;
    $("ex-hf").innerHTML = `${mha(d.err_mHa.hf)}<small>mHa</small>`;
    $("ex-q").innerHTML = `${d.qubits} → ${d.qubits_tapered}`;
    $("ex-q-k").textContent = `Qubits after tapering · ${d.pauli_terms_tapered} Pauli terms`;
    renderWF($("ex-wf"), d.wavefunction, d.space === "fv108" ? 8 : 6);
    const w = d.wavefunction, hfw = (100 * w.hf_weight).toFixed(1);
    const story = [];
    story.push(w.hf_weight > .8 ? `<p>One arrangement holds <b>${hfw}%</b> of the molecule here, so single-reference shortcuts mostly work.</p>` : `<p>The best single arrangement holds only <b>${hfw}%</b>, and <b>${w.n_over_1pct}</b> arrangements each hold over 1%. This is static correlation, where single-reference methods struggle.</p>`);
    if (!d.cc_converged) story.push(`<p>CCSD(T) <b>did not converge</b> at ${ang(d.r)}: the gold standard returns no answer.</p>`);
    else if (e < -CHEM) story.push(`<p>CCSD(T) is <b>${absm(e)} mHa below</b> the exact energy. Coupled cluster is not variational, so it can undershoot; a correct method cannot.</p>`);
    else if (Math.abs(e) > CHEM) story.push(`<p>CCSD(T) is <b>${absm(e)} mHa</b> off, ${(Math.abs(e) / CHEM).toFixed(1)}× the 1.6 mHa budget.</p>`);
    else story.push(`<p>CCSD(T) is within chemical accuracy here (${mha(e)} mHa).</p>`);
    story.push(`<p class="hint">Exact answer from ${w.sector_dim.toLocaleString()} arrangements · solved in ${d.seconds}s.</p>`);
    $("ex-story").innerHTML = story.join("");
  }

  function renderWF(el, w, nOrb) {
    const max = w.top[0].weight;
    el.innerHTML = w.top.map(t => `<div class="r"><div class="occ" title="${t.is_hf ? "Hartree–Fock arrangement" : ""}"><div class="a">${t.alpha.map(b => `<i class="${b ? "on" : ""}"></i>`).join("")}</div><div class="b">${t.beta.map(b => `<i class="${b ? "on" : ""}"></i>`).join("")}</div></div><div class="wbar${t.is_hf ? "" : " warn"}"><i style="width:${(100 * t.weight / max).toFixed(1)}%"></i></div><span class="p">${(100 * t.weight).toFixed(t.weight > .1 ? 1 : 2)}%</span></div>`).join("")
      + `<p class="cap">Each row is one way to seat the electrons in ${nOrb} orbitals: top row spin up (blue), bottom row spin down (pink), lowest-energy orbital on the left. The Hartree–Fock arrangement has the blue bar.</p>`;
  }

  redraw.explore = function () {
    if (!REF) return;
    const sp = exSpace(), ref = REF.pes[sp] || [], comp = Object.values(computed[sp]).sort((a, b) => a.r - b.r);
    const all = ref.map(r => r.exact).concat(comp.map(c => c.exact));
    const lo = Math.min(...all, ...ref.filter(r => r.ccsd_t !== null).map(r => r.ccsd_t)) - .03, hi = Math.max(...ref.map(r => r.hf)) + .03;
    const r0 = exR();
    Plot.chart($("ex-pes"), {
      x: { min: .8, max: 3.2, label: "N–N distance (Å)", ticks: [1, 1.5, 2, 2.5, 3], fmt: v => v.toFixed(1) },
      y: { min: lo, max: hi, label: "energy (hartree)", fmt: v => v.toFixed(2) }, margin: { l: 92 },
      vlines: [{ x: r0, color: "--pink", label: ang(r0) }],
      tipX: v => ang(v), tipY: v => v.toFixed(6) + " Ha",
      series: [
        { name: "Hartree–Fock (one arrangement)", color: "--faint", dash: "6 5", points: ref.map(r => [r.r, r.hf]), marker: false },
        { name: "exact", color: "--chalk", points: ref.map(r => [r.r, r.exact]), size: 3 },
        { name: "CCSD(T)", color: "--orange", points: ref.map(r => [r.r, r.ccsd_t]), size: 4 },
        { name: "CCSD(T) below exact", color: "--pink", line: false, marker: "tri", size: 7, hideLegend: true, points: ref.filter(r => r.ccsd_t !== null && r.ccsd_t < r.exact - .0016).map(r => [r.r, r.ccsd_t, { tip: r.ccsd_t.toFixed(6) + " Ha (below exact)" }]) },
        { name: "ADAPT-VQE", color: "--cyan", line: false, marker: "square", points: ref.filter(r => r.adapt).map(r => [r.r, r.adapt]), size: 4 },
        { name: "computed now", color: "--pink", line: false, marker: "ring", size: 7, points: comp.map(c => [c.r, c.exact]) },
      ],
    });
    const ccPts = [], ncPts = [], belowPts = [];
    ref.forEach(r => { if (r.ccsd_t === null) ncPts.push([r.r, 600, { marker: "x", tip: "no convergence" }]); else { const e = 1e3 * (r.ccsd_t - r.exact); (e < -CHEM ? belowPts : ccPts).push([r.r, Math.max(Math.abs(e), .011), { tip: mha(e) + " mHa" + (e < -CHEM ? " (below exact)" : "") }]); } });
    const compPts = comp.filter(c => c.err_mHa && c.err_mHa.ccsd_t !== null && c.err_mHa.ccsd_t !== undefined).map(c => [c.r, Math.max(Math.abs(c.err_mHa.ccsd_t), .011), { tip: mha(c.err_mHa.ccsd_t) + " mHa" }]);
    const adPts = comp.filter(c => c.adapt).map(c => [c.r, Math.max(Math.abs(c.adapt.final_err_mHa), .011)]);
    Plot.chart($("ex-err"), {
      width: 560, height: 340,
      x: { min: .8, max: 3.2, label: "N–N distance (Å)", ticks: [1, 1.5, 2, 2.5, 3], fmt: v => v.toFixed(1) },
      y: { min: .01, max: 1000, log: true, label: "|error| (mHa)" },
      bands: [{ axis: "y", from: .01, to: CHEM, color: "--green", opacity: .1 }],
      hlines: [{ y: CHEM, label: "1.6 mHa target", below: true }],
      vlines: [{ x: r0, color: "--pink" }],
      tipX: v => ang(v), tipY: v => v.toPrecision(3) + " mHa",
      extraLegend: [{ text: "▼ below exact", color: "--pink" }, { text: "× no convergence", color: "--pink" }],
      series: [
        { name: "CCSD(T)", color: "--orange", points: ccPts },
        { name: "no convergence", color: "--pink", line: false, marker: "x", points: ncPts, hideLegend: true, size: 6 },
        { name: "below exact", color: "--pink", line: false, marker: "tri", points: belowPts, hideLegend: true, size: 7 },
        { name: "ADAPT-VQE", color: "--cyan", points: ref.filter(r => r.adapt).map(r => [r.r, Math.max(Math.abs(1e3 * (r.adapt - r.exact)), .011)]) },
        { name: "CCSD(T) computed now", color: "--pink", line: false, marker: "ring", size: 7, points: compPts },
        { name: "ADAPT computed now", color: "--cyan", line: false, marker: "ring", size: 7, points: adPts, hideLegend: !adPts.length },
      ],
    });
  };

  // ------------------------------------------------------------------ 2. ADAPT
  const adSpace = seg("ad-space", v => { $("ad-max").value = v === "fv108" ? 120 : 80; $("ad-max").dispatchEvent(new Event("input")); });
  const adR = slider("ad-r", "ad-rv", ang);
  const adMax = slider("ad-max", "ad-maxv", v => String(v));
  const adView = new JobView("ad-job", ["ad-run"]);
  let adData = null, adTrace = [];
  $("ad-form").onsubmit = async ev => {
    ev.preventDefault(); adTrace = []; adData = null; redraw.adapt();
    const res = await runJob("adapt", { r: adR(), space: adSpace(), max_ops: adMax() }, adView, p => { adTrace = p.trace || []; redraw.adapt(); });
    if (!res) return;
    adData = res; renderAd(res); redraw.adapt();
  };
  function renderAd(d) {
    $("ad-k").innerHTML = d.k_chem_acc ? `${d.k_chem_acc}<small>bricks</small>` : "not reached";
    $("ad-k").className = "v " + (d.k_chem_acc ? "good" : "bad");
    const c = d.circuit_at_chem_acc;
    $("ad-cx").innerHTML = c ? `${c.cx.toLocaleString()}` : "–";
    $("ad-cx-k").textContent = c ? `CNOTs at that point (${c.cx_untapered.toLocaleString()} without tapering)` : "CNOTs at that point";
    $("ad-final").innerHTML = `${sci(Math.abs(d.final_err_mHa), 1)}<small>mHa</small>`;
    $("ad-s2").textContent = sci(d.iterations[d.iterations.length - 1].s2, 1);
    $("ad-ops").tBodies[0].innerHTML = d.iterations.slice(1).map(it => `<tr${d.k_chem_acc === it.k ? ' class="sel"' : ""}><td class="num">${it.k}</td><td><code>${esc(it.op)}</code></td><td class="num">${it.err_mHa.toPrecision(3)}</td></tr>`).join("");
  }
  redraw.adapt = function () {
    const pts = adData ? adData.iterations.map(it => [it.k, Math.max(it.err_mHa, 1e-5), { tip: `${it.err_mHa.toPrecision(3)} mHa · ${it.op}` }]) : adTrace.map(t => [t.iter, Math.max(t.err_mHa, 1e-5), { tip: `${t.err_mHa.toPrecision(3)} mHa · ${t.op}` }]);
    const n = Math.max(10, pts.length ? pts[pts.length - 1][0] : 10);
    Plot.chart($("ad-chart"), {
      x: { min: 0, max: n, label: "bricks (ADAPT iterations)" },
      y: { min: 1e-5, max: 2000, log: true, label: "error vs exact (mHa)" },
      bands: [{ axis: "y", from: 1e-5, to: CHEM, color: "--green", opacity: .08 }],
      hlines: [{ y: CHEM, label: "1.6 mHa = chemical accuracy", below: true, left: true }],
      vlines: adData && adData.k_chem_acc ? [{ x: adData.k_chem_acc, color: "--green", label: `${adData.k_chem_acc} bricks` }] : [],
      tipX: v => `brick ${v}`,
      series: [{ name: adData ? "error (physical energy)" : "error while running", color: "--cyan", points: pts, size: 3 }],
    });
  };

  // ------------------------------------------------------------------ 3. noise
  const nzR = slider("nz-r", "nz-rv", ang);
  const nzDepth = seg("nz-depth");
  const nzR2i = slider("nz-r2", "nz-r2v", v => sci(R2_STEPS[v], 1)), nzR2 = () => R2_STEPS[nzR2i()];
  const nzRo = slider("nz-ro", "nz-rov", v => (100 * v).toFixed(1) + "%");
  const nzView = new JobView("nz-job", ["nz-run", "nz-sweep"]);
  let nzSweep = null;
  const LABELS = { raw: "Raw", rem: "Readout fix (REM)", "rem+zne_lin": "+ ZNE linear", "rem+zne_rich": "+ ZNE Richardson", "rem+zne_exp": "+ ZNE exponential", "rem+zne_exp_asym": "+ ZNE exp. (known limit)" };
  $("nz-form").onsubmit = async ev => {
    ev.preventDefault();
    const res = await runJob("noise", { r: nzR(), depth: nzDepth(), r2: nzR2(), p_ro: nzRo() }, nzView);
    if (!res) return;
    const b = res.best;
    $("nz-best").innerHTML = `${absm(b.abs_err_mHa)}<small>mHa</small>`;
    $("nz-best").className = "v " + (b.within_target ? "good" : "bad");
    $("nz-best-k").innerHTML = `Best mitigated error (${esc(LABELS[b.method] || b.method)})${b.within_target ? '<span class="pill good">on target</span>' : '<span class="pill bad">off target</span>'}`;
    $("nz-gates").innerHTML = `${res.gates.cx.toLocaleString()}<small>CNOT</small>`;
    $("nz-pec").textContent = res.pec.shots_per_term ? sci(res.pec.shots_per_term, 1) : "–";
    $("nz-ideal").innerHTML = `${absm(res.noiseless_err_mHa, 3)}<small>mHa</small>`;
    const lmin = Math.log10(.1), lmax = Math.log10(5000);
    $("nz-abl").innerHTML = Object.entries(res.ablation).map(([k, v]) => { const e = v.err_mHa; if (e === null) return `<div class="b"><span>${LABELS[k]}</span><div class="wbar"></div><span class="num">no fit</span></div>`; const w = Math.max(2, 100 * (Math.log10(Math.max(Math.abs(e), .1)) - lmin) / (lmax - lmin)); return `<div class="b"><span>${LABELS[k]}</span><div class="wbar${Math.abs(e) <= CHEM ? "" : " warn"}"><i style="width:${w.toFixed(1)}%"></i></div><span class="num">${mha(e)} mHa</span></div>`; }).join("")
      + `<p class="hint">Bars on a log scale. Target: within 1.6 mHa. Circuit: ${res.k} bricks at ${ang(res.r)}; noise ${sci(res.noise_model.r2)} two-qubit, ${sci(res.noise_model.r1)} one-qubit, ${(100 * res.noise_model.p_ro).toFixed(1)}% readout.</p>`;
  };
  $("nz-sweep").onclick = async () => {
    nzSweep = null;
    const res = await runJob("noise_sweep", { r: nzR(), depth: nzDepth(), p_ro: nzRo() }, nzView, p => { nzSweep = { sweep: p.sweep, partial: true }; redraw.noise(); });
    if (!res) return;
    nzSweep = res; redraw.noise();
    const ft = res.fault_tolerant, ok = res.max_r2_within_target;
    $("nz-ft").innerHTML = `<p>Mitigation reaches 1.6 mHa up to a gate error of <b>${ok ? sci(ok) : "none in the sweep"}</b>${ok ? ` (today ≈ 3 × 10⁻³, so ${Math.round(3e-3 / ok)}× better gates)` : ""}.</p>` + (ft ? `<p>Error-corrected instead: <b>${ft.logical_qubits} logical qubits</b>, surface-code distance <b>${ft.code_distance}</b>, about <b>${ft.physical_qubits_data_and_routing.toLocaleString()}</b> physical qubits before magic-state factories, <b>${ft.t_gates.toLocaleString()}</b> T gates.</p>` : "");
  };
  redraw.noise = function () {
    const series = [], s = nzSweep && nzSweep.sweep;
    if (s && s.length) {
      series.push({ name: "readout fix only", color: "--faint", points: s.map(x => [x.r2, Math.max(Math.abs(x.rem), .1)]) });
      series.push({ name: "+ best ZNE (this run)", color: "--cyan", points: s.map(x => [x.r2, Math.max(x.best_zne ?? NaN, .1)]) });
    } else if (REF && REF.noise) {
      for (const [g, c] of [["1.098", "--cyan"], ["2.1", "--pink"]]) { const sw = REF.noise[g] && REF.noise[g].sweep; if (sw) series.push({ name: `published, ${g} Å`, color: c, points: sw.map(x => [x.r2, Math.max(Math.min(Math.abs(x["rem+zne_rich"]), Math.abs(x["rem+zne_exp"])), .1)]) }); }
    }
    Plot.chart($("nz-chart"), {
      width: 620, height: 330,
      x: { min: 1e-6, max: 1e-2, log: true, label: "two-qubit gate error" },
      y: { min: .1, max: 5000, log: true, label: "|error| (mHa)" },
      bands: [{ axis: "x", from: 1e-3, to: 3e-3, color: "--orange", label: "today", opacity: .14 }, { axis: "x", from: 3e-5, to: 1e-4, color: "--green", label: "needed", opacity: .14 }],
      hlines: [{ y: CHEM, label: "1.6 mHa", below: true }],
      tipX: v => sci(v), tipY: v => v.toPrecision(3) + " mHa",
      series,
    });
  };

  // ------------------------------------------------------------------ 4. SQD
  const sqSpace = seg("sq-space");
  const sqR = slider("sq-r", "sq-rv", ang);
  const sqK = slider("sq-k", "sq-kv", v => `${v} brick${v > 1 ? "s" : ""}`);
  const sqR2i = slider("sq-r2", "sq-r2v", v => sci(R2_STEPS[v], 1)), sqR2 = () => R2_STEPS[sqR2i()];
  const sqView = new JobView("sq-job", ["sq-run"]);
  let sqData = null;
  $("sq-form").onsubmit = async ev => {
    ev.preventDefault();
    sqData = null;
    const res = await runJob("sqd", { r: sqR(), space: sqSpace(), bricks: sqK(), r2: sqR2(), shots: +$("sq-shots").value, seed: +$("sq-seed").value || 0 }, sqView, p => { sqData = { budget_rows: p.rows, partial: true }; redraw.sqd(); });
    if (!res) return;
    sqData = res;
    const rows = res.budget_rows, b = $("sq-budget");
    b.max = rows.length - 1; b.disabled = false;
    const i25 = rows.reduce((best, r, i) => Math.abs(r.frac - .25) < Math.abs(rows[best].frac - .25) ? i : best, 0);
    b.value = i25;
    $("sq-vqe").innerHTML = `${absm(res.noisy_vqe_err_mHa)}<small>mHa</small>`; $("sq-vqe").className = "v bad";
    $("sq-note").innerHTML = `<p>${res.gates.cx} CNOTs, ${res.shots.toLocaleString()} shots at ${sci(res.noise_model.r2)} gate error. <b>${(100 * res.frac_right_electron_count.chip).toFixed(0)}%</b> of chip shots had the right electron count before repair (random bitstrings: ${(100 * res.frac_right_electron_count.random).toFixed(0)}%).</p><p>The chip found <b>${res.distinct_strings.chip}</b> of ${res.n_strings} orbital patterns; random bitstrings found ${res.distinct_strings.random}. Using everything the chip found gives ${absm(res.uncapped.err_mHa, 3)} mHa, but in a space this small random sampling gets there too, so compare at a fixed budget.</p><p class="hint">${esc(res.note)}</p>`;
    renderWF($("sq-top"), { top: res.top_samples.map(t => ({ ...t, weight: t.count / res.shots, is_hf: t.right_count })) }, res.space === "fv108" ? 8 : 6);
    $("sq-top").querySelector(".cap").textContent = "Raw measured arrangements, most frequent first. Blue bar: right electron count; orange: broken by noise and repaired before solving.";
    sqBudget();
  };
  $("sq-budget").addEventListener("input", () => sqBudget());
  function sqBudget() {
    if (!sqData || sqData.partial) return;
    const row = sqData.budget_rows[+$("sq-budget").value];
    $("sq-bv").textContent = `${row.K} patterns → ${row.dets.toLocaleString()} of ${sqData.space_dets.toLocaleString()} arrangements (${(100 * row.frac).toFixed(0)}%)`;
    const set = (id, v) => { $(id).innerHTML = v === null || v === undefined ? "–" : `${absm(v)}<small>mHa</small>`; };
    set("sq-chip", row.chip); set("sq-rnd", row.random); set("sq-cls", row.classical);
    redraw.sqd();
  }
  redraw.sqd = function () {
    const series = [], v = [];
    const mk = (rows, key) => rows.filter(r => r[key] !== null && r[key] !== undefined).map(r => [r.frac, Math.max(Math.abs(r[key]), .1), { tip: `${mha(r[key])} mHa · K=${r.K}` }]);
    if (sqData && sqData.budget_rows) {
      const rows = sqData.budget_rows;
      series.push({ name: "random picks", color: "--orange", points: mk(rows, "random") }, { name: "classical ranking", color: "--faint", points: mk(rows, "classical") }, { name: "chip-picked", color: "--green", points: mk(rows, "chip"), size: 5 });
      if (!sqData.partial) v.push({ x: rows[+$("sq-budget").value].frac, color: "--yellow", label: "your budget" });
    } else if (REF && REF.sqd_capped) {
      const rows = REF.sqd_capped.rows.filter(r => Math.abs(r.r - 2.1) < 1e-6).map(r => ({ frac: r.frac, K: r.K, chip: r.noisy.median, random: r.random.median, classical: r.hf_ranked }));
      series.push({ name: "random (published, median of 10)", color: "--orange", points: mk(rows, "random") }, { name: "classical ranking", color: "--faint", points: mk(rows, "classical") }, { name: "chip (published, 2.1 Å)", color: "--green", points: mk(rows, "chip") });
    }
    Plot.chart($("sq-chart"), {
      x: { min: 0, max: 1, label: "share of all arrangements solved exactly", ticks: [0, .25, .5, .75, 1], fmt: x => Math.round(100 * x) + "%" },
      y: { min: .1, max: 2000, log: true, label: "|error| (mHa)" },
      bands: [{ axis: "y", from: .1, to: CHEM, color: "--green", opacity: .08 }],
      hlines: [{ y: CHEM, label: "1.6 mHa target", below: true, left: true }], vlines: v,
      tipX: x => Math.round(100 * x) + "% of space", series,
    });
  };

  // ------------------------------------------------------------------ 5. runs
  const SUMMARY = p => Object.entries(p || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.length}]` : typeof v === "number" && v < .01 && v > 0 ? sci(v) : v}`).join(" · ");
  let selJob = null;
  async function refreshRuns() {
    if ($("runs").hidden) return;
    try {
      const list = await api("/api/jobs");
      $("rn-table").tBodies[0].innerHTML = list.length ? list.map(j => `<tr class="click${j.id === selJob ? " sel" : ""}" data-id="${j.id}"><td><code>${j.type}</code></td><td>${esc(SUMMARY(j.params))}</td><td><span class="st ${j.status}">${j.status}${j.cached ? " · cache" : ""}</span></td><td class="num">${j.seconds ?? "–"}</td></tr>`).join("") : `<tr><td colspan="4" class="empty">No jobs yet in this session.</td></tr>`;
    } catch (e) { /* engine offline: status pill shows it */ }
  }
  setInterval(refreshRuns, 3000);
  $("rn-table").onclick = async ev => { const tr = ev.target.closest("tr[data-id]"); if (!tr) return; selJob = tr.dataset.id; const j = await api(`/api/jobs/${selJob}`); const txt = JSON.stringify(j, null, 1); $("rn-detail").textContent = txt.length > 200000 ? txt.slice(0, 200000) + "\n…" : txt; refreshRuns(); };
  $("rn-clear").onclick = async () => { const r = await api("/api/cache", "DELETE"); $("rn-detail").textContent = `Removed ${r.removed} cached results. Jobs will recompute.`; };

  // ------------------------------------------------------------------ 6. proof
  const pfView = new JobView("pf-job", ["pf-run"]);
  function renderSuites(s) {
    $("pf-suites").innerHTML = s.map(x => `<div class="card suite"><h4>${esc(x.suite)} <span class="pill ${x.ok ? "good" : "bad"}">${x.ok ? "pass" : "fail"}</span><small class="hint">${x.kind} · ${x.pass_marks} PASS marks · ${x.seconds}s</small></h4><details${x.ok ? "" : " open"}><summary class="hint">Output</summary><pre class="pre">${esc(x.output.join("\n"))}</pre></details></div>`).join("");
  }
  $("pf-run").onclick = async () => { const res = await runJob("tests", {}, pfView, p => renderSuites(p.suites || [])); if (res) { renderSuites(res.suites); pfView.el.querySelector(".m").textContent = res.all_ok ? "All suites pass." : "Some checks failed: see the output."; } };

  // ------------------------------------------------------------------ boot
  api("/api/reference").then(r => { REF = r; Object.values(redraw).forEach(f => f()); }).catch(() => { $("ex-story").innerHTML = "<p>Engine offline. Start it with <code>./run.sh</code> and reload.</p>"; });
  show(location.hash.slice(1) || store.get("tab", "explore"));
})();
