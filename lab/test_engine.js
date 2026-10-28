// Node regression test: the JS engine must reproduce the verified Python engine.
// Run: node lab/test_engine.js results/lab_data.json
const fs = require("fs");
const E = require("./engine.js");
const D = JSON.parse(fs.readFileSync(process.argv[2] || "results/lab_data.json", "utf8"));
let fails = 0, passes = 0;
const check = (name, ok, info) => { if (ok) passes++; else fails++; console.log(`${ok ? "PASS" : "FAIL"} ${name}${info ? "  " + info : ""}`); };
const binom = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return Math.round(r); };
const strings = (n, k) => { const out = []; for (let s = 0; s < (1 << n); s++) if (E.popcount(s) === k) out.push(s); return out; };

// 1. ADAPT statevector energies at every iteration
for (const c of D.cas) {
  let worst = 0;
  for (let k = 0; k < c.adapt.energies.length; k++) {
    const st = E.prepare(c.taper.n, c.taper.ref_bits, E.adaptRotations(c.adapt, k));
    worst = Math.max(worst, Math.abs(E.energySV(st, c.H) - c.adapt.energies[k]));
  }
  check(`ADAPT energies, CAS R=${c.r} (${c.adapt.energies.length} iterations)`, worst < 1e-8, `max |Δ| = ${worst.toExponential(2)} Ha`);
  if (c.k_chem_acc) {
    const g = E.countGates(E.rotationGates(E.adaptRotations(c.adapt, c.k_chem_acc), c.taper.n));
    check(`gate count at k=${c.k_chem_acc}, R=${c.r}`, g.cx === c.gates_at_k.cx && g["1q"] === c.gates_at_k["1q"], `${g.cx} CX`);
  }
}
for (const f of D.fv) {
  let worst = 0;
  for (let k = 0; k < f.adapt.energies.length; k++) worst = Math.max(worst, Math.abs(E.energySV(E.prepare(f.taper.n, f.taper.ref_bits, E.adaptRotations(f.adapt, k)), f.H) - f.adapt.energies[k]));
  check(`ADAPT energies, FV R=${f.r}`, worst < 1e-8, `max |Δ| = ${worst.toExponential(2)} Ha`);
}

// 2. Slater-Condon Hamiltonian: full space reproduces exact diagonalisation
for (const c of [D.cas[2], D.cas[8], D.cas[11]]) {
  const r = E.subspaceEnergy(c, strings(c.n_orb, c.n_alpha));
  const ref = c.exact_lowest_any_spin ?? c.exact;  // full Sz=0 space: lowest state of any spin (septet at 3.0 A)
  check(`full-space Slater-Condon = exact (any spin), CAS R=${c.r}`, Math.abs(r.energy - ref) < 1e-8 && r.dim === binom(c.n_orb, c.n_alpha) ** 2, `dim ${r.dim}, Δ ${(r.energy - ref).toExponential(2)}`);
}
{ const t = Date.now(); const f = D.fv[2]; const r = E.subspaceEnergy(f, strings(f.n_orb, f.n_alpha));
  check(`full-space Slater-Condon = exact, FV R=${f.r}`, Math.abs(r.energy - f.exact) < 1e-8, `dim ${r.dim}, Δ ${(r.energy - f.exact).toExponential(2)}, ${Date.now() - t} ms`); }
for (const s of D.checks.subspace) {
  const p = (s.kind === "cas" ? D.cas : D.fv)[s.index], r = E.subspaceEnergy(p, s.strings);
  check(`subspace energy vs Python (${s.kind} R=${p.r}, ${s.dim} dets)`, Math.abs(r.energy - s.energy) < 1e-8 && r.dim === s.dim, `Δ ${(r.energy - s.energy).toExponential(2)}`);
}

// 3. Density-matrix noise experiments vs Python
for (const s of D.checks.noise) {
  const c = D.cas[s.cas_index], gates = E.rotationGates(E.adaptRotations(c.adapt, s.k), c.taper.n);
  const t = Date.now(), r = E.noiseExperiment(gates, c.taper.n, c.taper.ref_bits, c.H, { r2: s.r2, r1: s.r1, p_ro: s.p_ro });
  const worst = Math.max(...["raw", "rem", "zne_lin", "zne_rich"].map(k => Math.abs(r[k] - s[k])));
  check(`noisy energies vs Python, R=${c.r}, r2=${s.r2}`, worst < 1e-8, `max |Δ| = ${worst.toExponential(2)}, raw err ${(1e3 * (r.raw - c.exact)).toFixed(2)} mHa, ${Date.now() - t} ms`);
}

// 4. Trajectory sampler matches the density-matrix distribution
{ const c = D.cas[2], gates = E.rotationGates(E.adaptRotations(c.adapt, c.k_chem_acc), c.taper.n), noise = { r2: 3e-3, r1: 2.5e-4, p_ro: 0 };
  const rho = E.dmRun(gates, c.taper.n, c.taper.ref_bits, noise, 1), d = 1 << c.taper.n;
  const shots = 20000, t = Date.now(), s = E.sampleNoisy(gates, c.taper.n, c.taper.ref_bits, noise, shots, { seed: 5 });
  const hist = new Float64Array(d); for (const b of s) hist[b] += 1 / shots;
  let tvd = 0; for (let b = 0; b < d; b++) tvd += Math.abs(hist[b] - rho.re[b * d + b]) / 2;
  check("trajectory sampling = density-matrix distribution (shot-noise TVD ≈ 0.03)", tvd < 0.04, `TVD ${tvd.toFixed(4)} over ${shots} shots, ${Date.now() - t} ms`); }

// 5. Untapering: noiseless samples land in the right particle sector
{ const c = D.cas[8], gates = E.rotationGates(E.adaptRotations(c.adapt, c.k_chem_acc), c.taper.n);
  const s = E.sampleNoisy(gates, c.taper.n, c.taper.ref_bits, { r2: 0, r1: 0, p_ro: 0 }, 2000, { seed: 2 });
  const full = Array.from(s, b => E.untaper(c.taper, b)), sub = E.sqdSubspace(full, c.n_orb, c.n_alpha, c.n_beta, false);
  check("untapered noiseless samples are all valid determinants", sub.fracGood === 1, `good fraction ${sub.fracGood}`); }

// 5b. Trajectory energy converges to the density-matrix energy (REM applied in both)
{ const c = D.cas[2], gates = E.rotationGates(E.adaptRotations(c.adapt, 6), c.taper.n), noise = { r2: 3e-3, r1: 2.5e-4, p_ro: 0 };
  const exact = E.dmEnergy(E.dmRun(gates, c.taper.n, c.taper.ref_bits, noise, 1), c.H, 0, true), r = E.trajectoryEnergy(gates, c.taper.n, c.taper.ref_bits, c.H, noise, 4000);
  check("trajectory energy = density-matrix energy within 4 stderr", Math.abs(r.mean - exact) < 4 * r.stderr + 1e-9, `Δ ${(1e3 * (r.mean - exact)).toFixed(3)} mHa, stderr ${(1e3 * r.stderr).toFixed(3)} mHa`); }

// 6. End-to-end SQD on the noisy shallow FV circuit (Python reference: 0.49 mHa at 1000 shots, stochastic)
{ const f = D.fv[2], gates = E.rotationGates(E.adaptRotations(f.adapt, 4), f.taper.n), noise = { r2: 3e-3, r1: 2.5e-4, p_ro: 1.5e-2 };
  for (const shots of [300, 1000, 3000]) {
    const t = Date.now(), s = E.sampleNoisy(gates, f.taper.n, f.taper.ref_bits, noise, shots, { seed: 11, perTraj: 10 });
    const r = E.sqdEnergy(f, Array.from(s, b => E.untaper(f.taper, b)));
    const err = 1e3 * (r.energy - f.exact);
    check(`SQD FV R=2.1, 4 ops, ${shots} shots`, err > -1e-6 && err < 20, `err ${err.toFixed(3)} mHa, ${r.dim} dets, good ${r.fracGood.toFixed(2)}, ${Date.now() - t} ms`);
  } }

console.log(`\n${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
