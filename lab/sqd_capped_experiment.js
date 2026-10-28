// Honest SQD test (expert-panel request): error vs subspace size at a FIXED budget of K strings
// (subspace = K^2 determinants out of 56^2 = 3,136), for four ways of choosing the strings:
//   noisy circuit samples · noiseless circuit samples · uniform random bitstrings · classical HF/CISD-like ranking
// FV(10e,8o), R = 2.1 A, 4-brick ADAPT circuit (192 CX), r2 = 3e-3, readout 1.5%, 3,000 shots, 10 seeds, medians.
const fs = require("fs"), E = require("./engine.js");
const D = JSON.parse(fs.readFileSync(process.argv[2] || "results/lab_data.json", "utf8"));
const out = { setup: {}, rows: [] };
const median = a => { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
for (const fi of [2, 0]) {
  const f = D.fv[fi], nq = f.taper.n, SHOTS = 3000, SEEDS = 10, K_LIST = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56];
  const gates = E.rotationGates(E.adaptRotations(f.adapt, 4), nq), noisy = { r2: 3e-3, r1: 2.5e-4, p_ro: .015 }, clean = { r2: 0, r1: 0, p_ro: 0 };
  const res = { noisy: {}, clean: {}, random: {} }; K_LIST.forEach(K => { res.noisy[K] = []; res.clean[K] = []; res.random[K] = []; });
  const t0 = Date.now();
  for (let s = 1; s <= SEEDS; s++) {
    const runs = { noisy: E.sampleNoisy(gates, nq, f.taper.ref_bits, noisy, SHOTS, { seed: 100 + s }), clean: E.sampleNoisy(gates, nq, f.taper.ref_bits, clean, SHOTS, { seed: 200 + s }), random: E.randomTapered(nq, SHOTS, 300 + s) };
    for (const kind of Object.keys(runs)) {
      const ranked = E.rankStrings(f, Array.from(runs[kind], b => E.untaper(f.taper, b)));
      for (const K of K_LIST) res[kind][K].push(K <= ranked.length ? 1e3 * (E.cappedEnergy(f, ranked, K).energy - f.exact) : NaN);
    }
    process.stderr.write(`R=${f.r} seed ${s} (${((Date.now() - t0) / 1000).toFixed(0)} s)\n`);
  }
  const hf = E.hfRanked(f);
  for (const K of K_LIST) {
    const row = { r: f.r, K, dets: K * K, frac: K * K / 3136, hf_ranked: 1e3 * (E.cappedEnergy(f, hf, K).energy - f.exact) };
    for (const kind of ["noisy", "clean", "random"]) { const v = res[kind][K].filter(Number.isFinite); row[kind] = { median: v.length ? median(v) : null, min: Math.min(...v), max: Math.max(...v), n: v.length }; }
    out.rows.push(row);
    console.log(`R=${f.r} K=${String(K).padStart(2)} (${(100 * row.frac).toFixed(0).padStart(3)}% of space): noisy ${row.noisy.median?.toFixed(2)}  clean ${row.clean.median?.toFixed(2)}  random ${row.random.median?.toFixed(2)}  HF-ranked ${row.hf_ranked.toFixed(2)} mHa`);
  }
}
out.setup = { system: "FV(10e,8o) N2/STO-3G", circuit: "4 ADAPT bricks, 192 CX, 11 qubits", noise: "r2=3e-3, r1=2.5e-4, readout 1.5%", shots: 3000, seeds: 10, space: 3136 };
fs.writeFileSync("results/sqd_capped.json", JSON.stringify(out, null, 1));
console.log("saved results/sqd_capped.json");
