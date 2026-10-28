/* Arjuna Lab quantum engine: a faithful JavaScript port of the verified NumPy engine.
 *
 * Conventions match src/pauli.py: a Pauli term is (x, z) bitmasks, P(x,z) = i^{popcount(x&z)} X^x Z^z,
 * qubit q <-> bit q. Spin orbitals: alpha 0..n-1, beta n..2n-1 (Jordan-Wigner order).
 *
 * Contents
 *   statevector   : Pauli rotations, ADAPT state preparation, <H>
 *   gates         : same decomposition as src/noise.py rotation_gates (H / Rx(pi/2) basis change, CX ladder, Rz)
 *   density matrix: gate-level simulation with depolarising noise (same model as src/noise.py), readout damping,
 *                   readout-error mitigation, zero-noise extrapolation by folding every gate
 *   trajectories  : stochastic Pauli-noise statevector runs for sampling bitstrings (SQD)
 *   sqd           : untapering, configuration recovery, Slater-Condon subspace Hamiltonian, eigen-solvers
 * Works in browsers, Web Workers and Node (module.exports).
 */
(function (root) {
  "use strict";
  const popcount = v => { v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24; };
  const IPOW_RE = [1, 0, -1, 0], IPOW_IM = [0, 1, 0, -1];

  function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ---------------- statevector ---------------- */
  function basisState(n, bits) { const d = 1 << n, re = new Float64Array(d), im = new Float64Array(d); re[bits] = 1; return { n, re, im }; }
  // psi <- exp(-i angle/2 P) psi = cos(a/2) psi - i sin(a/2) P psi
  function applyPauliRot(st, x, z, angle) {
    const { re, im } = st, d = re.length, c = Math.cos(angle / 2), s = Math.sin(angle / 2);
    const k = popcount(x & z) & 3, pr = IPOW_RE[k], pi = IPOW_IM[k];
    const nre = new Float64Array(d), nim = new Float64Array(d);
    for (let b = 0; b < d; b++) {
      const sg = (popcount(z & b) & 1) ? -1 : 1;
      // (P psi)[b^x] = i^k * sg * psi[b]
      const a = re[b] * sg, bb = im[b] * sg;
      const qr = pr * a - pi * bb, qi = pr * bb + pi * a; // value of (P psi) at b^x
      const t = b ^ x;
      // -i s * (P psi)[t]
      nre[t] += s * qi; nim[t] += -s * qr;
    }
    for (let b = 0; b < d; b++) { nre[b] += c * re[b]; nim[b] += c * im[b]; }
    st.re = nre; st.im = nim; return st;
  }
  // exp(theta A), A = sum_j (i a_j) P_j with commuting P_j -> product of rotations with angle -2 a_j theta
  function opRotations(op, theta) { const out = []; for (const [x, z, , ci] of op) { if (Math.abs(ci) < 1e-14) continue; out.push([x, z, -2 * ci * theta]); } return out; }
  function adaptRotations(adapt, k) { if (k <= 0) return []; const th = adapt.thetas[k - 1]; let rots = []; for (let i = 0; i < k; i++) rots = rots.concat(opRotations(adapt.ops[i], th[i])); return rots; }
  function prepare(n, refBits, rots) { const st = basisState(n, refBits); for (const [x, z, a] of rots) applyPauliRot(st, x, z, a); return st; }
  function pauliExpectSV(st, x, z) {
    const { re, im } = st, d = re.length, k = popcount(x & z) & 3, pr = IPOW_RE[k], pi = IPOW_IM[k];
    let sr = 0, si = 0;
    for (let b = 0; b < d; b++) {
      const sg = (popcount(z & b) & 1) ? -1 : 1, t = b ^ x;
      const vr = sg * (pr * re[b] - pi * im[b]), vi = sg * (pr * im[b] + pi * re[b]); // (P psi)[t]
      sr += re[t] * vr + im[t] * vi; si += re[t] * vi - im[t] * vr; // conj(psi[t]) * (P psi)[t]
    }
    return sr;
  }
  function energySV(st, H) { let e = 0; for (const [x, z, cr] of H) e += (x === 0 && z === 0) ? cr : cr * pauliExpectSV(st, x, z); return e; }

  /* ---------------- gate decomposition (identical to src/noise.py) ---------------- */
  const S2 = Math.SQRT1_2;
  const H_G = { re: [S2, S2, S2, -S2], im: [0, 0, 0, 0] };
  const RX90 = { re: [S2, 0, 0, S2], im: [0, -S2, -S2, 0] };
  const RX90D = { re: [S2, 0, 0, S2], im: [0, S2, S2, 0] };
  function rz(a) { return { re: [Math.cos(a / 2), 0, 0, Math.cos(a / 2)], im: [-Math.sin(a / 2), 0, 0, Math.sin(a / 2)] }; }
  function rotationGates(rots, n) {
    const gates = [];
    for (const [x, z, ang] of rots) {
      const qs = []; for (let q = 0; q < n; q++) if (((x | z) >> q) & 1) qs.push(q);
      if (!qs.length) continue;
      const pre = [], post = [];
      for (const q of qs) { const xb = (x >> q) & 1, zb = (z >> q) & 1; if (xb && !zb) { pre.push(["1q", q, H_G]); post.push(["1q", q, H_G]); } else if (xb && zb) { pre.push(["1q", q, RX90]); post.push(["1q", q, RX90D]); } }
      const ladder = []; for (let i = 0; i < qs.length - 1; i++) ladder.push(["cx", qs[i], qs[i + 1]]);
      gates.push(...pre, ...ladder, ["1q", qs[qs.length - 1], rz(ang)], ...ladder.slice().reverse(), ...post);
    }
    return gates;
  }
  function countGates(gates) { let cx = 0; for (const g of gates) if (g[0] === "cx") cx++; return { cx, "1q": gates.length - cx }; }

  /* ---------------- density matrix ---------------- */
  function dmBasis(n, bits) { const d = 1 << n, re = new Float64Array(d * d), im = new Float64Array(d * d); re[bits * d + bits] = 1; return { n, d, re, im }; }
  function dmApply1q(rho, q, U) {
    const { d, re, im } = rho, m = 1 << q, [u0r, u1r, u2r, u3r] = U.re, [u0i, u1i, u2i, u3i] = U.im;
    // rows: rho <- U rho
    for (let i0 = 0; i0 < d; i0++) { if (i0 & m) continue; const i1 = i0 | m, r0 = i0 * d, r1 = i1 * d;
      for (let c = 0; c < d; c++) { const ar = re[r0 + c], ai = im[r0 + c], br = re[r1 + c], bi = im[r1 + c];
        re[r0 + c] = u0r * ar - u0i * ai + u1r * br - u1i * bi; im[r0 + c] = u0r * ai + u0i * ar + u1r * bi + u1i * br;
        re[r1 + c] = u2r * ar - u2i * ai + u3r * br - u3i * bi; im[r1 + c] = u2r * ai + u2i * ar + u3r * bi + u3i * br; } }
    // cols: rho <- rho U^dagger ; (rho U^†)[r][c] = sum_k rho[r][k] conj(U[c][k])
    for (let r = 0; r < d; r++) { const base = r * d;
      for (let c0 = 0; c0 < d; c0++) { if (c0 & m) continue; const c1 = c0 | m, ar = re[base + c0], ai = im[base + c0], br = re[base + c1], bi = im[base + c1];
        re[base + c0] = ar * u0r + ai * u0i + br * u1r + bi * u1i; im[base + c0] = ai * u0r - ar * u0i + bi * u1r - br * u1i;
        re[base + c1] = ar * u2r + ai * u2i + br * u3r + bi * u3i; im[base + c1] = ai * u2r - ar * u2i + bi * u3r - br * u3i; } }
  }
  function dmApplyCX(rho, c, t) {
    const { d, re, im } = rho, cm = 1 << c, tm = 1 << t, nre = new Float64Array(d * d), nim = new Float64Array(d * d);
    for (let i = 0; i < d; i++) { const pi = (i & cm) ? (i ^ tm) : i, ri = i * d, rp = pi * d;
      for (let j = 0; j < d; j++) { const pj = (j & cm) ? (j ^ tm) : j; nre[ri + j] = re[rp + pj]; nim[ri + j] = im[rp + pj]; } }
    rho.re = nre; rho.im = nim;
  }
  function dmDepolarize(rho, qubits, lam) {
    if (lam <= 0) return;
    const { d, re, im } = rho; let M = 0; for (const q of qubits) M |= 1 << q;
    const subs = []; for (let s = 0; s < d; s++) if ((s & ~M) === 0) subs.push(s); // all bit patterns on M
    const K = subs.length, inv = 1 / K, nM = ~M;
    const nre = new Float64Array(d * d), nim = new Float64Array(d * d);
    for (let i = 0; i < d; i++) { if (i & M) continue;
      for (let j = 0; j < d; j++) { if (j & M) continue;
        let sr = 0, si = 0; for (const s of subs) { sr += re[(i | s) * d + (j | s)]; si += im[(i | s) * d + (j | s)]; }
        sr *= inv; si *= inv;
        for (const s of subs) { nre[(i | s) * d + (j | s)] = sr; nim[(i | s) * d + (j | s)] = si; } } }
    for (let k = 0; k < d * d; k++) { re[k] = (1 - lam) * re[k] + lam * nre[k]; im[k] = (1 - lam) * im[k] + lam * nim[k]; }
    void nM;
  }
  function dmRun(gates, n, refBits, noise, fold) {
    fold = fold || 1; const rho = dmBasis(n, refBits);
    const l2 = noise ? 4 / 3 * noise.r2 : 0, l1 = noise ? 2 * noise.r1 : 0;
    for (const g of gates) {
      if (g[0] === "1q") { dmApply1q(rho, g[1], g[2]); if (l1) for (let f = 0; f < fold; f++) dmDepolarize(rho, [g[1]], l1); }
      else for (let f = 0; f < fold; f++) { dmApplyCX(rho, g[1], g[2]); if (l2) dmDepolarize(rho, [g[1], g[2]], l2); }
    }
    return rho;
  }
  function dmPauliExpect(rho, x, z) {
    const { d, re, im } = rho, k = popcount(x & z) & 3, pr = IPOW_RE[k], pi = IPOW_IM[k];
    let s = 0; // Tr(P rho) = sum_c i^k (-1)^{pc(z&c)} rho[c][c^x]
    for (let c = 0; c < d; c++) { const sg = (popcount(z & c) & 1) ? -1 : 1, idx = c * d + (c ^ x); s += sg * (pr * re[idx] - pi * im[idx]); }
    return s;
  }
  function dmEnergy(rho, H, pRO, rem) {
    let e = 0;
    for (const [x, z, cr] of H) {
      if (x === 0 && z === 0) { e += cr; continue; }
      const damp = Math.pow(1 - 2 * (pRO || 0), popcount(x | z));
      let m = dmPauliExpect(rho, x, z) * damp; if (rem) m /= damp; e += cr * m;
    }
    return e;
  }
  function zne(scales, vals, kind, asym) {
    if (kind === "linear") { const [s0, s1] = scales, [v0, v1] = vals; return v0 - s0 * (v1 - v0) / (s1 - s0); }
    if (kind === "richardson") { let e = 0; for (let i = 0; i < scales.length; i++) { let L = 1; for (let j = 0; j < scales.length; j++) if (j !== i) L *= (0 - scales[j]) / (scales[i] - scales[j]); e += L * vals[i]; } return e; }
    if (kind === "exponential") { // exact a + b e^{c s} through three equally spaced points (free asymptote)
      const [v1, v3, v5] = vals, q = (v5 - v3) / (v3 - v1); if (!(q > 0) || Math.abs(q - 1) < 1e-12) return NaN;
      const be = (v3 - v1) / (q - 1); return v1 - be + be / Math.sqrt(q);
    }
    if (kind === "exponential_asym") { // v(s) = E_inf - B e^{-cs}, E_inf = fully-mixed energy; least squares on ln(E_inf - v)
      if (asym === undefined) throw new Error("exponential ZNE needs the asymptote");
      const ys = vals.map(v => asym - v); if (ys.some(y => !(y > 0))) return NaN;
      const L = ys.map(Math.log), n = scales.length, mx = scales.reduce((a, b) => a + b) / n, my = L.reduce((a, b) => a + b) / n;
      let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (scales[i] - mx) * (L[i] - my); sxx += (scales[i] - mx) ** 2; }
      return asym - Math.exp(my - (sxy / sxx) * mx);
    }
    throw new Error(kind);
  }
  function pecOverhead(gates, noise) {
    const l2 = 4 / 3 * noise.r2, l1 = 2 * noise.r1, g2 = 1 + 15 * l2 / (8 * (1 - l2)), g1 = 1 + 3 * l1 / (2 * (1 - l1));
    const c = countGates(gates); return Math.pow(g2, c.cx) * Math.pow(g1, c["1q"]);
  }
  // full noise experiment on one circuit: raw, REM, ZNE variants
  function noiseExperiment(gates, n, refBits, H, noise) {
    const rhos = [1, 3, 5].map(s => dmRun(gates, n, refBits, noise, s));
    const raw = dmEnergy(rhos[0], H, noise.p_ro, false), rem = rhos.map(r => dmEnergy(r, H, noise.p_ro, true));
    // single-shot standard deviation of the energy estimator (one shot per Pauli term), as in src/noise.py
    let var1 = 0; for (const [x, z, cr] of H) { if (x === 0 && z === 0) continue; const ev = dmPauliExpect(rhos[0], x, z); var1 += cr * cr * Math.max(1 - ev * ev, 0); }
    let asym = 0; for (const [x, z, cr] of H) if (x === 0 && z === 0) asym += cr;
    return { raw, rem: rem[0], zne_lin: zne([1, 3], rem.slice(0, 2), "linear"), zne_rich: zne([1, 3, 5], rem, "richardson"), zne_exp: zne([1, 3, 5], rem, "exponential"), zne_exp_asym: zne([1, 3, 5], rem, "exponential_asym", asym), asym, pec_gamma: pecOverhead(gates, noise), sigma1: Math.sqrt(var1), remScaled: rem };
  }

  /* ---------------- stochastic trajectories (sampling) ---------------- */
  function svApply1q(st, q, U) { const { re, im } = st, d = re.length, m = 1 << q, [a, b, c, e] = U.re, [ai, bi, ci, ei] = U.im;
    for (let i0 = 0; i0 < d; i0++) { if (i0 & m) continue; const i1 = i0 | m, xr = re[i0], xi = im[i0], yr = re[i1], yi = im[i1];
      re[i0] = a * xr - ai * xi + b * yr - bi * yi; im[i0] = a * xi + ai * xr + b * yi + bi * yr;
      re[i1] = c * xr - ci * xi + e * yr - ei * yi; im[i1] = c * xi + ci * xr + e * yi + ei * yr; } }
  function svApplyCX(st, c, t) { const { re, im } = st, d = re.length, cm = 1 << c, tm = 1 << t;
    for (let i = 0; i < d; i++) if ((i & cm) && !(i & tm)) { const j = i | tm; let v = re[i]; re[i] = re[j]; re[j] = v; v = im[i]; im[i] = im[j]; im[j] = v; } }
  function svPauli(st, x, z) { // apply X^x Z^z (global phase irrelevant for sampling)
    const { re, im } = st, d = re.length, nre = new Float64Array(d), nim = new Float64Array(d);
    for (let b = 0; b < d; b++) { const sg = (popcount(z & b) & 1) ? -1 : 1; nre[b ^ x] = sg * re[b]; nim[b ^ x] = sg * im[b]; }
    st.re = nre; st.im = nim; }
  // one noisy trajectory: depolarising channel = random Pauli (uniform over all, identity included) with probability lambda
  function trajectory(gates, n, refBits, noise, R) {
    const l2 = 4 / 3 * noise.r2, l1 = 2 * noise.r1, st = basisState(n, refBits);
    for (const g of gates) {
      if (g[0] === "1q") { svApply1q(st, g[1], g[2]); if (l1 && R() < l1) { const p = Math.floor(R() * 4); svPauli(st, (p & 1) << g[1], (p >> 1) << g[1]); } }
      else { svApplyCX(st, g[1], g[2]); if (l2 && R() < l2) { const p = Math.floor(R() * 16); svPauli(st, ((p & 1) << g[1]) | (((p >> 1) & 1) << g[2]), (((p >> 2) & 1) << g[1]) | (((p >> 3) & 1) << g[2])); } }
    }
    return st;
  }
  // noisy energy estimate (readout-mitigated) averaged over trajectories: what plain VQE would report
  function trajectoryEnergy(gates, n, refBits, H, noise, T, opts) {
    opts = opts || {}; const R = rng(opts.seed || 3); let sum = 0, sum2 = 0;
    for (let t = 0; t < T; t++) { const e = energySV(trajectory(gates, n, refBits, noise, R), H); sum += e; sum2 += e * e; if (opts.onProgress && t % 10 === 0) opts.onProgress(t / T); }
    const mean = sum / T; return { mean, stderr: Math.sqrt(Math.max(sum2 / T - mean * mean, 0) / T) };
  }
  // shots: by default every shot is its own trajectory, exactly like hardware (perTraj > 1 trades variance for speed)
  function sampleNoisy(gates, n, refBits, noise, shots, opts) {
    opts = opts || {}; const R = rng(opts.seed || 1), perTraj = Math.max(1, opts.perTraj || 1), out = new Int32Array(shots);
    let k = 0;
    while (k < shots) {
      const st = trajectory(gates, n, refBits, noise, R);
      if (opts.onProgress && k % 50 === 0) opts.onProgress(k / shots);
      const d = st.re.length, cdf = new Float64Array(d); let acc = 0; for (let b = 0; b < d; b++) { acc += st.re[b] * st.re[b] + st.im[b] * st.im[b]; cdf[b] = acc; }
      for (let s = 0; s < perTraj && k < shots; s++, k++) {
        const u = R() * acc; let lo = 0, hi = d - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
        let bits = lo; if (noise.p_ro) for (let q = 0; q < n; q++) if (R() < noise.p_ro) bits ^= 1 << q;
        out[k] = bits;
      }
    }
    return out;
  }

  /* ---------------- SQD ---------------- */
  function untaper(tp, tbits) {
    let full = 0; tp.keep.forEach((q, i) => { if ((tbits >> i) & 1) full |= 1 << q; });
    tp.generators.forEach((s, i) => { const q = tp.pivots[i], want = tp.eigs[i] === 1 ? 0 : 1; const par = popcount(full & s) & 1; if (par ^ want) full |= 1 << q; });
    return full;
  }
  function recoverStrings(strings, nOrb, nEl, occ) {
    return strings.map(s0 => { const bits = []; for (let q = 0; q < nOrb; q++) bits.push((s0 >> q) & 1); let cnt = popcount(s0);
      while (cnt !== nEl) {
        if (cnt > nEl) { let best = -1; for (let q = 0; q < nOrb; q++) if (bits[q] && (best < 0 || occ[q] < occ[best])) best = q; bits[best] = 0; cnt--; }
        else { let best = -1; for (let q = 0; q < nOrb; q++) if (!bits[q] && (best < 0 || occ[q] > occ[best])) best = q; bits[best] = 1; cnt++; }
      }
      return bits.reduce((a, b, q) => a | (b << q), 0); });
  }
  function sqdSubspace(fullBits, nOrb, nA, nB, recover) {
    const amask = (1 << nOrb) - 1; let A = [], B = [], good = [];
    for (const v of fullBits) { const a = v & amask, b = (v >>> nOrb) & amask; A.push(a); B.push(b); good.push(popcount(a) === nA && popcount(b) === nB); }
    const ng = good.filter(Boolean).length, fracGood = fullBits.length ? ng / fullBits.length : 0;
    if (recover !== false && ng > 0) {
      const oa = new Array(nOrb).fill(0), ob = new Array(nOrb).fill(0);
      A.forEach((a, i) => { if (good[i]) for (let q = 0; q < nOrb; q++) { oa[q] += (a >> q) & 1; ob[q] += (B[i] >> q) & 1; } });
      for (let q = 0; q < nOrb; q++) { oa[q] /= ng; ob[q] /= ng; }
      A = recoverStrings(A, nOrb, nA, oa); B = recoverStrings(B, nOrb, nB, ob);
    } else { A = A.filter((_, i) => good[i]); B = B.filter((_, i) => good[i]); }
    const strings = Array.from(new Set(A.concat(B))).sort((x, y) => x - y);
    const dets = []; for (const a of strings) for (const b of strings) dets.push(a | (b << nOrb));
    dets.sort((x, y) => x - y);
    return { strings, dets, fracGood };
  }
  // spin-orbital integrals from spatial (chemist) integrals
  function makeIntegrals(p) {
    const n = p.n_orb, h = p.h, g = p.g, N = 2 * n;
    const so1 = (a, b) => ((a < n) === (b < n)) ? h[a % n][b % n] : 0;
    const G = (a, b, c, e) => g[((a * n + b) * n + c) * n + e];
    const phys = (pp, q, r, s) => (((pp < n) === (r < n)) && ((q < n) === (s < n))) ? G(pp % n, r % n, q % n, s % n) : 0; // <pq|rs> = (pr|qs)
    const anti = (pp, q, r, s) => phys(pp, q, r, s) - phys(pp, q, s, r);
    return { n, N, ecore: p.ecore, so1, anti };
  }
  const below = (D, p) => popcount(D & ((1 << p) - 1));
  function hElement(I, D1, D2) { // <D1|H|D2>
    const diff = D1 ^ D2, nd = popcount(diff);
    if (nd === 0) {
      const occ = []; for (let p = 0; p < I.N; p++) if ((D1 >> p) & 1) occ.push(p);
      let e = I.ecore; for (const i of occ) e += I.so1(i, i);
      for (let x = 0; x < occ.length; x++) for (let y = x + 1; y < occ.length; y++) e += I.anti(occ[x], occ[y], occ[x], occ[y]);
      return e;
    }
    if (nd === 2) {
      const i = 31 - Math.clz32(diff & D2), a = 31 - Math.clz32(diff & D1); // D1 = a†_a a_i D2
      let sign = (below(D2, i) & 1) ? -1 : 1; const Dm = D2 & ~(1 << i); if (below(Dm, a) & 1) sign = -sign;
      let v = I.so1(a, i); for (let j = 0; j < I.N; j++) if (((D2 >> j) & 1) && j !== i) v += I.anti(a, j, i, j);
      return sign * v;
    }
    if (nd === 4) {
      const ann = [], cre = []; for (let p = 0; p < I.N; p++) { if ((diff >> p) & 1) { if ((D2 >> p) & 1) ann.push(p); else cre.push(p); } }
      const [i, j] = ann, [a, b] = cre; // D1 = a†_a a†_b a_j a_i D2
      let D = D2, sign = 1;
      if (below(D, i) & 1) sign = -sign; D &= ~(1 << i);
      if (below(D, j) & 1) sign = -sign; D &= ~(1 << j);
      if (below(D, b) & 1) sign = -sign; D |= 1 << b;
      if (below(D, a) & 1) sign = -sign; D |= 1 << a;
      return sign * I.anti(a, b, i, j);
    }
    return 0;
  }
  function buildSparse(I, dets) {
    const m = dets.length, rows = [], cols = [], vals = [], diag = new Float64Array(m);
    for (let r = 0; r < m; r++) { diag[r] = hElement(I, dets[r], dets[r]);
      for (let c = r + 1; c < m; c++) { const nd = popcount(dets[r] ^ dets[c]); if (nd !== 2 && nd !== 4) continue; const v = hElement(I, dets[r], dets[c]); if (v !== 0) { rows.push(r); cols.push(c); vals.push(v); } } }
    return { m, diag, rows: Int32Array.from(rows), cols: Int32Array.from(cols), vals: Float64Array.from(vals) };
  }
  function matvec(S, x, y) { y.set(S.diag.map((d, i) => d * x[i])); const { rows, cols, vals } = S; for (let k = 0; k < vals.length; k++) { const r = rows[k], c = cols[k], v = vals[k]; y[r] += v * x[c]; y[c] += v * x[r]; } }
  // symmetric tridiagonal eigenvalues (QL with implicit shifts), returns sorted ascending
  function tridiagEigs(dIn, eIn) {
    const n = dIn.length, d = Float64Array.from(dIn), e = new Float64Array(n); for (let i = 0; i < n - 1; i++) e[i] = eIn[i];
    for (let l = 0; l < n; l++) { let iter = 0, m;
      do { for (m = l; m < n - 1; m++) { const dd = Math.abs(d[m]) + Math.abs(d[m + 1]); if (Math.abs(e[m]) <= 1e-15 * dd) break; }
        if (m !== l) { if (iter++ > 60) break;
          let g = (d[l + 1] - d[l]) / (2 * e[l]), r = Math.hypot(g, 1); g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
          let s = 1, c = 1, p = 0, i;
          for (i = m - 1; i >= l; i--) { let f = s * e[i]; const b = c * e[i]; r = Math.hypot(f, g); e[i + 1] = r;
            if (r === 0) { d[i + 1] -= p; e[m] = 0; break; }
            s = f / r; c = g / r; g = d[i + 1] - p; r = (d[i] - g) * s + 2 * c * b; p = s * r; d[i + 1] = g + p; g = c * r - b; }
          if (r === 0 && i >= l) continue; d[l] -= p; e[l] = g; e[m] = 0; } } while (m !== l); }
    return Array.from(d).sort((a, b) => a - b);
  }
  function lowestEigen(S, opts) {
    opts = opts || {}; const m = S.m; if (m === 1) return S.diag[0];
    const R = rng(opts.seed || 7), maxIt = Math.min(m, opts.maxIt || 220), V = [];
    let v = new Float64Array(m); for (let i = 0; i < m; i++) v[i] = R() - .5; let nrm = Math.hypot(...v.slice(0, 0)); nrm = Math.sqrt(v.reduce((s, a) => s + a * a, 0)); v = v.map(a => a / nrm);
    const alpha = [], beta = []; let w = new Float64Array(m), prev = 1e300;
    for (let it = 0; it < maxIt; it++) {
      V.push(v); matvec(S, v, w); const a = w.reduce((s, x, i) => s + x * v[i], 0); alpha.push(a);
      for (const u of V) { const c = w.reduce((s, x, i) => s + x * u[i], 0); for (let i = 0; i < m; i++) w[i] -= c * u[i]; } // full reorthogonalisation (twice is enough)
      for (const u of V) { const c = w.reduce((s, x, i) => s + x * u[i], 0); for (let i = 0; i < m; i++) w[i] -= c * u[i]; }
      const b = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
      if (it % 10 === 9 || b < 1e-10 || it === maxIt - 1) { const ev = tridiagEigs(alpha, beta)[0]; if (b < 1e-10 || Math.abs(ev - prev) < 1e-12) return ev; prev = ev; }
      beta.push(b); v = w.map(x => x / b); w = new Float64Array(m);
    }
    return tridiagEigs(alpha, beta.slice(0, alpha.length - 1))[0];
  }
  function sqdEnergy(problem, fullBits, recover) {
    const sub = sqdSubspace(fullBits, problem.n_orb, problem.n_alpha, problem.n_beta, recover);
    if (!sub.dets.length) return { energy: NaN, dim: 0, fracGood: sub.fracGood };
    const I = makeIntegrals(problem), S = buildSparse(I, sub.dets);
    return { energy: lowestEigen(S), dim: sub.dets.length, fracGood: sub.fracGood, strings: sub.strings.length };
  }
  // capped SQD: keep the K most frequent strings (alpha and beta pooled, spin-closed product space K^2)
  function rankStrings(problem, fullBits, recover) {
    const sub = sqdSubspaceCounts(fullBits, problem.n_orb, problem.n_alpha, problem.n_beta, recover);
    return Object.entries(sub.counts).sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(e => +e[0]);
  }
  function sqdSubspaceCounts(fullBits, nOrb, nA, nB, recover) {
    const amask = (1 << nOrb) - 1; let A = [], B = [], good = [];
    for (const v of fullBits) { const a = v & amask, b = (v >>> nOrb) & amask; A.push(a); B.push(b); good.push(popcount(a) === nA && popcount(b) === nB); }
    const ng = good.filter(Boolean).length;
    if (recover !== false && ng > 0) {
      const oa = new Array(nOrb).fill(0), ob = new Array(nOrb).fill(0);
      A.forEach((a, i) => { if (good[i]) for (let q = 0; q < nOrb; q++) { oa[q] += (a >> q) & 1; ob[q] += (B[i] >> q) & 1; } });
      for (let q = 0; q < nOrb; q++) { oa[q] /= ng; ob[q] /= ng; }
      A = recoverStrings(A, nOrb, nA, oa); B = recoverStrings(B, nOrb, nB, ob);
    } else { A = A.filter((_, i) => good[i]); B = B.filter((_, i) => good[i]); }
    const counts = {}; for (const s of A.concat(B)) counts[s] = (counts[s] || 0) + 1;
    return { counts, fracGood: fullBits.length ? ng / fullBits.length : 0 };
  }
  function cappedEnergy(problem, ranked, K) { const strings = ranked.slice(0, K); if (!strings.length) return { energy: NaN, dim: 0 }; return subspaceEnergy(problem, strings); }
  // classical baseline: strings ordered by excitation level from Hartree-Fock (CISD-like ranking)
  function hfRanked(problem) { const n = problem.n_orb, k = problem.n_alpha, hf = (1 << k) - 1, out = []; for (let s = 0; s < (1 << n); s++) if (popcount(s) === k) out.push(s); return out.sort((a, b) => popcount(a ^ hf) - popcount(b ^ hf) || a - b); }
  function randomTapered(n, shots, seed) { const R = rng(seed), out = new Int32Array(shots); for (let i = 0; i < shots; i++) out[i] = Math.floor(R() * (1 << n)); return out; }
  function allDetsFromStrings(problem, strings) { const dets = []; for (const a of strings) for (const b of strings) dets.push(a | (b << problem.n_orb)); dets.sort((x, y) => x - y); return dets; }
  function subspaceEnergy(problem, strings) { const I = makeIntegrals(problem), dets = allDetsFromStrings(problem, strings); return { energy: lowestEigen(buildSparse(I, dets)), dim: dets.length }; }

  const api = { popcount, rng, basisState, applyPauliRot, opRotations, adaptRotations, prepare, energySV, rotationGates, countGates,
    dmRun, dmEnergy, zne, pecOverhead, noiseExperiment, sampleNoisy, trajectoryEnergy, untaper, sqdSubspace, sqdEnergy, subspaceEnergy, rankStrings, cappedEnergy, hfRanked, randomTapered, makeIntegrals, hElement, buildSparse, lowestEigen, tridiagEigs };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.QEngine = api;
})(typeof self !== "undefined" ? self : this);
