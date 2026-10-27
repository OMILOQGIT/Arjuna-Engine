// Minimal SVG charting for the Arjuna web app: linear/log axes, lines, markers, bands, reference
// lines, legend and a hover read-out. No dependencies, works offline.
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const color = c => (c && c.startsWith("--") ? css(c) : c);
  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const pow10 = v => { const e = Math.round(Math.log10(v)); if (e === 0) return "1"; if (e === 1) return "10"; return "10" + (e < 0 ? "⁻" : "") + String(Math.abs(e)).split("").map(d => SUP[d]).join(""); };

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function scale(ax, r0, r1) {
    if (ax.log) { const a = Math.log10(ax.min), b = Math.log10(ax.max); return v => r0 + (Math.log10(Math.max(v, ax.min * 1e-3)) - a) / (b - a) * (r1 - r0); }
    return v => r0 + (v - ax.min) / (ax.max - ax.min) * (r1 - r0);
  }
  function autoTicks(ax) {
    if (ax.ticks) return ax.ticks;
    if (ax.log) { const out = []; for (let e = Math.ceil(Math.log10(ax.min)); e <= Math.floor(Math.log10(ax.max)); e++) out.push(10 ** e); return out; }
    const span = ax.max - ax.min, raw = span / 5, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= 6);
    const out = []; for (let v = Math.ceil(ax.min / step) * step; v <= ax.max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function marker(g, kind, x, y, s, fill, stroke) {
    if (kind === "tri") return el("path", { d: `M${x - s} ${y - s * .8}L${x + s} ${y - s * .8}L${x} ${y + s}Z`, fill, stroke, "stroke-width": 1.5 }, g);
    if (kind === "x") { const p = el("path", { d: `M${x - s} ${y - s}L${x + s} ${y + s}M${x + s} ${y - s}L${x - s} ${y + s}`, stroke: stroke || fill, "stroke-width": 2.4, fill: "none" }, g); return p; }
    if (kind === "square") return el("rect", { x: x - s, y: y - s, width: 2 * s, height: 2 * s, fill, stroke, "stroke-width": 1.8 }, g);
    if (kind === "ring") return el("circle", { cx: x, cy: y, r: s, fill: "none", stroke: stroke || fill, "stroke-width": 2.2 }, g);
    return el("circle", { cx: x, cy: y, r: s, fill, stroke, "stroke-width": 1.5 }, g);
  }

  function chart(container, cfg) {
    container.innerHTML = "";
    const W = cfg.width || 760, H = cfg.height || 340;
    const m = Object.assign({ l: 70, r: 18, t: 18, b: 50 }, cfg.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "plot", role: "img", "aria-label": cfg.title || "chart" }, container);
    const X = scale(cfg.x, m.l, W - m.r), Y = scale(cfg.y, H - m.b, m.t);
    const ink = color("--chalk"), faint = color("--faint"), grid = color("--line");
    const inside = (x, y) => x >= m.l - 1 && x <= W - m.r + 1 && y >= m.t - 1 && y <= H - m.b + 1;
    // bands
    for (const b of cfg.bands || []) {
      const c = color(b.color || "--green");
      if (b.axis === "x") { const x0 = X(b.from), x1 = X(b.to); el("rect", { x: Math.min(x0, x1), y: m.t, width: Math.abs(x1 - x0), height: H - m.b - m.t, fill: c, opacity: b.opacity || .12 }, svg); if (b.label) el("text", { x: (x0 + x1) / 2, y: m.t + 14, "text-anchor": "middle", class: "plot-band", fill: c }, svg, b.label); }
      else { const y0 = Y(b.from), y1 = Y(b.to); el("rect", { x: m.l, y: Math.min(y0, y1), width: W - m.r - m.l, height: Math.abs(y1 - y0), fill: c, opacity: b.opacity || .12 }, svg); }
    }
    // grid + axes
    const g = el("g", { class: "plot-axes" }, svg);
    for (const v of autoTicks(cfg.y)) { const y = Y(v); if (y < m.t - 1 || y > H - m.b + 1) continue; el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: grid, "stroke-width": 1 }, g); el("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", fill: faint }, g, (cfg.y.fmt || (cfg.y.log ? pow10 : String))(v)); }
    for (const v of autoTicks(cfg.x)) { const x = X(v); if (x < m.l - 1 || x > W - m.r + 1) continue; el("line", { x1: x, x2: x, y1: H - m.b, y2: H - m.b + 5, stroke: faint }, g); el("text", { x, y: H - m.b + 20, "text-anchor": "middle", fill: faint }, g, (cfg.x.fmt || (cfg.x.log ? pow10 : String))(v)); }
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: faint }, g);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, stroke: faint }, g);
    if (cfg.x.label) el("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", fill: faint, class: "plot-label" }, g, cfg.x.label);
    if (cfg.y.label) el("text", { x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", fill: faint, class: "plot-label" }, g, cfg.y.label);
    // reference lines
    for (const h of cfg.hlines || []) { const y = Y(h.y), c = color(h.color || "--green"); el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: c, "stroke-width": 1.6, "stroke-dasharray": h.dash || "6 5" }, svg); if (h.label) el("text", { x: h.left ? m.l + 6 : W - m.r - 4, y: y + (h.below ? 15 : -6), "text-anchor": h.left ? "start" : "end", fill: c, class: "plot-note" }, svg, h.label); }
    for (const v of cfg.vlines || []) { const x = X(v.x), c = color(v.color || "--yellow"); el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: c, "stroke-width": 1.4, "stroke-dasharray": v.dash || "3 5" }, svg); if (v.label) el("text", { x: x + 5, y: m.t + 12, fill: c, class: "plot-note" }, svg, v.label); }
    // series
    const pts = [];
    for (const s of cfg.series || []) {
      const c = color(s.color || "--cyan");
      const P = (s.points || []).filter(p => p && Number.isFinite(p[0]) && Number.isFinite(p[1]) && (!cfg.y.log || p[1] > 0) && (!cfg.x.log || p[0] > 0));
      if (s.line !== false && P.length > 1) {
        let d = "", pen = false;
        for (const p of s.points) {  // a missing or invalid value breaks the line
          if (!(p && P.includes(p)) || (p[2] && p[2].noLine)) { pen = false; continue; }
          const x = X(p[0]), y = Y(p[1]); d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true;
        }
        el("path", { d, fill: "none", stroke: c, "stroke-width": s.width || 2.6, "stroke-dasharray": s.dash, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity }, svg);
      }
      if (s.marker !== false) for (const p of P) {
        const x = X(p[0]), y = Y(p[1]); if (!inside(x, y)) continue;
        const kind = (p[2] && p[2].marker) || s.marker || "circle";
        const mk = marker(svg, kind, x, y, s.size || 4.2, kind === "ring" || kind === "x" ? "none" : c, kind === "ring" || kind === "x" ? c : color("--board"));
        if (p[2] && p[2].label) el("title", {}, mk, p[2].label);
        pts.push({ x, y, s, p });
      }
    }
    for (const a of cfg.annotations || []) { const x = X(a.x), y = Y(a.y); el("text", { x: x + (a.dx || 0), y: y + (a.dy || 0), "text-anchor": a.anchor || "start", fill: color(a.color || "--chalk"), class: a.hand ? "plot-hand" : "plot-note" }, svg, a.text); }
    // legend
    if (cfg.legend !== false) {
      const leg = document.createElement("div"); leg.className = "plot-legend";
      for (const s of cfg.series || []) if (s.name && !s.hideLegend) { const i = document.createElement("span"); i.innerHTML = `<i style="background:${color(s.color || "--cyan")}${s.dash ? ";opacity:.7" : ""}"></i>`; i.append(s.name); leg.appendChild(i); }
      for (const l of cfg.extraLegend || []) { const i = document.createElement("span"); i.style.color = color(l.color); i.textContent = l.text; leg.appendChild(i); }
      if (leg.children.length) container.insertBefore(leg, svg);
    }
    // hover read-out
    const tip = document.createElement("div"); tip.className = "plot-tip"; tip.hidden = true; container.appendChild(tip);
    const fx = cfg.tipX || (v => (+v).toPrecision(4)), fy = cfg.tipY || (v => (+v).toPrecision(4));
    svg.addEventListener("pointermove", ev => {
      if (!pts.length) return;
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) / r.width * W, sy = (ev.clientY - r.top) / r.height * H;
      let best = null, bd = 900;
      for (const q of pts) { const d = (q.x - sx) ** 2 + (q.y - sy) ** 2; if (d < bd) { bd = d; best = q; } }
      if (!best) { tip.hidden = true; return; }
      tip.hidden = false;
      tip.textContent = `${best.s.name ? best.s.name + " · " : ""}${fx(best.p[0])} → ${(best.p[2] && best.p[2].tip) || fy(best.p[1])}`;
      tip.style.left = Math.min(r.width - 10, best.x / W * r.width) + "px"; tip.style.top = (best.y / H * r.height) + "px";
    });
    svg.addEventListener("pointerleave", () => { tip.hidden = true; });
    return { X, Y, svg };
  }
  root.Plot = { chart, pow10 };
})(window);
