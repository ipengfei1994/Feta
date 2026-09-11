/* ============================================================
   charts.js — 零依赖 SVG 图表（离线可用，无 CDN）
   renderDonut(el, [{label, value, color}])
   renderLine(el, [{label, total, high}], opts)
   renderBars(el, [{name, count}], colorMap?)
   ============================================================ */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function svgRoot(w, h) {
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet" });
    s.style.width = "100%"; s.style.height = "100%";
    return s;
  }

  /* ---------- 环形图 ---------- */
  function renderDonut(container, data) {
    container.innerHTML = "";
    const total = data.reduce((s, d) => s + d.value, 0);
    const W = 220, H = 220, cx = 110, cy = 108, r = 74, sw = 26;
    const svg = svgRoot(W, H);
    if (!total) {
      svg.appendChild(el("circle", { cx, cy, r, fill: "none", stroke: "#eef0f4", "stroke-width": sw }));
      svg.appendChild(text(cx, cy + 5, "暂无数据", "#9298a8", 13, "middle"));
    } else {
      let angle = -90;
      for (const d of data) {
        if (!d.value) continue;
        const frac = d.value / total;
        const a2 = angle + frac * 360;
        const large = frac > 0.5 ? 1 : 0;
        const p1 = polar(cx, cy, r, angle), p2 = polar(cx, cy, r, a2);
        svg.appendChild(el("path", {
          d: `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${large} 1 ${p2.x} ${p2.y}`,
          fill: "none", stroke: d.color, "stroke-width": sw, "stroke-linecap": "butt",
        }));
        angle = a2;
      }
      svg.appendChild(text(cx, cy - 2, String(total), "#1a1d27", 26, "middle", 800));
      svg.appendChild(text(cx, cy + 20, "动态总数", "#9298a8", 11, "middle"));
    }
    container.appendChild(svg);
  }
  function polar(cx, cy, r, deg) {
    const rad = (deg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }
  function text(x, y, str, fill, size, anchor, weight) {
    const t = el("text", { x, y, fill, "font-size": size, "text-anchor": anchor || "start",
      "font-weight": weight || 400, "font-family": "inherit" });
    t.textContent = str;
    return t;
  }

  /* ---------- 折线/面积图（双序列：总量 + 高危） ---------- */
  function renderLine(container, series) {
    container.innerHTML = "";
    const W = 420, H = 200, padL = 30, padR = 10, padT = 12, padB = 24;
    const svg = svgRoot(W, H);
    const n = series.length;
    const maxV = Math.max(3, ...series.map(s => s.total));
    const X = i => padL + (n <= 1 ? 0 : (i * (W - padL - padR)) / (n - 1));
    const Y = v => padT + (1 - v / maxV) * (H - padT - padB);

    // 网格 + y 轴刻度
    for (let g = 0; g <= 3; g++) {
      const v = (maxV / 3) * g, y = Y(v);
      svg.appendChild(el("line", { x1: padL, y1: y, x2: W - padR, y2: y,
        stroke: "#eef0f4", "stroke-width": 1 }));
      svg.appendChild(text(padL - 6, y + 3.5, Math.round(v), "#9298a8", 9.5, "end"));
    }
    // x 轴标签（最多 8 个）
    const step = Math.max(1, Math.ceil(n / 8));
    series.forEach((s, i) => {
      if (i % step === 0 || i === n - 1)
        svg.appendChild(text(X(i), H - 7, s.label, "#9298a8", 9, "middle"));
    });

    const path = key => series.map((s, i) =>
      `${i ? "L" : "M"} ${X(i).toFixed(1)} ${Y(s[key]).toFixed(1)}`).join(" ");
    // 高危面积
    svg.appendChild(el("path", {
      d: path("high") + ` L ${X(n - 1)} ${Y(0)} L ${X(0)} ${Y(0)} Z`,
      fill: "rgba(220,38,38,.10)", stroke: "none",
    }));
    svg.appendChild(el("path", { d: path("total"), fill: "none",
      stroke: "#4f46e5", "stroke-width": 2, "stroke-linejoin": "round" }));
    svg.appendChild(el("path", { d: path("high"), fill: "none",
      stroke: "#dc2626", "stroke-width": 2, "stroke-linejoin": "round",
      "stroke-dasharray": "1 0" }));
    series.forEach((s, i) => {
      svg.appendChild(el("circle", { cx: X(i), cy: Y(s.total), r: 2.4, fill: "#4f46e5" }));
      if (s.high) svg.appendChild(el("circle", { cx: X(i), cy: Y(s.high), r: 2.8, fill: "#dc2626" }));
    });
    container.appendChild(svg);
  }

  /* ---------- 横向条形图 ---------- */
  function renderBars(container, data) {
    container.innerHTML = "";
    const W = 300, rowH = 30, padT = 6;
    const H = padT + data.length * rowH + 6;
    const svg = svgRoot(W, H);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const maxV = Math.max(1, ...data.map(d => d.count));
    const labelW = 108, barMax = W - labelW - 40;
    data.forEach((d, i) => {
      const y = padT + i * rowH;
      svg.appendChild(text(labelW - 8, y + 15, d.name, "#5b6172", 10.5, "end"));
      const bw = Math.max(3, (d.count / maxV) * barMax);
      svg.appendChild(el("rect", { x: labelW, y: y + 4, width: bw, height: 14,
        rx: 4, fill: d.color || "#4f46e5" }));
      svg.appendChild(text(labelW + bw + 6, y + 15.5, String(d.count), "#1a1d27", 11, "start", 700));
    });
    container.appendChild(svg);
  }

  window.FetaCharts = { renderDonut, renderLine, renderBars };
})();
