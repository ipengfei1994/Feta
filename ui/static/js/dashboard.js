/* ============================================================
   dashboard.js — 监测后台逻辑
   - 时间范围切换（24h / 7d / 30d）→ /api/metrics?range=
   - 图表：风险分布环图 / 动态趋势双线 / 压力源 Top 条形图
   - 告警列表：状态 × 级别 双维筛选，标记已处理（POST resolve）
   - 单条文本实时分析（支持强制风险等级演示危机路径）
   - 指标 20s 自动轮询（近实时）
   ============================================================ */
(function () {
  "use strict";

  const $ = id => document.getElementById(id);
  const RANGE_HOURS = { "24h": 24, "7d": 168, "30d": 720 };
  const RISK_META = {
    Normal:     { color: "#16a34a", label: "正常", cls: "normal" },
    Anxiety:    { color: "#d97706", label: "焦虑", cls: "anxiety" },
    Depression: { color: "#ea580c", label: "抑郁", cls: "depression" },
    Suicidal:   { color: "#dc2626", label: "高危", cls: "suicidal" },
  };

  let range = "7d";
  let alertStatus = "all", alertLevel = "all";
  let pollTimer = null;

  function esc(s) {
    return (s || "").replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ---------------- 指标 + 图表 ---------------- */
  async function loadMetrics() {
    try {
      const r = await fetch(`/api/metrics?range=${range}`);
      const m = await r.json();
      const k = m.kpi;
      $("kpiTotal").textContent = k.total;
      $("kpiHigh").textContent = k.high_risk;
      $("kpiPending").textContent = k.referrals_pending;
      $("kpiConf").textContent = (k.avg_confidence * 100).toFixed(1) + "%";
      const rangeNote = range === "24h" ? "过去 24 小时" : `过去 ${RANGE_HOURS[range] / 24} 天`;
      $("kpiRangeNote").textContent = rangeNote;
      $("kpiRangeDelta").textContent = rangeNote;

      // 风险分布环图
      const donutData = Object.entries(m.risk_distribution).map(([lv, v]) => ({
        label: RISK_META[lv].label, value: v, color: RISK_META[lv].color,
      }));
      FetaCharts.renderDonut($("chartDonut"), donutData);
      $("legendRisk").innerHTML = donutData.map(d =>
        `<span><i style="background:${d.color}"></i>${d.label} ${d.value}</span>`).join("");

      // 趋势双线
      FetaCharts.renderLine($("chartTrend"), m.trend);
      $("legendTrend").innerHTML =
        `<span><i style="background:#4f46e5"></i>动态总量</span>` +
        `<span><i style="background:#dc2626"></i>高危（抑郁+高危）</span>`;

      // 压力源条形图
      FetaCharts.renderBars($("chartBars"),
        m.issues.map(i => ({ name: i.name.replace(/_/g, " "), count: i.count })));
    } catch (e) { /* 网络异常静默，下轮轮询重试 */ }
  }

  /* ---------------- 告警列表 ---------------- */
  async function loadAlerts() {
    const r = await fetch(`/api/alerts?status=${alertStatus}&level=${alertLevel}`);
    const data = await r.json();
    $("alertCounts").textContent = `待处理 ${data.counts.pending} / 共 ${data.counts.total}`;
    const box = $("alertList");
    if (!data.alerts.length) {
      box.innerHTML = '<div class="empty">当前筛选条件下没有告警</div>';
      return;
    }
    box.innerHTML = data.alerts.map(a => {
      const meta = RISK_META[a.risk_level] || { cls: "plain", label: a.risk_level };
      const t = new Date(a.timestamp).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
      return `<div class="alert-row ${a.status === "resolved" ? "resolved" : ""}">
        <span class="time">${t}</span>
        <span class="chip ${meta.cls}"><i class="d"></i>${meta.label}</span>
        <span class="issue">${esc(a.target_issue.replace(/_/g, " "))}${a.referred ? ' <span class="chip plain">已转介</span>' : ""}</span>
        <span class="reason">${esc(a.reason)}</span>
        ${a.status === "pending"
          ? `<button class="btn small" data-resolve="${esc(a.id)}">标记已处理</button>`
          : '<span class="st resolved">✓ 已处理</span>'}
      </div>`;
    }).join("");
    box.querySelectorAll("[data-resolve]").forEach(b =>
      b.addEventListener("click", async () => {
        await fetch("/api/alerts/resolve", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: b.dataset.resolve }),
        });
        loadAlerts(); loadMetrics();
      }));
  }

  /* ---------------- 实时分析 ---------------- */
  async function analyze() {
    const text = $("anaText").value.trim();
    if (!text) { $("anaResult").innerHTML = '<span style="color:#9298a8">请输入文本</span>'; return; }
    const force = $("anaForce").value;
    $("anaResult").innerHTML = '<div class="spinner"></div>分析中…';
    try {
      const r = await fetch("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, force_risk_level: force || null }),
      });
      const d = await r.json();
      if (d.error) { $("anaResult").innerHTML = esc(d.error); return; }
      const meta = RISK_META[d.risk_level] || { cls: "plain", label: d.risk_level };
      const crisis = (d.crisis_resources || []).map(c => `
        <div class="crisis-mini">
          <b style="color:#dc2626">🚨 危机卡（仅 user_facing 下发）</b>
          <div>${esc(c.message || c.summary || c.text)}</div>
          ${c.urgency ? `<div style="color:#9298a8;font-size:12px;margin-top:4px">urgency: ${c.urgency}/5</div>` : ""}
        </div>`).join("");
      $("anaResult").innerHTML = `
        <div class="row"><span class="k">状态</span><b>${esc(d.status)}</b></div>
        <div class="row"><span class="k">风险等级</span><span class="chip ${meta.cls}"><i class="d"></i>${meta.label}</span></div>
        <div class="row"><span class="k">判定来源</span><span>${d.risk_source === "keyword" ? "关键词规则" : "NLP 模型"}（置信度 ${(d.risk_confidence * 100).toFixed(0)}%）</span></div>
        <div class="row"><span class="k">压力源归因</span><b>${esc(d.target_issue.replace(/_/g, " "))}</b></div>
        ${d.referral_logged ? '<div class="referral-flag">⚠ 高危已后台转介（demo_risk_referrals.csv），推荐未拦截</div>' : ""}
        ${d.recommendations.map(rec =>
          `<div class="rec-item">${esc(rec.text)}<div style="color:#9298a8;font-size:11.5px;margin-top:2px">sim ${(rec.similarity_score * 100).toFixed(0)}% · final ${rec.final_score}</div></div>`).join("")}
        ${crisis}`;
      loadAlerts(); loadMetrics();
    } catch (e) {
      $("anaResult").innerHTML = "分析失败，请重试";
    }
  }

  /* ---------------- 控件绑定 ---------------- */
  document.querySelectorAll("#segRange button").forEach(b =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#segRange button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      range = b.dataset.range;
      loadMetrics();
    }));
  document.querySelectorAll("#segStatus button").forEach(b =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#segStatus button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      alertStatus = b.dataset.status;
      loadAlerts();
    }));
  document.querySelectorAll("#segLevel button").forEach(b =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#segLevel button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      alertLevel = b.dataset.level;
      loadAlerts();
    }));
  $("btnAnalyze").addEventListener("click", analyze);
  $("anaText").addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
  });

  loadMetrics();
  loadAlerts();
  pollTimer = setInterval(loadMetrics, 20000);   // 近实时轮询
})();
