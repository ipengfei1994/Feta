/* ============================================================
   feed.js — 信息流页逻辑
   - 无限滚动：IntersectionObserver 触底加载下一页
   - 互动：点赞（本地态切换）、评论展开 / 本地追加
   - 危机卡演示：顶栏按钮 → /api/feed?crisis=1，重载第一页
   - 风险标签显隐切换（演示监测视角）
   ============================================================ */
(function () {
  "use strict";

  const feed = document.getElementById("feed");
  const sentinel = document.getElementById("sentinel");
  const feedWrap = document.querySelector(".layout .cols");

  let page = 0, hasMore = true, loading = false;
  let crisisMode = false;
  let highlightNext = false;
  let demoComments = [];
  const likedSet = new Set();

  const ICONS = {
    heart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l1 1L12 21.2l7.8-7.8 1-1a5.5 5.5 0 0 0 0-7.8z"/></svg>',
    comment: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.4 8.4 0 0 1-8.5 8.3c-1.4 0-2.8-.3-4-1L3 20l1.3-4.3a8.1 8.1 0 0 1-1.3-4.2A8.4 8.4 0 0 1 11.5 3.2 8.4 8.4 0 0 1 21 11.5z"/></svg>',
    share: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3l4 4-4 4M21 7H9a6 6 0 0 0-6 6v1"/></svg>',
  };
  const RISK_LABEL = { Normal: "正常", Anxiety: "焦虑", Depression: "抑郁", Suicidal: "高危" };

  function esc(s) {
    return (s || "").replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function hue(handle) {
    let h = 0;
    for (const ch of handle) h = (h * 31 + ch.charCodeAt(0)) % 360;
    return h;
  }
  function avatarHtml(handle, name) {
    const h = hue(handle);
    const initials = name.split(/\s+/).map(w => w[0]).slice(0, 2).join("").toUpperCase();
    return `<div class="avatar" style="background:linear-gradient(135deg,hsl(${h},62%,52%),hsl(${(h+40)%360},62%,42%))">${esc(initials)}</div>`;
  }
  function timeAgo(iso) {
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return "刚刚";
    if (s < 3600) return Math.floor(s / 60) + " 分钟前";
    if (s < 86400) return Math.floor(s / 3600) + " 小时前";
    if (s < 86400 * 7) return Math.floor(s / 86400) + " 天前";
    return new Date(iso).toLocaleDateString("zh-CN");
  }

  function postCard(p) {
    const img = p.image ? `
      <div class="img" style="background:linear-gradient(135deg,hsl(${p.image.hue},55%,72%),hsl(${(p.image.hue+50)%360},60%,45%))">
        <span>${p.image.emoji}</span><div class="cap">${esc(p.image.desc)}</div>
      </div>` : "";
    const chip = p.risk_level
      ? `<span class="chip ${p.risk_level.toLowerCase()} risk-chip"><i class="d"></i>${RISK_LABEL[p.risk_level] || p.risk_level}</span>`
      : "";
    const recoTag = p.id.startsWith("rec-")
      ? '<span class="chip plain reco-tag">✦ Feta 推荐</span>'
      : "";
    const liked = likedSet.has(p.id);
    const likes = p.likes + (liked ? 1 : 0);
    const commentsHtml = `
      <div class="comments" data-cid="${p.id}">
        <div class="c-list"></div>
        <div class="comment-input">
          <input type="text" placeholder="写点鼓励的话…" />
          <button class="btn small primary">发送</button>
        </div>
      </div>`;
    return `
      <article class="card post" data-pid="${p.id}">
        <div class="head">
          ${avatarHtml(p.handle, p.author)}
          <div class="who">
            <div class="name">${esc(p.author)}${p.is_own ? ' <span class="own-tag">我</span>' : ""}</div>
            <div class="meta"><span>@${esc(p.handle)}</span><span>·</span><span>${timeAgo(p.created_at)}</span>${chip}${recoTag}</div>
          </div>
          ${p.is_own ? '<button class="link-btn">高亮相关推荐</button>' : ""}
        </div>
        <div class="body">${esc(p.text)}</div>
        ${img}
        <div class="actions">
          <button class="act like ${liked ? "liked" : ""}" data-act="like">${ICONS.heart}<span>${likes}</span></button>
          <button class="act" data-act="comment">${ICONS.comment}<span>${p.comments}</span></button>
          <button class="act" data-act="share">${ICONS.share}<span>${p.shares}</span></button>
        </div>
        ${commentsHtml}
      </article>`;
  }

  function crisisCard(c) {
    const res = (c.resources || []).map(r =>
      `<div class="res">• ${esc(r.name || r.region || "")} ${esc(r.phone || r.contact || "")}</div>`).join("");
    return `
      <div class="crisis-card" id="crisisBanner">
        <h3>🚨 你并不孤单，帮助就在身边</h3>
        <p>${esc(c.message || c.summary || c.text)}</p>
        <div class="hotline">24 小时心理援助热线 400-161-9995</div>
        ${res}
      </div>`;
  }

  function bindPost(root) {
    root.querySelectorAll(".post").forEach(card => {
      const pid = card.dataset.pid;
      // 点赞
      const likeBtn = card.querySelector(".like");
      likeBtn.addEventListener("click", () => {
        const on = likedSet.has(pid);
        on ? likedSet.delete(pid) : likedSet.add(pid);
        likeBtn.classList.toggle("liked", !on);
        likeBtn.querySelector("span").textContent =
          parseInt(likeBtn.querySelector("span").textContent) + (on ? -1 : 1);
      });
      // 评论展开 / 发送
      const cbtn = card.querySelector('[data-act="comment"]');
      const cwrap = card.querySelector(".comments");
      const clist = cwrap.querySelector(".c-list");
      let seeded = false;
      const seed = () => {
        if (seeded) return; seeded = true;
        demoComments.forEach(c => clist.insertAdjacentHTML("beforeend", commentHtml(c.handle, c.text)));
      };
      cbtn.addEventListener("click", () => {
        seed();
        cwrap.classList.toggle("open");
      });
      cwrap.querySelector(".comment-input .btn").addEventListener("click", () => {
        const inp = cwrap.querySelector("input");
        const v = inp.value.trim();
        if (!v) return;
        clist.insertAdjacentHTML("beforeend", commentHtml("me", v));
        inp.value = "";
        cbtn.querySelector("span").textContent = parseInt(cbtn.querySelector("span").textContent) + 1;
      });
      // 演示讲解：高亮自己的博文 + 其后紧跟的系统推荐（再点一次取消）
      const linkBtn = card.querySelector(".link-btn");
      if (linkBtn) linkBtn.addEventListener("click", () => {
        const on = !card.classList.contains("linked");
        document.querySelectorAll(".post.linked").forEach(x =>
          x.classList.remove("linked", "linked-own", "linked-rec"));
        let count = 0;
        if (on) {
          card.classList.add("linked", "linked-own");
          let sib = card.nextElementSibling;
          while (sib && sib.classList.contains("post") && sib.dataset.pid.startsWith("rec-")) {
            sib.classList.add("linked", "linked-rec");
            count++;
            sib = sib.nextElementSibling;
          }
          card.dataset.recoCount = count;
        }
        linkBtn.textContent = on ? "取消高亮" : "高亮相关推荐";
      });
    });
  }
  function commentHtml(handle, text) {
    const name = handle === "me" ? "我" : "@" + handle;
    return `<div class="comment">${avatarHtml(handle, handle === "me" ? "Me" : handle)}
      <div class="bubble"><span class="h">${esc(name)}</span>${esc(text)}</div></div>`;
  }

  async function loadPage() {
    if (loading || !hasMore) return;
    loading = true;
    sentinel.innerHTML = '<div class="spinner"></div>正在加载…';
    try {
      const url = `/api/feed?page=${page + 1}` + (crisisMode ? "&crisis=1" : "");
      const r = await fetch(url);
      const data = await r.json();
      demoComments = data.demo_comments || demoComments;
      page = data.page; hasMore = data.has_more;

      const crisis = crisisMode && data.crisis_card ? crisisCard(data.crisis_card) : "";
      if (page === 1) {
        document.getElementById("crisis-slot").innerHTML = crisis;
        feed.innerHTML = data.cards.map(c => postCard(c)).join("");
        if (highlightNext) {
          highlightNext = false;
          const first = feed.querySelector(".post");
          if (first) {
            first.classList.add("flash");
            first.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }
      } else {
        feed.insertAdjacentHTML("beforeend",
          data.cards.map(c => postCard(c)).join(""));
      }
      bindPost(feed);
      sentinel.innerHTML = hasMore ? "" : "— 已经到底啦 —";
    } catch (e) {
      sentinel.innerHTML = "加载失败，请重试";
    }
    loading = false;
  }

  // 无限滚动
  new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) loadPage();
  }, { rootMargin: "400px" }).observe(sentinel);

  // 顶栏控件
  document.getElementById("btnCrisis").addEventListener("click", e => {
    crisisMode = !crisisMode;
    e.currentTarget.classList.toggle("primary", crisisMode);
    e.currentTarget.textContent = crisisMode ? "退出高危演示" : "模拟高危会话";
    page = 0; hasMore = true;
    loadPage();
  });
  document.getElementById("btnRiskTags").addEventListener("click", e => {
    feed.classList.toggle("show-risk");
    const on = feed.classList.contains("show-risk");
    e.currentTarget.classList.toggle("primary", on);
    e.currentTarget.textContent = on ? "隐藏风险标签" : "显示风险标签";
  });

  // 侧栏关注按钮：点击切换关注态
  document.querySelectorAll(".follow-list .btn").forEach(b =>
    b.addEventListener("click", () => {
      const on = b.classList.toggle("primary");
      b.textContent = on ? "已关注" : "关注";
      if (on) b.classList.remove("primary");
    }));

  // 发布：走真实 pipeline，发布后自己的推文置顶并紧跟无痕推荐
  const composerText = document.getElementById("composerText");
  const btnPublish = document.getElementById("btnPublish");
  async function publishText(text) {
    if (!text) { composerText.focus(); return false; }
    btnPublish.disabled = true; btnPublish.textContent = "发布中…";
    let ok = false;
    try {
      const r = await fetch("/api/publish", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const d = await r.json();
      if (d.ok) {
        ok = true;
        highlightNext = true;
        page = 0; hasMore = true;
        await loadPage();
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    } catch (e) { /* 静默：下次发布再试 */ }
    btnPublish.disabled = false; btnPublish.textContent = "发布";
    return ok;
  }
  btnPublish.addEventListener("click", async () => {
    const text = composerText.value.trim();
    if (await publishText(text)) composerText.value = "";
  });

  loadPage();
})();
