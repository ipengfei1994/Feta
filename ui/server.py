# -*- coding: utf-8 -*-
"""
Feta UI 本地服务（标准库实现，零新增依赖）。

页面：
    GET /            信息流（类微博 feed）
    GET /dashboard   监测后台

API：
    GET  /api/feed?page=N[&crisis=1]   分页动态（时间线倒序），crisis=1 演示危机卡渲染路径
    GET  /api/metrics?range=24h|7d|30d 核心指标 + 图表聚合数据（风险分布 / 趋势 / 压力源）
    GET  /api/alerts?status=&level=    告警列表（内存态，落盘 data/ui_alerts.csv）
    POST /api/alerts/resolve           标记告警已处理 {"id": "..."}
    POST /api/analyze                  单条文本实时分析 {"text": "...", "force_risk_level": 可选}

演示数据边界（重要护栏）：
    - 启动时对演示语料逐条跑真实 pipeline（MiniLM + 风险模型），分析结果全量缓存。
    - 演示转介写入 data/demo_risk_referrals.csv，绝不污染真实 data/risk_referrals.csv。
    - 危机卡只下发 user_facing 字段；staff_actions（内部运营字段）永不出 API。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)  # pipeline/referral 均按相对路径读 data/、configs/

import yaml

from src.pipeline import Pipeline
from ui.mock_data import build_posts, MOCK_COMMENTS, AUTHORS

PORT = 8765
TEMPLATE_DIR = os.path.join(BASE_DIR, "ui", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "ui", "static")
DEMO_REFERRAL_CSV = "data/demo_risk_referrals.csv"
ALERTS_CSV = "data/ui_alerts.csv"

RISK_COLORS = {"Normal": "normal", "Anxiety": "anxiety",
               "Depression": "depression", "Suicidal": "suicidal"}


# ---------------------------------------------------------------------------
# 启动期：语料分析缓存
# ---------------------------------------------------------------------------
class AppState:
    """全局状态：演示动态 + 逐条分析结果 + 告警列表。"""

    def __init__(self):
        with open("configs/config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 演示转介走独立 CSV，不污染真实转介记录
        cfg.setdefault("referral", {})["output"] = DEMO_REFERRAL_CSV
        self.pipeline = Pipeline("configs/config.yaml")
        self.pipeline.recommender.referral_logger.output = DEMO_REFERRAL_CSV

        self.posts = build_posts()
        self.analysis: dict[str, dict] = {}
        for p in self.posts:
            r = self.pipeline.run(p["text"])
            self.analysis[p["id"]] = {
                "risk_level": r["risk_level"],
                "target_issue": r["profile"].get("target_issue", "General"),
                "risk_source": r["profile"].get("risk_source", "model"),
                "risk_confidence": r["profile"].get("risk_confidence", 0.0),
                "status": r["status"],
                "recommendations": r["recommendations"],
            }
        self.alerts = self._build_alerts()

    def _build_alerts(self) -> list[dict]:
        """告警 = 非 Normal 动态聚合；Suicidal 附加「已转介」标记（双轨模型）。"""
        alerts = []
        for p in self.posts:
            a = self.analysis[p["id"]]
            if a["risk_level"] == "Normal":
                continue
            reason = {
                "Anxiety": "模型检出焦虑信号，建议关注",
                "Depression": "模型检出抑郁信号，建议人工复核",
                "Suicidal": "命中高危等级，已自动后台转介",
            }[a["risk_level"]]
            alerts.append({
                "id": f"a{p['id']}",
                "post_id": p["id"],
                "timestamp": p["created_at"],
                "risk_level": a["risk_level"],
                "target_issue": a["target_issue"],
                "reason": reason,
                "status": "pending",
                "referred": a["risk_level"] in ("Suicidal",),
            })
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        self._flush_alerts(alerts)
        return alerts

    @staticmethod
    def _flush_alerts(alerts: list[dict]):
        os.makedirs(os.path.dirname(ALERTS_CSV), exist_ok=True)
        with open(ALERTS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "id", "post_id", "timestamp", "risk_level",
                "target_issue", "reason", "status", "referred"])
            w.writeheader()
            w.writerows(alerts)


STATE: AppState = None  # main() 中初始化


# ---------------------------------------------------------------------------
# API 组装
# ---------------------------------------------------------------------------
def _post_view(p: dict) -> dict:
    """对外输出一条动态（含演示性风险标签，供「监测模式」切换展示）。"""
    a = STATE.analysis[p["id"]]
    return {**p, "risk_level": a["risk_level"], "risk_confidence": a["risk_confidence"]}


def _reco_card(post_id: str) -> list[dict]:
    """把某条动态的 pipeline 推荐结果包装成「无痕」普通用户推文。

    产品逻辑：干预内容以真实用户口吻原生植入信息流（原生内容推荐），
    不带任何品牌标签、风险标签或匹配度指标——用户视角完全无痕；
    数据侧仍可通过 id 前缀 `rec-` 与埋点区分推荐内容。
    作者/时间/互动数由 post_id 确定性哈希生成，同一刷新会话内稳定。
    """
    a = STATE.analysis.get(post_id)
    if not a or not a["recommendations"]:
        return []
    anchor = next((p for p in STATE.posts if p["id"] == post_id), None)
    if not anchor:
        return []

    import hashlib
    from datetime import datetime, timedelta
    base = int(hashlib.md5(post_id.encode()).hexdigest(), 16)
    anchor_dt = datetime.fromisoformat(anchor["created_at"])
    age_min = max(2.0, (datetime.now() - anchor_dt).total_seconds() / 60)

    out = []
    for i, rec in enumerate(a["recommendations"][:2]):
        author = AUTHORS[(base >> (i * 3)) % len(AUTHORS)]
        # 发布时间落在锚点动态之后、当前时刻之前的合理区间内
        shift = max(1, min(int(age_min * 0.4), 5 + (base >> (i * 5)) % 30))
        ts = anchor_dt + timedelta(minutes=shift)
        if ts >= datetime.now():
            ts = datetime.now() - timedelta(minutes=1 + (base % 9))
        out.append({
            "type": "post",
            "id": f"rec-{post_id}-{i}",
            "author": author[0],
            "handle": author[1],
            "text": rec["text"],
            "created_at": ts.isoformat(timespec="seconds"),
            "likes": base % 90 + i * 11,
            "comments": (base >> 4) % 12,
            "shares": (base >> 8) % 9,
            "image": None,
        })
    return out


def api_feed(qs: dict) -> dict:
    page = max(1, int(qs.get("page", ["1"])[0]))
    size = 8
    posts = STATE.posts  # 已按时间倒序
    total = len(posts)
    start = (page - 1) * size
    page_posts = posts[start:start + size]

    cards: list[dict] = []
    for i, p in enumerate(page_posts):
        cards.append({"type": "post", **_post_view(p)})
        if p.get("is_own"):
            # 用户刚发布的动态：推荐内容紧随其后插入（发布后立刻可见）
            cards.extend(_reco_card(p["id"]))
        elif (i + 1) % 4 == 0:                    # 常规节奏：每 4 条插入无痕推荐
            cards.extend(_reco_card(p["id"]))

    crisis_card = None
    if qs.get("crisis", ["0"])[0] == "1":
        # 演示：强制高危会话，验证危机卡渲染（仅 user_facing 字段下发）
        r = STATE.pipeline.run(page_posts[0]["text"], force_risk_level="Suicidal")
        if r["crisis_resources"]:
            c = r["crisis_resources"][0]
            crisis_card = {"type": "crisis", "id": c.get("id", "crisis"),
                           "text": c.get("text", ""),
                           "summary": c.get("summary", ""),
                           "message": c.get("message", ""),
                           "resources": c.get("resources", [])}
        # 危机演示的转介记录（独立演示 CSV）
        if r["referral"]:
            STATE.pipeline.recommender.referral_logger.log(
                risk_level="Suicidal",
                target_issue=r["profile"].get("target_issue", ""),
                context="ui_crisis_demo")

    return {"page": page, "page_size": size, "total": total,
            "has_more": start + size < total, "cards": cards,
            "crisis_card": crisis_card,
            "demo_comments": [{"handle": h, "text": t} for h, t in MOCK_COMMENTS[:3]]}


def _in_range(iso_ts: str, hours: float) -> bool:
    from datetime import datetime, timedelta
    t = datetime.fromisoformat(iso_ts)
    return datetime.now() - t <= timedelta(hours=hours)


def api_publish(body: dict) -> dict:
    """发布动态：走真实 pipeline 分析，插到信息流顶部，随后紧跟无痕推荐。"""
    text = (body.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}
    from datetime import datetime
    pid = f"p{int(datetime.now().timestamp() * 1000)}"
    post = {
        "id": pid,
        "author": "Afu",
        "handle": "afu",
        "text": text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "likes": 0, "comments": 0, "shares": 0,
        "image": None,
        "is_own": True,
    }
    # 真实四段式 pipeline：分析结果进入与演示语料相同的缓存与指标口径
    r = STATE.pipeline.run(text)
    STATE.analysis[pid] = {
        "risk_level": r["risk_level"],
        "target_issue": r["profile"].get("target_issue", "General"),
        "risk_source": r["profile"].get("risk_source", "model"),
        "risk_confidence": r["profile"].get("risk_confidence", 0.0),
        "status": r["status"],
        "recommendations": r["recommendations"],
    }
    if STATE.posts and STATE.posts[0].get("is_own"):
        # 简化演示：同一会话多次发布时只保留最新一条自带推荐的动态
        old = STATE.posts.pop(0)
        STATE.analysis.pop(old["id"], None)
    STATE.posts.insert(0, post)
    recos = _reco_card(pid)
    return {
        "ok": True,
        "post": {**post, "risk_level": r["risk_level"]},
        "risk_level": r["risk_level"],
        "target_issue": STATE.analysis[pid]["target_issue"],
        "followed_recommendations": len(recos),
    }


def api_metrics(qs: dict) -> dict:
    rng = (qs.get("range", ["7d"])[0] or "7d").lower()
    hours = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30}.get(rng, 168)
    bucket = 1 if rng == "24h" else 24          # 24h 按小时，其余按天

    posts = [p for p in STATE.posts if _in_range(p["created_at"], hours)]
    risks = {"Normal": 0, "Anxiety": 0, "Depression": 0, "Suicidal": 0}
    issues: dict[str, int] = {}
    confs = []
    for p in posts:
        a = STATE.analysis[p["id"]]
        risks[a["risk_level"]] += 1
        issues[a["target_issue"]] = issues.get(a["target_issue"], 0) + 1
        if a["risk_source"] == "model":
            confs.append(a["risk_confidence"])

    # 趋势序列（总动态 + 高危两条线）
    from datetime import datetime, timedelta
    now = datetime.now()
    nbuckets = hours // bucket
    trend = [{"label": "", "total": 0, "high": 0} for _ in range(nbuckets)]
    for p in posts:
        a = STATE.analysis[p["id"]]
        delta_h = (now - datetime.fromisoformat(p["created_at"])).total_seconds() / 3600
        idx = nbuckets - 1 - int(delta_h // bucket)
        if 0 <= idx < nbuckets:
            trend[idx]["total"] += 1
            if a["risk_level"] in ("Depression", "Suicidal"):
                trend[idx]["high"] += 1
    for i, t in enumerate(trend):
        if rng == "24h":
            t["label"] = (now - timedelta(hours=nbuckets - 1 - i)).strftime("%H:00")
        else:
            t["label"] = (now - timedelta(days=nbuckets - 1 - i)).strftime("%m-%d")

    pending = sum(1 for a in STATE.alerts
                  if a["status"] == "pending" and _in_range(a["timestamp"], hours))
    kpi = {
        "total": len(posts),
        "high_risk": risks["Depression"] + risks["Suicidal"],
        "referrals_pending": pending,
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
    }
    issue_sorted = sorted(issues.items(), key=lambda kv: -kv[1])[:8]
    return {"range": rng, "kpi": kpi, "risk_distribution": risks,
            "trend": trend, "issues": [{"name": k, "count": v} for k, v in issue_sorted]}


def api_alerts(qs: dict) -> dict:
    status = (qs.get("status", ["all"])[0] or "all").lower()
    level = (qs.get("level", ["all"])[0] or "all").lower()
    rows = STATE.alerts
    if status != "all":
        rows = [a for a in rows if a["status"] == status]
    if level != "all":
        rows = [a for a in rows if a["risk_level"].lower() == level]
    counts = {"pending": sum(1 for a in STATE.alerts if a["status"] == "pending"),
              "total": len(STATE.alerts)}
    return {"alerts": rows, "counts": counts}


def api_alerts_resolve(body: dict) -> dict:
    aid = body.get("id", "")
    for a in STATE.alerts:
        if a["id"] == aid:
            a["status"] = "resolved"
            AppState._flush_alerts(STATE.alerts)
            return {"ok": True, "alert": a}
    return {"ok": False, "error": "alert not found"}


def api_analyze(body: dict) -> dict:
    text = (body.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}
    force = body.get("force_risk_level") or None
    r = STATE.pipeline.run(text, force_risk_level=force)
    a = STATE.analysis.get("__last__", {})
    prof = r["profile"]
    # 危机卡只下发 user_facing 字段（护栏：staff_actions 永不出 API）
    crisis = [{k: c[k] for k in ("id", "text", "summary", "message", "urgency")
               if k in c} for c in r.get("crisis_resources", [])]
    return {
        "status": r["status"],
        "risk_level": r["risk_level"],
        "risk_source": prof.get("risk_source", "model"),
        "risk_confidence": prof.get("risk_confidence", 0.0),
        "target_issue": prof.get("target_issue", "General"),
        "strategy_weights": prof.get("strategy_weights", {}),
        "recommendations": r["recommendations"],
        "crisis_resources": crisis,
        "referral_logged": bool(r.get("referral")),
    }


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".json": "application/json",
        ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默默认访问日志
        pass

    # ---------------- helpers ----------------
    def _send(self, code: int, data: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, rel_path: str, root: str):
        path = os.path.normpath(os.path.join(root, rel_path))
        if not path.startswith(os.path.normpath(root)) or not os.path.isfile(path):
            self._json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            self._send(200, f.read(), MIME.get(ext, "application/octet-stream"))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---------------- GET ----------------
    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path == "/":
                self._file("feed.html", TEMPLATE_DIR)
            elif u.path == "/dashboard":
                self._file("dashboard.html", TEMPLATE_DIR)
            elif u.path.startswith("/static/"):
                self._file(u.path[len("/static/"):], STATIC_DIR)
            elif u.path == "/api/feed":
                self._json(api_feed(qs))
            elif u.path == "/api/metrics":
                self._json(api_metrics(qs))
            elif u.path == "/api/alerts":
                self._json(api_alerts(qs))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 500)

    # ---------------- POST ----------------
    def do_POST(self):
        u = urlparse(self.path)
        body = self._body()
        try:
            if u.path == "/api/alerts/resolve":
                self._json(api_alerts_resolve(body))
            elif u.path == "/api/analyze":
                self._json(api_analyze(body))
            elif u.path == "/api/publish":
                self._json(api_publish(body))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 500)


def main():
    global STATE
    print("正在初始化 pipeline 与演示语料分析（首次启动需加载本地模型）...")
    STATE = AppState()
    print(f"语料 {len(STATE.posts)} 条分析完成，"
          f"告警 {len(STATE.alerts)} 条（转介落盘 {DEMO_REFERRAL_CSV}）")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Feta UI: http://127.0.0.1:{PORT}/  |  后台: http://127.0.0.1:{PORT}/dashboard")
    srv.serve_forever()


if __name__ == "__main__":
    main()
