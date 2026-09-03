# -*- coding: utf-8 -*-
"""
风险转介记录模块  src/referral.py

负责人：李鹏飞

职责：把高危用户画像「后台记录」下来，供其他业务线（人工客服 / 社工 / 辅导员）消费。
    —— 不拦截推荐、不剥夺用户使用权，是一条独立于推荐轨道的「风险轨道」。

设计原则：
  - 只记录、不拦截：推荐照常，高危用户额外生成一条转介单
  - 落盘为 CSV（data/risk_referrals.csv）；未来可替换为 HTTP 推送 / 消息队列 /
    腾讯文档等 Sink，接口签名不变
  - 隐私：context（原文）可选，默认不落；正式上线建议只存脱敏画像 + 去标识 ID
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional


class ReferralLogger:
    """高危转介记录器：把一条转介单追加写入后台 CSV。"""

    FIELDS = ["timestamp", "risk_level", "target_issue", "reason", "context", "status"]

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        r = cfg.get("referral", {})
        self.enabled = bool(r.get("enabled", True))
        self.output_path = r.get("output", "data/risk_referrals.csv")

    def log(self, risk_level: str, target_issue: str,
            context: str = "", reason: str = "") -> dict:
        """写入一条转介记录，返回该记录 dict（含 logged 标记）。

        context：可选原始文本（转介给业务线同学的上下文）；隐私敏感，默认空。
        """
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "risk_level": risk_level,
            "target_issue": target_issue,
            "reason": reason or f"High_risk_{risk_level}",
            "context": context,
            "status": "pending",  # pending → 业务线处理后置为 resolved
        }
        if not self.enabled:
            record["logged"] = False
            return record

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        is_new = not os.path.exists(self.output_path)
        with open(self.output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({k: record[k] for k in self.FIELDS})
        record["logged"] = True
        return record
