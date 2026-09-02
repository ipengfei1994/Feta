# -*- coding: utf-8 -*-
"""
3.3 召回与精排推荐层  src/recommender.py

负责人：李鹏飞

输入（Data Contract，与 3.1 / 3.2 对齐）：
    user_vector  : numpy.ndarray (384,)   —— 基于 embedding_prompt 编码的用户向量
    profile_data : dict                   —— 3.1 吐出的结构化心理画像

profile_data 结构：
    {
      "risk_level": "Level_2_Moderate",        # Level_1_Low ~ Level_4_High
      "target_issue": "Academic_Stress",       # 8 大类压力源之一
      "strategy_weights": {                    # 干预策略分类权重
          "relaxation": 0.35, "cognitive": 0.25,
          "healing": 0.25, "lifestyle": 0.15
      },
      "negative_rules": {                      # 负向黑名单过滤规则
          "exclude_tags": ["fast_paced", "comparison"],
          "max_arousal_score": 0.4
      }
    }

输出（dict）：
    {
      "status": "SUCCESS" | "CIRCUIT_BREAKER_TRIGGERED",
      "risk_level": "Level_2_Moderate",
      "recommendations": [
          {"id", "text", "target_issue", "similarity_score", "final_score"}, ...
      ]
    }

五步：①安全断路器 ②归因硬/软召回 ③余弦相似度 ④多因子加权+黑名单 ⑤Top-K 截取
"""

from __future__ import annotations

import csv
from typing import Optional

import numpy as np

# 触发断路器的高危等级默认值（config 未配置时兜底）
DEFAULT_HIGH_RISK_LEVELS = {"Level_4_High", "Level_4_High_Risk", "Suicide"}

# 危机热线卡片默认内容（config 未配置时兜底）
DEFAULT_HOTLINE_CARD = {
    "id": "hotline-001",
    "text": "你现在可能正经历非常艰难的时刻，请立即联系可信赖的人，"
            "或拨打 24 小时心理援助热线 400-161-9995。",
    "target_issue": "Crisis_Intervention",
    "similarity_score": 1.0,
    "final_score": 1.0,
}


class Recommender:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.paths = cfg.get("paths", {})
        rec = cfg.get("recommender", {})
        self.top_k = int(rec.get("top_k", 5))
        w = rec.get("weights", {})
        self.w_sim = float(w.get("w_sim", 0.8))
        self.w_boost = float(w.get("w_boost", 0.2))

        safety = cfg.get("safety", {})
        self.high_risk_levels = set(
            safety.get("circuit_breaker_levels", DEFAULT_HIGH_RISK_LEVELS)
        )
        self.hotline_card = dict(safety.get("hotline", DEFAULT_HOTLINE_CARD))
        # 补齐热线的相似度/最终分字段（保证输出结构统一）
        self.hotline_card.setdefault("similarity_score", 1.0)
        self.hotline_card.setdefault("final_score", 1.0)

        self.items: list[dict] = []          # 干预池元数据（含 tags/boost/strategy）
        self.embeddings: np.ndarray = None   # (N, 384) 已归一化

    # ------------------------------ 加载 ------------------------------
    def load(self, csv_path: Optional[str] = None) -> "Recommender":
        """读取干预池（含离线预计算的 384 维 embedding），做一次归一化。"""
        csv_path = csv_path or self.paths.get("interventions", "data/interventions.csv")
        items, vecs = [], []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append({
                    "id": int(row["id"]),
                    "text": row["text"],
                    "target_issue": row["target_issue"],
                    "boost_score": float(row.get("boost_score", 0.0)),
                    "tags": {t.strip() for t in row.get("tags", "").split(",") if t.strip()},
                    "arousal_score": float(row.get("arousal_score", 0.0)),
                    "strategy": row.get("strategy", ""),
                })
                vecs.append([float(x) for x in row["embedding"].split()])

        self.items = items
        self.embeddings = np.array(vecs, dtype="float32")
        # 归一化：cosine 相似度退化为点积
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        return self

    # ------------------------------ 主入口 ------------------------------
    def recommend(self, user_vector: np.ndarray,
                  profile_data: dict, top_k: Optional[int] = None) -> dict:
        """五步：断路器 → 召回 → 相似度 → 加权+黑名单 → Top-K。"""
        top_k = top_k or self.top_k
        risk_level = profile_data.get("risk_level", "Level_1_Low")

        # ① 安全断路器：高危直接降级热线，切断博文推荐
        if risk_level in self.high_risk_levels:
            return {
                "status": "CIRCUIT_BREAKER_TRIGGERED",
                "risk_level": risk_level,
                "recommendations": [dict(self.hotline_card)],
            }

        # ② 归因召回：target_issue 硬过滤，不足 top_k 软降级为全量库
        idxs = self._recall(profile_data, top_k)
        if not idxs:
            return {"status": "SUCCESS", "risk_level": risk_level, "recommendations": []}

        # ③ 余弦相似度
        sims = self._cosine_similarity(user_vector, self.embeddings[idxs])

        # ④ 多因子加权 + 负向黑名单剔除
        scored = self._rank_and_filter(idxs, sims, profile_data)

        # ⑤ Top-K 截取
        scored.sort(key=lambda x: x[2], reverse=True)
        recommendations = [
            {
                "id": self.items[i]["id"],
                "text": self.items[i]["text"],
                "target_issue": self.items[i]["target_issue"],
                "similarity_score": round(float(sim), 4),
                "final_score": round(float(final), 4),
            }
            for i, sim, final in scored[:top_k]
        ]
        return {
            "status": "SUCCESS",
            "risk_level": risk_level,
            "recommendations": recommendations,
        }

    # ------------------------ ② 归因硬/软召回 ------------------------
    def _recall(self, profile_data: dict, top_k: int) -> list[int]:
        """target_issue 精确召回；命中量不足 top_k 时软降级为全量召回。"""
        target = profile_data.get("target_issue")
        hard = [i for i, it in enumerate(self.items) if it["target_issue"] == target]
        if len(hard) >= top_k:
            return hard
        # 软降级：不足 top_k，退回全量库（负向规则会在精排阶段兜底过滤）
        return list(range(len(self.items)))

    # ------------------------ ③ 余弦相似度 ------------------------
    def _cosine_similarity(self, user_vector: np.ndarray,
                           item_vectors: np.ndarray) -> np.ndarray:
        """用户向量 vs 候选集 Item 向量，返回 [0,1] 区间的余弦相似度。

        说明：item_vectors 在 load() 已归一化，此处只需归一化 user_vector，
        再点积即可得到余弦相似度（比逐对计算快一个量级）。
        """
        u = np.asarray(user_vector, dtype="float32")
        n = np.linalg.norm(u)
        u = u / (n + 1e-8)
        return (item_vectors @ u).astype("float32")

    # ------------------- ④ 多因子加权 + 黑名单 -------------------
    def _rank_and_filter(self, idxs: list[int], sims: np.ndarray,
                         profile_data: dict) -> list[tuple[int, float, float]]:
        """Score = w_sim * S_sim + w_boost * S_boost，并剔除负向规则命中项。"""
        neg = profile_data.get("negative_rules", {})
        exclude_tags = set(neg.get("exclude_tags", []))
        max_arousal = float(neg.get("max_arousal_score", 1.0))
        strategy_weights = profile_data.get("strategy_weights", {})

        scored: list[tuple[int, float, float]] = []  # (全局索引, 相似度, 最终分)
        for j, i in enumerate(idxs):
            it = self.items[i]
            # 负向黑名单：命中 exclude_tags 直接剔除
            if it["tags"] & exclude_tags:
                continue
            # 负向黑名单：唤醒度超出阈值剔除
            if it["arousal_score"] > max_arousal:
                continue
            # 正能量 boost（用 strategy_weights 对所属策略做轻调制，缺省不调制）
            s_boost = it["boost_score"] * strategy_weights.get(it["strategy"], 1.0)
            final = self.w_sim * float(sims[j]) + self.w_boost * s_boost
            scored.append((i, float(sims[j]), final))
        return scored


# ---------------------------------------------------------------------------
# 独立测试（mock 用户向量 + 画像，不依赖 classifier/embedder）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import yaml

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rec = Recommender(cfg).load()

    # 构造指向「容貌焦虑」的用户向量：取该主题 Item 向量的均值 + 噪声
    issue = "Appearance_Anxiety"
    issue_emb = rec.embeddings[[i for i, it in enumerate(rec.items)
                                if it["target_issue"] == issue]].mean(axis=0)
    rng = np.random.default_rng(0)
    user_vec = issue_emb + 0.1 * rng.standard_normal(issue_emb.shape)

    profile = {
        "risk_level": "Level_2_Moderate",
        "target_issue": issue,
        "strategy_weights": {"relaxation": 0.35, "cognitive": 0.25,
                             "healing": 0.25, "lifestyle": 0.15},
        "negative_rules": {"exclude_tags": ["fast_paced", "comparison"],
                           "max_arousal_score": 0.4},
    }

    print("=" * 64)
    print(f"正常画像推荐（target_issue={issue}）：")
    print("=" * 64)
    result = rec.recommend(user_vec, profile)
    print(f"status={result['status']}  risk_level={result['risk_level']}")
    for r in result["recommendations"]:
        print(f"  [id={r['id']} | sim={r['similarity_score']:.3f} | "
              f"final={r['final_score']:.3f}] ({r['target_issue']}) {r['text']}")

    print("\n" + "=" * 64)
    print("高危画像（Level_4_High）触发断路器：")
    print("=" * 64)
    high_profile = dict(profile, risk_level="Level_4_High", target_issue="Emotional_Health")
    result2 = rec.recommend(user_vec, high_profile)
    print(f"status={result2['status']}")
    for r in result2["recommendations"]:
        print(f"  [id={r['id']}] ({r['target_issue']}) {r['text']}")
