# -*- coding: utf-8 -*-
"""
召回与精排推荐层。

输入（Data Contract）：
    user_vector  : numpy.ndarray (384,)   基于 embedding_prompt 编码的用户向量
    profile_data : dict                   心理画像层（classifier）吐出的结构化数据

profile_data 结构：
    {
      "risk_level": "Normal",                    Normal | Anxiety | Depression | Suicidal
      "target_issue": "Academic_Stress",       8 大类压力源之一
      "strategy_weights": {                    干预策略分类权重
          "relaxation": 0.35, "cognitive": 0.25,
          "healing": 0.25, "lifestyle": 0.15
      },
      "negative_rules": {                      负向黑名单过滤规则
          "exclude_tags": ["fast_paced", "comparison"],
          "max_arousal_score": 0.4
      }
    }

输出（dict）：
    {
      "status": "SUCCESS" | "SUCCESS_WITH_REFERRAL",
      "risk_level": "Normal",
      "recommendations": [
          {"id", "text", "target_issue", "similarity_score", "final_score"}, ...
      ],
      "referral": None | {...},        高危时后台转介记录（risk_level/target_issue/reason/logged）
      "crisis_resources": [...]        高危时附带的危机资源卡片（热线等），不作为唯一输出
    }

五步：①归因硬/软召回 ②余弦相似度 ③多因子加权+黑名单 ④Top-K 截取
      ⑤风险转介（高危后台记录，不拦截、不剥夺使用权）
"""

from __future__ import annotations

import csv
import functools
import os
from typing import Optional

import numpy as np
import yaml

try:
    from src.referral import ReferralLogger   # 项目内包导入（pipeline 场景）
except ImportError:                            # 直接执行 src/recommender.py 时
    from referral import ReferralLogger

# 危机话术库路径（按风险等级分级，见 data/crisis_playbook.yaml）
_PLAYBOOK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "crisis_playbook.yaml")
)


@functools.lru_cache(maxsize=1)
def _load_crisis_playbook() -> dict:
    """加载危机话术库 YAML。lru_cache 避免重复 IO。"""
    with open(_PLAYBOOK_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# 触发后台转介的高危等级默认值（config 未配置时兜底）
DEFAULT_REFERRAL_LEVELS = {"Suicidal"}

# 危机资源卡片默认内容（话术库缺失时兜底）——附在 feed 后，不作为唯一输出
DEFAULT_HOTLINE_CARD = {
    "id": "hotline-001",
    "text": "你现在可能正经历非常艰难的时刻，请立即联系可信赖的人，"
            "或拨打 24 小时心理援助热线 400-161-9995。",
    "target_issue": "Crisis_Intervention",
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
        self.referral_levels = set(
            safety.get("referral_levels", DEFAULT_REFERRAL_LEVELS)
        )
        self.hotline_card = dict(safety.get("hotline", DEFAULT_HOTLINE_CARD))

        # 风险轨道：高危后台转介记录器（不拦截推荐）
        self.referral_logger = ReferralLogger(cfg)

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
        """五步：召回 → 相似度 → 加权+黑名单 → Top-K → 风险转介(高危后台记录，不拦截)。"""
        top_k = top_k or self.top_k
        risk_level = profile_data.get("risk_level", "Normal")

        # ⑤ 风险转介判定（先算，后拼装）——高危只后台记录，不拦截推荐
        status, referral, crisis_resources = self._referral_for(risk_level, profile_data)

        # ① 归因召回：target_issue 硬过滤，不足 top_k 软降级为全量库
        idxs = self._recall(profile_data, top_k)
        recommendations: list[dict] = []
        if idxs:
            # ② 余弦相似度
            sims = self._cosine_similarity(user_vector, self.embeddings[idxs])

            # ③ 多因子加权 + 负向黑名单剔除
            scored = self._rank_and_filter(idxs, sims, profile_data)

            # ④ Top-K 截取
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
            "status": status,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "referral": referral,
            "crisis_resources": crisis_resources,
        }

    # ------------------------ ⑤ 风险转介（不拦截） ------------------------
    def _referral_for(self, risk_level: str, profile_data: dict):
        """高危 → 后台转介 + 附加危机资源；否则正常。返回 (status, referral, resources)。"""
        if risk_level in self.referral_levels:
            referral = self.referral_logger.log(
                risk_level=risk_level,
                target_issue=profile_data.get("target_issue", ""),
                context=profile_data.get("original_text", ""),
            )
            return "SUCCESS_WITH_REFERRAL", referral, [self._crisis_card(risk_level, profile_data)]
        return "SUCCESS", None, []

    def _crisis_card(self, risk_level: str, profile_data: dict) -> dict:
        """按风险等级组装危机话术卡：分级处置动作 + 压力源建议 + 紧急热线。

        话术库缺失或读取异常时回退到单行热线卡，保证高危路径始终有内容下发。
        """
        try:
            playbook = _load_crisis_playbook()
        except Exception:
            return dict(self.hotline_card)

        tiers = playbook.get("tiers", {})
        tier = tiers.get(risk_level) or tiers.get("Normal")
        if not tier:
            return dict(self.hotline_card)

        advice = playbook.get("stressor_advice", {})
        target_issue = profile_data.get("target_issue", "")
        card = {
            "id": f"crisis-{risk_level.lower()}",
            "text": tier["summary"],
            "target_issue": "Crisis_Intervention",
            "level": tier["level"],
            "urgency": tier["urgency"],
            "summary": tier["summary"],
            "actions": list(tier["actions"]),
            "monitor": tier["monitor"],
            "stressor_advice": advice.get(target_issue) or advice.get("General", ""),
        }
        if tier["urgency"] >= 4:
            card["resources"] = [dict(r) for r in playbook.get("resources", [])]
        return card

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
        "risk_level": "Normal",
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
    print("高危画像（Suicidal）：照常推荐 + 后台转介 + 危机资源")
    print("=" * 64)
    high_profile = dict(profile, risk_level="Suicidal", target_issue="Emotional_Health",
                        original_text="（演示）用户表达了自伤念头，需要人工介入。")
    result2 = rec.recommend(user_vec, high_profile)
    print(f"status={result2['status']}")
    print(f"referral={result2['referral']}")
    print(f"crisis_resources={result2['crisis_resources']}")
    print("--- 博文推荐（未被拦截，照常返回）---")
    for r in result2["recommendations"]:
        print(f"  [id={r['id']}] ({r['target_issue']}) {r['text']}")
