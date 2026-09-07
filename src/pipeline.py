# -*- coding: utf-8 -*-
"""
主流程编排 Pipeline。

把四个模块串成一条流水线：
    原始文本 → preprocess → classifier(profile_data) → embedder(user_vector)
             → recommender(user_vector, profile_data) → Top-K Feed

对外只暴露 run_pipeline()，UI 层只调这一个入口即可。

测试钩子：run_pipeline(raw_text, force_risk_level="Suicidal")
    可强制覆盖 classifier 输出的风险等级，用于 UI 阶段验证危机卡片 / 转介落盘
    渲染路径。生产代码不传此参数，行为与原版完全一致。
"""

from __future__ import annotations

import os
import sys

# 保证既能 `python src/pipeline.py` 也能 `python -m src.pipeline` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src import preprocess, classifier, embedder
from src.recommender import Recommender


class Pipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.embedder = embedder.Embedder()          # auto: BERT / 哈希基线
        self.recommender = Recommender(self.config).load()

    def run(self, raw_text: str, force_risk_level: str | None = None) -> dict:
        """端到端：返回结构化结果，UI 直接渲染。

        force_risk_level: 仅供 UI 测试 / Demo 使用，可强制覆盖 classifier 输出的
                          风险等级（如 "Suicidal"），用于验证危机卡片 UI 渲染。
                          生产路径不传，行为与原版完全一致。
        """
        # 1. 清洗
        cleaned = preprocess.clean(raw_text)
        # 2. 分类 → profile_data
        profile_data = classifier.classify(cleaned)
        # 2.1 附加原始文本（转介上下文，供后台业务线查看；可选字段）
        profile_data["original_text"] = cleaned
        # 2.2 [可选] 测试用：强制风险等级，便于 UI 验证高危路径渲染
        if force_risk_level:
            profile_data["risk_level"] = force_risk_level
        # 3. 向量化（基于 embedding_prompt 编码）
        user_vector = self.embedder.embed(profile_data["embedding_prompt"])
        # 4. 推荐（高危不拦截，后台转介）
        result = self.recommender.recommend(user_vector, profile_data)

        return {
            "cleaned": cleaned,
            "profile": profile_data,
            "status": result["status"],
            "risk_level": result["risk_level"],
            "recommendations": result["recommendations"],
            "referral": result.get("referral"),
            "crisis_resources": result.get("crisis_resources", []),
        }


def run_pipeline(raw_text: str,
                 config_path: str = "configs/config.yaml",
                 force_risk_level: str | None = None) -> dict:
    """UI 唯一入口。force_risk_level 同 Pipeline.run，仅供 UI 测试。"""
    return Pipeline(config_path).run(raw_text, force_risk_level=force_risk_level)


if __name__ == "__main__":
    p = Pipeline()
    print(f"嵌入后端：{p.embedder.name}")

    # 当前主链路为英文：风险模型与干预池均为英文语料
    samples = [
        "I feel so anxious about my exams, I cannot sleep at night.",
        "I failed two courses and my grades are dropping, I cannot focus on studying.",
        "I want to die, nobody cares about me anymore.",
    ]
    for s in samples:
        print("\n" + "=" * 64)
        print(f"输入：{s}")
        r = p.run(s)
        prof = r["profile"]
        print(f"status={r['status']}  风险等级={r['risk_level']}  归因={prof['target_issue']}")
        for rec in r["recommendations"]:
            print(f"  [id={rec['id']} | sim={rec['similarity_score']:.3f} | "
                  f"final={rec['final_score']:.3f}] ({rec['target_issue']}) {rec['text']}")
        if r["referral"]:
            print(f"  ⚠ 后台转介：{r['referral']['risk_level']} / "
                  f"{r['referral']['target_issue']} → 已写入 risk_referrals.csv")
        for res in r["crisis_resources"]:
            print(f"  [危机资源] {res['text']}")
