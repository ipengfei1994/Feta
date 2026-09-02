# -*- coding: utf-8 -*-
"""
生成演示用干预池 data/interventions.csv，供 recommender 独立测试。

字段（对齐 3.3 层的 Data Contract）：
    id, text, target_issue, boost_score, tags, arousal_score, strategy, embedding
其中 embedding 为 384 维向量（空格分隔的浮点串）。

用法：
    python scripts/make_demo_pool.py
输出：
    data/interventions.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedder import HashingEmbedder

# (id, 正文, 压力源归因, 正能量boost, 内容标签, 唤醒度, 干预策略)
# 注意：id=3 / id=6 是「负向样本」，带 comparison/fast_paced 或高唤醒度，
#       用于验证 negative_rules 黑名单过滤（正常推荐会被剔除）。
DEMO_ITEMS = [
    (1, "把大目标拆成每天 25 分钟的小块，用番茄钟迈出第一步。", "Academic_Stress", 0.90, "planning,cognitive", 0.2, "cognitive"),
    (2, "熬夜刷题不如睡够 7 小时，睡眠才是效率的地基。", "Academic_Stress", 0.80, "sleep,lifestyle", 0.1, "lifestyle"),
    (3, "别人一天学 12 个小时，你再不卷就彻底完了。", "Academic_Stress", 0.10, "comparison,fast_paced", 0.7, "cognitive"),
    (4, "每天花 10 分钟记录三件让你感到平静的小事。", "Appearance_Anxiety", 0.90, "mindfulness,relaxation", 0.2, "relaxation"),
    (5, "试着把镜子里的自己当作朋友，而不是评审官。", "Appearance_Anxiety", 0.85, "self_compassion,cognitive", 0.2, "cognitive"),
    (6, "瘦下来才会被人喜欢，胖就是失败。", "Appearance_Anxiety", 0.10, "comparison,toxic", 0.6, "cognitive"),
    (7, "和朋友聊一聊最近的不开心，被倾听本身就是疗愈。", "Interpersonal", 0.90, "social,healing", 0.2, "healing"),
    (8, "冲突时先深呼吸 5 次，再表达需求而不是指责。", "Interpersonal", 0.85, "communication,cognitive", 0.3, "cognitive"),
    (9, "运动 30 分钟释放的内啡肽，是天然的情绪调节剂。", "Emotional_Health", 0.90, "exercise,lifestyle", 0.4, "lifestyle"),
    (10, "当你连续两周情绪低落，请考虑预约学校心理中心的咨询。", "Emotional_Health", 0.95, "professional,healing", 0.2, "healing"),
    (11, "允许自己休息，不是每一分钟都要有意义。", "Emotional_Health", 0.85, "self_care,relaxation", 0.1, "relaxation"),
    (12, "睡前一小时放下手机，用温水泡脚帮助入眠。", "Sleep_Disorder", 0.80, "sleep,relaxation", 0.1, "relaxation"),
    (13, "把找工作的焦虑写下来，拆成明天能做的三件小事。", "Career_Future", 0.80, "planning,cognitive", 0.3, "cognitive"),
    (14, "和家人坦诚一次感受，比憋在心里更容易被理解。", "Family", 0.80, "communication,healing", 0.3, "healing"),
]


def main():
    embedder = HashingEmbedder()

    # 用「压力源英文标签 + 正文」做向量，保证归因标签进入向量空间（基线做法，
    # 与 classifier 的 embedding_prompt 同构，才能检索出同 topic 的干预项）
    texts = [f"{d[2]} {d[1]}" for d in DEMO_ITEMS]
    embeddings = embedder.embed_batch(texts)

    os.makedirs("data", exist_ok=True)
    with open("data/interventions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "target_issue", "boost_score",
                    "tags", "arousal_score", "strategy", "embedding"])
        for (idx, text, issue, boost, tags, arousal, strategy), vec in zip(DEMO_ITEMS, embeddings):
            w.writerow([idx, text, issue, boost, tags, arousal, strategy,
                        " ".join(f"{x:.6f}" for x in vec)])

    print(f"已生成 {len(DEMO_ITEMS)} 条演示干预项（嵌入后端：{embedder.__class__.__name__}）")
    print("  -> data/interventions.csv")


if __name__ == "__main__":
    main()
