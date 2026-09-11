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
from src.embedder import Embedder, HashingEmbedder

# (id, text, target_issue, boost_score, tags, arousal_score, strategy)
# id=3 / id=6 carry comparison/fast_paced tags or high arousal — used to verify
# that negative_rules filtering correctly excludes them from normal recommendations.
DEMO_ITEMS = [
    (1, "Break big goals into 25-minute chunks and use a pomodoro timer to take the first step.", "Academic_Stress", 0.90, "planning,cognitive", 0.2, "cognitive"),
    (2, "Pulling an all-nighter before the exam is never worth it — seven hours of sleep is the real foundation of focus.", "Academic_Stress", 0.80, "sleep,lifestyle", 0.1, "lifestyle"),
    (3, "Everyone else studies twelve hours a day — if you don't grind harder, you're already behind.", "Academic_Stress", 0.10, "comparison,fast_paced", 0.7, "cognitive"),
    (4, "Spend ten minutes each day writing down three small things that brought you peace.", "Appearance_Anxiety", 0.90, "mindfulness,relaxation", 0.2, "relaxation"),
    (5, "Try looking in the mirror and seeing a friend instead of a critic.", "Appearance_Anxiety", 0.85, "self_compassion,cognitive", 0.2, "cognitive"),
    (6, "You'll only be liked if you lose weight — anything else is a failure.", "Appearance_Anxiety", 0.10, "comparison,toxic", 0.6, "cognitive"),
    (7, "Talk to a friend about what's been weighing on you — being heard is healing in itself.", "Interpersonal", 0.90, "social,healing", 0.2, "healing"),
    (8, "Take five deep breaths before you respond in a conflict — express what you need instead of blaming.", "Interpersonal", 0.85, "communication,cognitive", 0.3, "cognitive"),
    (9, "Thirty minutes of movement releases endorphins that are nature's mood boosters.", "Emotional_Health", 0.90, "exercise,lifestyle", 0.4, "lifestyle"),
    (10, "If you've felt low for two weeks straight, consider booking a session at your school's counseling center.", "Emotional_Health", 0.95, "professional,healing", 0.2, "healing"),
    (11, "Give yourself permission to rest — not every minute has to be productive.", "Emotional_Health", 0.85, "self_care,relaxation", 0.1, "relaxation"),
    (12, "Put your phone away an hour before bed and soak your feet in warm water to help you drift off.", "Sleep_Disorder", 0.80, "sleep,relaxation", 0.1, "relaxation"),
    (13, "Write down your job-search worries and break them into three small things you can do tomorrow.", "Career_Future", 0.80, "planning,cognitive", 0.3, "cognitive"),
    (14, "Be honest with your family about how you feel — it's easier than keeping it bottled up.", "Family", 0.80, "communication,healing", 0.3, "healing"),
]


def main():
    # 与 configs/config.yaml pool_builder.embedding_backend 对齐：auto 优先本地 MiniLM，
    # 保证池向量与 pipeline 用户向量处于同一空间（否则余弦相似度失真）；失败回退哈希基线
    try:
        embedder = Embedder(backend="auto")
    except Exception:
        embedder = HashingEmbedder()

    # 向量编码：只用正文，不带 issue 前缀（与 classifier.embedding_prompt 保持一致）
    texts = [d[1] for d in DEMO_ITEMS]
    embeddings = embedder.embed_batch(texts)

    os.makedirs("data", exist_ok=True)
    with open("data/interventions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "target_issue", "boost_score",
                    "tags", "arousal_score", "strategy", "embedding"])
        for (idx, text, issue, boost, tags, arousal, strategy), vec in zip(DEMO_ITEMS, embeddings):
            w.writerow([idx, text, issue, boost, tags, arousal, strategy,
                        " ".join(f"{x:.6f}" for x in vec)])

    print(f"Generated {len(DEMO_ITEMS)} demo intervention items (embed backend: {embedder.__class__.__name__})")
    print("  -> data/interventions.csv")


if __name__ == "__main__":
    main()
