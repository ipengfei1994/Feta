# -*- coding: utf-8 -*-
"""
基线对比实验：哈希基线（TF-IDF 臂） vs all-MiniLM-L6-v2（BERT 臂）

支撑项目计划书的「TF-IDF vs BERT 基线对比」要求。

按定稿 Data Contract 模拟真实链路：classifier 吐英文 embedding_prompt →
embedder 编码 → recommender 在干预池（英文，f"{issue} {text}" 编码）中检索。

两组查询对照：
  A. 英文 embedding_prompt（生产链路的真实输入，契约规定）
  B. 中文用户原文（对照组——验证「必须用英文 Prompt」的契约价值）

指标（Top-5）：
  MRR@5  首条 target_issue 命中结果的倒数排名均值
  Hit@5  Top-5 中 target_issue 命中条数比例
  Sim@5  Top-5 平均余弦相似度（后端内部可比）

运行：python scripts/compare_backends.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedder import HashingEmbedder, SentenceTransformerEmbedder

POOL = os.path.join("data", "interventions.csv")
TOP_K = 5

# (中文用户原文, classifier 产出的英文 embedding_prompt（不带 issue 前缀）, 期望归因)
QUERIES = [
    ("照镜子总觉得自己长得不好看，好焦虑",
     "Seeking self-acceptance and body confidence, coping with anxiety about looks",
     "Appearance_Anxiety"),
    ("考试挂了两科，绩点要完了，复习不进去",
     "Coping with exam pressure and study stress, seeking motivation and calm focus",
     "Academic_Stress"),
    ("室友天天闹矛盾，宿舍待不下去了",
     "Resolving friendship and roommate conflict, seeking communication skills and connection",
     "Interpersonal"),
    ("爸妈总吵架，家里的气氛让我窒息",
     "Coping with family tension and parents arguing, seeking emotional support",
     "Family"),
    ("生活费不够用，又不好意思找家里要",
     "Coping with money worries and budgeting stress as a student",
     "Financial_Stress"),
    ("找不到实习，毕业就失业，前途一片黑暗",
     "Overcoming job hunting anxiety and career uncertainty, staying hopeful about the future",
     "Career_Future"),
    ("心里空落落的，做什么都提不起劲",
     "Lifting low mood and emptiness, finding hope and emotional balance",
     "Emotional_Health"),
    ("连续失眠两周，白天根本撑不住",
     "Improving sleep quality, relaxation techniques for insomnia relief",
     "Sleep_Disorder"),
    ("心情低落，想放松一下",
     "Relaxation and mindfulness, gentle comfort and positive energy",
     "General"),
]


def load_pool() -> tuple[list[str], np.ndarray, list[str]]:
    """读取干预池：文本（与构建时同构，不带 issue 前缀）、向量、归因标签。"""
    texts, vecs, issues = [], [], []
    with open(POOL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts.append(r["text"])
            vecs.append(np.fromstring(r["embedding"], sep=" ", dtype="float32"))
            issues.append(r["target_issue"])
    return texts, np.stack(vecs), issues


def evaluate(qvec: np.ndarray, pool_vecs: np.ndarray, issues: list[str],
             want: str, top_k: int = TOP_K) -> dict:
    """单条查询 → MRR@5 / Hit@5 / Sim@5 / Top1。"""
    sims = pool_vecs @ qvec
    order = np.argsort(-sims)[:top_k]
    top_issues = [issues[i] for i in order]

    mrr = 0.0
    for rank, iss in enumerate(top_issues, start=1):
        if iss == want:
            mrr = 1.0 / rank
            break
    hit = sum(1 for iss in top_issues if iss == want) / top_k
    return {"mrr": mrr, "hit": hit, "sim": float(np.mean(sims[order])),
            "top1": issues[order[0]], "top1_sim": float(sims[order[0]])}


def run_group(name: str, queries: list[tuple[str, str]], hasher: HashingEmbedder,
              minilm: SentenceTransformerEmbedder, hash_pool: np.ndarray,
              minilm_pool: np.ndarray, issues: list[str]) -> list[tuple]:
    print(f"\n===== 对照组 {name} =====")
    rows = []
    for text, want in queries:
        rh = evaluate(hasher.embed(text), hash_pool, issues, want)
        rm = evaluate(minilm.embed(text).astype("float32"), minilm_pool, issues, want)
        rows.append((text, want, rh, rm))
        print(f"{text:<16s} → {want:<19s} | 哈希 {rh['mrr']:.2f}/{rh['hit']:.0%} | "
              f"MiniLM {rm['mrr']:.2f}/{rm['hit']:.0%} | Top1[{rm['top1']}] {rm['top1_sim']:.3f}")
    return rows


def main() -> None:
    texts, minilm_pool, issues = load_pool()
    print(f"干预池 {len(issues)} 条 | 测试查询 {len(QUERIES)} 条 | Top-{TOP_K}")
    print("提示：MRR/Hit 为「哈希基线」/「MiniLM」两后端数值")

    print("\n[1/2] 哈希基线编码 3000 条池子 …")
    hasher = HashingEmbedder()
    hash_pool = hasher.embed_batch(texts)
    print("[2/2] MiniLM 编码查询 …")
    minilm = SentenceTransformerEmbedder()

    # A：英文 embedding_prompt（生产链路真实输入）
    rows_a = run_group("A：英文 embedding_prompt（契约路径）",
                       [(p, w) for _, p, w in QUERIES], hasher, minilm,
                       hash_pool, minilm_pool, issues)
    # B：中文用户原文（对照组）
    rows_b = run_group("B：中文用户原文（对照）",
                       [(t, w) for t, _, w in QUERIES], hasher, minilm,
                       hash_pool, minilm_pool, issues)

    def line(label, key, fmt):
        agg_h = lambda rows: float(np.mean([r[2][key] for r in rows]))
        agg_m = lambda rows: float(np.mean([r[3][key] for r in rows]))
        print(f"{label:<8s} | {agg_h(rows_a):{fmt}} | {agg_m(rows_a):{fmt}} "
              f"| {agg_h(rows_b):{fmt}} | {agg_m(rows_b):{fmt}}")

    print("\n===== 汇总（9 条查询平均）=====")
    print(f"{'':8s} | {'A·哈希':>10s} | {'A·MiniLM':>10s} | {'B·哈希':>10s} | {'B·MiniLM':>10s}")
    print("-" * 62)
    line("MRR@5", "mrr", "10.3f")
    line("Hit@5", "hit", "10.1%")
    line("Sim@5", "sim", "10.3f")


if __name__ == "__main__":
    main()
