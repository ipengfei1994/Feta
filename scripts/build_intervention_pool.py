# -*- coding: utf-8 -*-
"""
从 Sentiment140 正向推文构建 data/interventions.csv 干预池（正向疏导知识库）。

四步流水线：
    ① 筛选与预清洗   —— 只取 target=4 正向推文；去 @用户名 / URL / HTML 实体 / 重复标点；长度过滤
    ② 正向疏导筛选   —— 加权关键词打分，剔除口语刷屏（Good morning 之类），保留治愈/鼓励/正念句子
    ③ 自动归因打标   —— 英文关键词规则映射到 8 大压力源 + General（默认），同时映射干预策略
    ④ 向量预计算     —— 用 all-MiniLM-L6-v2（缺依赖时回退哈希基线）批量编码 384 维向量

产出字段（对齐推荐层 Data Contract）：
    id, text, target_issue, boost_score, tags, arousal_score, strategy, embedding

用法：
    python scripts/build_intervention_pool.py                 # 全量构建
    python scripts/build_intervention_pool.py --limit 200000  # 只扫前 N 行源数据（快速验证）
    python scripts/build_intervention_pool.py --backend hashing

配置：
    configs/config.yaml 的 pool_builder 段
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html as _html
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedder import Embedder

# ---------------------------------------------------------------------------
# ② 正向疏导关键词（加权：强疗愈词 weight=2，普通正向词 weight=1）
# ---------------------------------------------------------------------------
INTERVENTION_KEYWORDS = {
    # —— 强疗愈 / 鼓励（weight 2）——
    2: ["deep breath", "breathe", "breathing", "keep going", "stay strong",
        "you can do", "you got this", "it gets better", "never give up",
        "dont give up", "proud of", "you matter", "you are enough",
        "be kind to yourself", "be gentle with yourself", "forgive yourself",
        "believe in", "you are not alone", "youre not alone", "reach out",
        "talk to someone", "one day at a time", "you are loved", "youre loved",
        "self care", "self-care", "take care of yourself", "mindful",
        "meditate", "meditation", "grateful", "gratitude", "everything will be okay",
        "everything will be alright", "small steps", "baby steps"],
    # —— 普通正向 / 放松（weight 1）——
    1: ["relax", "calm", "peaceful", "peace", "slow down", "hope", "hopeful",
        "sunshine", "warm", "gentle", "comfort", "rest", "recharge",
        "thankful", "blessed", "its okay", "it is okay", "it's okay",
        "take a deep breath", "breathe in", "inner peace", "let go"],
}

# ---------------------------------------------------------------------------
# ③ 归因打标：8 大压力源英文关键词（保守词表，避免 "beautiful day" 误标容貌）
# ---------------------------------------------------------------------------
TARGET_ISSUE_KEYWORDS = {
    "Academic_Stress": ["exam", "exams", "study", "studying", "homework", "grade",
                        "grades", "school", "college", "university", "class", "test",
                        "tests", "gpa", "thesis", "assignment", "deadline", "final",
                        "finals", "semester", "essay"],
    "Appearance_Anxiety": ["body image", "weight", "my body", "my look", "my looks",
                           "ugly", "fat", "skinny", "mirror", "appearance",
                           "self image", "self-image"],
    "Interpersonal": ["friend", "friends", "friendship", "relationship", "breakup",
                      "lonely", "roommate", "social anxiety", "fight", "argue",
                      "best friend", "boyfriend", "girlfriend"],
    "Family": ["family", "mom", "dad", "mother", "father", "parent", "parents",
               "brother", "sister", "grandma", "grandpa"],
    "Financial_Stress": ["money", "broke", "poor", "rent", "bills", "debt",
                         "afford", "paycheck", "tuition"],
    "Career_Future": ["job", "career", "work", "working", "future", "interview",
                      "internship", "graduate", "graduation", "unemployed", "hired",
                      "resume"],
    "Emotional_Health": ["sad", "depressed", "depression", "anxiety", "anxious",
                         "overwhelmed", "cry", "crying", "mental health", "therapy",
                         "therapist", "panic", "loneliness", "heartbroken"],
    "Sleep_Disorder": ["sleep", "sleeping", "insomnia", "tired", "awake",
                       "restless", "insomniac", "cant sleep", "can't sleep",
                       "nap", "wake", "woke"],
}

# ③ 干预策略映射（供 recommender 的 strategy_weights 轻调制）
STRATEGY_KEYWORDS = {
    "relaxation": ["relax", "breathe", "breath", "calm", "peace", "peaceful",
                   "slow down", "meditate", "meditation", "mindful", "deep breath",
                   "inner peace", "let go"],
    "cognitive": ["keep going", "stay strong", "you can do", "you got this",
                  "it gets better", "never give up", "dont give up", "proud of",
                  "believe in", "you are enough", "hope", "hopeful", "small steps",
                  "one day at a time"],
    "healing": ["you matter", "you are loved", "youre loved", "you are not alone",
                "youre not alone", "reach out", "talk to someone", "grateful",
                "gratitude", "kind to yourself", "gentle", "comfort", "forgive",
                "warm", "thankful"],
    "lifestyle": ["sunshine", "walk", "rest", "recharge", "self care", "self-care",
                  "sleep", "exercise", "fresh air", "stretch"],
}

# 高危内容硬性排除：自杀/自伤。Sentiment140 的标签有噪声（如 "I often think of
# suicide" 被标正向），干预池是「安全」知识库，命中即剔除，宁可多排除。
BLOCKLIST = [
    "suicide", "suicidal", "kill myself", "killing myself", "killed myself",
    "kill me", "self harm", "self-harm", "selfharm", "harm myself", "hurt myself",
    "end my life", "end it all", "want to die", "wanna die", "wish i was dead",
    "no reason to live", "dont want to live", "overdose", "jump off a bridge",
]

# 关键词匹配：单词走「集合交集」、短语走「子串」，都比逐词正则快一个量级，
# 且同样避免 final→finally、fat→fate 的误匹配（token 化后单词精确相等）。
# 匹配前统一 _norm（小写 + 去撇号 + 连字符转空格），提升口语变体命中率。
_WORD_RE_CACHE: dict[str, re.Pattern] = {}
_SINGLE_WORD_WEIGHTS: dict[str, int] = {}
_PHRASE_WEIGHTS: dict[str, int] = {}


def _norm(text: str) -> str:
    """匹配用归一化：小写、去撇号（don't→dont）、连字符转空格（self-care→self care）。"""
    return (text.lower().replace("'", "").replace("\u2019", "")
            .replace("-", " "))


def _init_keywords() -> None:
    """把 INTERVENTION_KEYWORDS 预拆为「单词权重表 + 短语权重表」（只初始化一次）。"""
    if _SINGLE_WORD_WEIGHTS:
        return
    for weight, kws in INTERVENTION_KEYWORDS.items():
        for k in kws:
            kn = _norm(k)
            if " " in kn:
                _PHRASE_WEIGHTS[kn] = weight
            else:
                _SINGLE_WORD_WEIGHTS[kn] = weight


def _hit(text: str, keyword: str) -> bool:
    """keyword 是否命中 text（小写）。含空格的短语走子串，单词走词边界。"""
    if " " in keyword:
        return keyword in text
    pat = _WORD_RE_CACHE.get(keyword)
    if pat is None:
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])")
        _WORD_RE_CACHE[keyword] = pat
    return pat.search(text) is not None


def clean(text: str) -> str:
    """① 清洗：去 @、URL、#（保留词）、HTML 实体、重复标点、多余空白。"""
    t = text
    t = re.sub(r"https?://\S+", " ", t)   # URL
    t = re.sub(r"www\.\S+", " ", t)       # www 链接
    t = re.sub(r"@\w+", " ", t)           # @用户名
    t = re.sub(r"#(\w+)", r"\1", t)       # 话题标签保留词
    t = _html.unescape(t)                 # &amp; &lt; 等
    t = re.sub(r"&\w+;", " ", t)          # 残留实体
    t = re.sub(r"([!?.,;:])\1+", r"\1", t)  # 重复标点压成单个
    t = re.sub(r"\s+", " ", t).strip()
    return t


def intervention_score(text: str) -> int:
    """② 加权打分：单词集合交集 + 短语子串（比逐词正则快一个量级）。"""
    t = _norm(text)
    total = 0
    for w in set(re.findall(r"[a-z0-9]+", t)):
        total += _SINGLE_WORD_WEIGHTS.get(w, 0)
    for p, wt in _PHRASE_WEIGHTS.items():
        if p in t:
            total += wt
    return total


def label_issue(text: str) -> str:
    """③ 归因打标：命中的压力源类别取命中词数最多者；无命中或多类并列 → General。"""
    tl = _norm(text)
    hits = {
        issue: sum(1 for k in kws if _hit(tl, _norm(k)))
        for issue, kws in TARGET_ISSUE_KEYWORDS.items()
    }
    hits = {k: v for k, v in hits.items() if v > 0}
    if not hits:
        return "General"
    best = max(hits.values())
    winners = [k for k, v in hits.items() if v == best]
    return winners[0] if len(winners) == 1 else "General"


def label_strategy(text: str) -> str:
    """③ 干预策略映射：命中最多者；无命中默认 healing。"""
    tl = _norm(text)
    hits = {
        s: sum(1 for k in kws if _hit(tl, _norm(k)))
        for s, kws in STRATEGY_KEYWORDS.items()
    }
    hits = {k: v for k, v in hits.items() if v > 0}
    if not hits:
        return "healing"
    return max(hits, key=hits.get)


def boost_from(score: int) -> float:
    """正能量 boost：0.8 起步，随关键词命中数递增，封顶 1.2。"""
    return round(min(0.8 + 0.1 * score, 1.2), 3)


def build_pool(cfg: dict, backend: str, limit: int | None) -> list[dict]:
    """扫描源数据，清洗 + 打分，去重后取 Top-N。"""
    pb = cfg.get("pool_builder", {})
    source = pb.get("source", "data/training.1600000.processed.noemoticon.csv/"
                             "training.1600000.processed.noemoticon.csv")
    min_words = int(pb.get("min_words", 5))
    max_words = int(pb.get("max_words", 30))
    min_score = int(pb.get("min_score", 1))
    max_pool = int(pb.get("max_pool_size", 3000))

    heap: list[tuple[int, int, str]] = []  # (score, 序号, 清洗后文本)
    seen: set[str] = set()
    seq = 0

    _init_keywords()  # 预拆关键词权重表，扫描热路径零初始化开销

    scanned = kept = dropped_len = dropped_score = dropped_block = 0
    with open(source, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            if row[0] != "4":  # 只取正向
                continue
            scanned += 1
            if limit is not None and scanned > limit:
                break
            t = clean(row[5])
            tn = _norm(t)
            if any(k in tn for k in BLOCKLIST):  # 高危内容硬性排除
                dropped_block += 1
                continue
            wc = len(t.split())
            if wc < min_words or wc > max_words:  # 长度过滤
                dropped_len += 1
                continue
            s = intervention_score(t)
            if s < min_score:                      # ② 正向疏导筛选
                dropped_score += 1
                continue
            if t in seen:                          # 去重
                continue
            seen.add(t)
            seq += 1
            if len(heap) < max_pool:
                heapq.heappush(heap, (s, seq, t))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, seq, t))
            kept += 1

    # 按 score 降序（同分按进入顺序）
    items = sorted(heap, key=lambda x: (-x[0], x[1]))
    print(f"[扫描] 正向样本 {scanned} | 高危剔除 {dropped_block} | 长度剔除 {dropped_len} "
          f"| 疏导分不足剔除 {dropped_score}")
    print(f"[筛选] 保留 {kept} 条（去重后） → 取 Top {len(items)}")
    return [
        {
            "text": t,
            "target_issue": label_issue(t),
            "boost_score": boost_from(s),
            "strategy": label_strategy(t),
            "score": s,
        }
        for s, _, t in items
    ]


def main():
    ap = argparse.ArgumentParser(description="从 Sentiment140 构建干预池")
    ap.add_argument("--limit", type=int, default=None,
                    help="只扫描源数据前 N 行（快速验证用）")
    ap.add_argument("--backend", type=str, default="auto",
                    help="embedding 后端：auto | hashing | all-MiniLM-L6-v2")
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pb = cfg.get("pool_builder", {})
    output = pb.get("output", "data/interventions.csv")

    # ②③ 清洗 + 筛选 + 打标
    records = build_pool(cfg, args.backend, args.limit)

    # ④ 向量预计算（离线批量，向量空间与用户向量同构：issue 标签并入正文）
    embedder = Embedder(backend=args.backend)
    texts = [f"{r['target_issue']} {r['text']}" for r in records]
    print(f"[嵌入] 后端={embedder.name}，批量编码 {len(texts)} 条 …")
    vecs = embedder.embed_batch(texts)

    # 写入 interventions.csv
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "target_issue", "boost_score",
                    "tags", "arousal_score", "strategy", "embedding"])
        for i, (r, vec) in enumerate(zip(records, vecs), start=1):
            w.writerow([
                i, r["text"], r["target_issue"], r["boost_score"],
                "", 0.2, r["strategy"],
                " ".join(f"{x:.6f}" for x in vec),
            ])

    # 归因分布
    from collections import Counter
    dist = Counter(r["target_issue"] for r in records)
    print(f"[产出] {len(records)} 条 → {output}")
    print(f"[归因分布] " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))


if __name__ == "__main__":
    main()
