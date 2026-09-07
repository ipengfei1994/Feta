# -*- coding: utf-8 -*-
"""
心理感知与画像层。

接口：classify(cleaned_text: str) -> dict
返回字段契约（下游 recommender 只依赖这几个 key）：
    risk_level       : str               # Normal | Anxiety | Depression | Suicidal
    target_issue     : str               # 8 大类压力源之一（与干预池打标对齐，禁止改名）
    strategy_weights : dict              # {relaxation, cognitive, healing, lifestyle}
    negative_rules   : dict              # {exclude_tags: [...], max_arousal_score: float}
    embedding_prompt : str               # 英文语义描述（必须英文，详见 data/issue_lexicon.yaml）

附加诊断字段（下游不依赖，供调试与 UI 展示）：
    risk_source      : str               # "model" | "keyword" —— 风险等级的判定来源
    risk_confidence  : float             # 模型判定的置信度；keyword 模式下为 1.0

数据与逻辑分离：词表 / 关键词 / 英文 prompt 模板均存于 data/issue_lexicon.yaml，
运行时由 _load_lexicon() 读取并缓存（lru_cache）。改词表无需改本文件。

两个维度独立判定：
  - risk_level（情绪严重度）：用 NLP 同学的标准四档
      Suicidal > Depression > Anxiety > Normal
  - target_issue（压力来源）：用项目自己的 8 大类
      取命中数最多的归因，零命中兜底 General
"""

from __future__ import annotations

import functools
import os
import re
from typing import Dict, List

import yaml

try:
    from src import risk_model   # 项目内包导入（pipeline 场景）
except ImportError:              # 直接执行 src/classifier.py 时
    import risk_model

# 词表路径（相对本文件所在 src/ 的同级 data/ 目录）
_LEXICON_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "issue_lexicon.yaml")
)

# 默认策略权重 / 负向规则（属于系统行为配置而非"数据资产"，留在代码里；
# 如需覆盖请在 configs/config.yaml 中读取并合并）
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "relaxation": 0.35,
    "cognitive": 0.25,
    "healing": 0.25,
    "lifestyle": 0.15,
}

DEFAULT_NEGATIVE_RULES: Dict[str, object] = {
    "exclude_tags": ["fast_paced", "comparison"],
    "max_arousal_score": 0.4,
}

# 拉丁字母检测：纯中文文本无拉丁字母，英文模型必然输出恒定值，走规则分支
_LATIN_RE = re.compile(r"[A-Za-z]")


@functools.lru_cache(maxsize=1)
def _load_lexicon() -> dict:
    """加载词表 YAML。文件不存在时抛 FileNotFoundError。lru_cache 避免重复 IO。"""
    with open(_LEXICON_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_target_issues() -> Dict[str, List[str]]:
    return _load_lexicon()["target_issues"]


def _get_suicide_keywords() -> List[str]:
    return _load_lexicon()["suicide_keywords"]


def _get_risk_level_keywords() -> Dict[str, List[str]]:
    return _load_lexicon()["risk_level_keywords"]


def _get_english_issue_prompts() -> Dict[str, str]:
    return _load_lexicon()["english_issue_prompts"]


def _hit(text_lower: str, keyword: str) -> bool:
    """关键词命中判定。

    纯 ASCII 关键词按单词边界匹配（避免 fat 命中 fate / father），
    含非 ASCII 字符（中文）按子串匹配——中文无空格分词，边界匹配会全部漏掉。
    """
    if keyword.isascii():
        return re.search(r"\b" + re.escape(keyword) + r"\b", text_lower) is not None
    return keyword in text_lower


def classify(cleaned_text: str) -> dict:
    """根据清洗后的文本，输出心理画像 dict。

    判定流程（两个维度独立判定，互不耦合）：

      risk_level（情绪严重度，NLP 同学标准）：
        1. 自杀关键词命中 → Suicidal（安全网：模型漏检时兜底，宁可假阳）
        2. 文本含拉丁字母且风险模型可用 → 模型输出 + 置信度
        3. 否则走关键词规则：Depression → Anxiety → Normal

      target_issue（压力来源，项目 8 大类）：
        1. 各归因词表命中计数，取最高分类
        2. 零命中 → General（兜底）

    embedding_prompt 拼接规则：f"{issue} {ENGLISH_ISSUE_PROMPTS[issue]}"
    若 issue 不在 prompts 中，fallback 到 "General" prompt。
    """
    text = cleaned_text or ""
    text_lower = text.lower()
    target_issues = _get_target_issues()
    suicide_kws = _get_suicide_keywords()
    risk_kws = _get_risk_level_keywords()
    prompts = _get_english_issue_prompts()

    # ── risk_level 判定（情绪严重度，与 target_issue 独立）──
    risk_source = "keyword"
    risk_confidence = 1.0

    if any(_hit(text_lower, k.lower()) for k in suicide_kws):
        # 安全网：模型 Suicidal recall 约 0.70，规则兜底拦截漏检
        risk_level = "Suicidal"
    elif _LATIN_RE.search(text) and risk_model.available():
        pred = risk_model.predict_risk(text)
        if pred is not None:
            risk_level, risk_confidence = pred
            risk_source = "model"
        else:
            risk_level = _keyword_risk_level(text_lower, risk_kws)
    else:
        risk_level = _keyword_risk_level(text_lower, risk_kws)

    # ── target_issue 判定（压力来源）──
    hit_counts = {
        issue: sum(1 for k in kws if _hit(text_lower, k.lower()))
        for issue, kws in target_issues.items()
    }
    hit_counts = {k: v for k, v in hit_counts.items() if v > 0}
    if hit_counts:
        issue = max(hit_counts, key=hit_counts.get)
    elif risk_level in ("Suicidal", "Depression"):
        # 高危但无明确压力源 → 归到情绪健康，避免兜底到泛化放松内容
        issue = "Emotional_Health"
    else:
        issue = "General"

    # embedding_prompt：英文自然语言描述，直接由模板提供。
    # 不再拼接 issue 枚举名（"Academic_Stress" 等带下划线的 token 会被 BERT 切成
    # subword，污染向量空间），模板本身已是语义完整的句子。
    prompt_template = prompts.get(issue, prompts["General"])
    embedding_prompt = prompt_template

    return {
        "risk_level": risk_level,
        "target_issue": issue,
        "strategy_weights": dict(DEFAULT_STRATEGY_WEIGHTS),
        "negative_rules": dict(DEFAULT_NEGATIVE_RULES),
        "embedding_prompt": embedding_prompt,
        "risk_source": risk_source,
        "risk_confidence": risk_confidence,
    }


def _keyword_risk_level(text_lower: str, risk_kws: Dict[str, List[str]]) -> str:
    """关键词规则基线：模型不可用或纯中文输入时使用。"""
    if any(_hit(text_lower, k.lower()) for k in risk_kws["Depression"]):
        return "Depression"
    if any(_hit(text_lower, k.lower()) for k in risk_kws["Anxiety"]):
        return "Anxiety"
    return "Normal"
