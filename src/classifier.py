# -*- coding: utf-8 -*-
"""
心理感知与画像层。

接口：classify(cleaned_text: str) -> dict
返回字段契约（下游 recommender 只依赖这几个 key）：
    risk_level       : str               # Level_1_Low | Level_2_Moderate | Level_3_High | Level_4_High
    target_issue     : str               # 8 大类压力源之一（与干预池打标对齐，禁止改名）
    strategy_weights : dict              # {relaxation, cognitive, healing, lifestyle}
    negative_rules   : dict              # {exclude_tags: [...], max_arousal_score: float}
    embedding_prompt : str               # 英文语义描述（必须英文，详见 data/issue_lexicon.yaml）

数据与逻辑分离：词表 / 关键词 / 英文 prompt 模板均存于 data/issue_lexicon.yaml，
运行时由 _load_lexicon() 读取并缓存（lru_cache）。改词表无需改本文件。
"""

from __future__ import annotations

import functools
import os
from typing import Dict, List

import yaml

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


@functools.lru_cache(maxsize=1)
def _load_lexicon() -> dict:
    """加载词表 YAML。文件不存在时抛 FileNotFoundError。lru_cache 避免重复 IO。"""
    with open(_LEXICON_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_target_issues() -> Dict[str, List[str]]:
    return _load_lexicon()["target_issues"]


def _get_suicide_keywords() -> List[str]:
    return _load_lexicon()["suicide_keywords"]


def _get_english_issue_prompts() -> Dict[str, str]:
    return _load_lexicon()["english_issue_prompts"]


def classify(cleaned_text: str) -> dict:
    """根据清洗后的文本，输出心理画像 dict。

    判定流程：
        1. 自杀关键词命中 → Level_4_High + Emotional_Health
        2. 8 大类词表命中计数，取最高分类
           - 命中 Emotional_Health / Sleep_Disorder → Level_3_High
           - 命中其他类                                  → Level_2_Moderate
        3. 零命中 → Emotional_Health + Level_1_Low（兜底）

    embedding_prompt 拼接规则：f"{issue} {ENGLISH_ISSUE_PROMPTS[issue]}"
    若 issue 不在 prompts 中，fallback 到 "General" prompt。
    """
    text = cleaned_text or ""
    target_issues = _get_target_issues()
    suicide_kws = _get_suicide_keywords()
    prompts = _get_english_issue_prompts()

    # 最高危：自杀/自伤意图关键词
    if any(k in text for k in suicide_kws):
        risk_level, issue = "Level_4_High", "Emotional_Health"
    else:
        # 各归因命中数统计，取最高
        hit_counts = {
            issue: sum(1 for k in kws if k in text)
            for issue, kws in target_issues.items()
        }
        hit_counts = {k: v for k, v in hit_counts.items() if v > 0}
        if hit_counts:
            issue = max(hit_counts, key=hit_counts.get)
            if issue in ("Emotional_Health", "Sleep_Disorder"):
                risk_level = "Level_3_High"
            else:
                risk_level = "Level_2_Moderate"
        else:
            issue, risk_level = "Emotional_Health", "Level_1_Low"

    # embedding_prompt：英文语义描述（契约路径，详见 data/issue_lexicon.yaml 注释）
    prompt_template = prompts.get(issue, prompts["General"])
    embedding_prompt = f"{issue} {prompt_template}"

    return {
        "risk_level": risk_level,
        "target_issue": issue,
        "strategy_weights": dict(DEFAULT_STRATEGY_WEIGHTS),
        "negative_rules": dict(DEFAULT_NEGATIVE_RULES),
        "embedding_prompt": embedding_prompt,
    }