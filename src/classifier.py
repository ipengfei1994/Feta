# -*- coding: utf-8 -*-
"""
3.1 NLP 心理感知与画像层（接口锁定 + 规则基线）

负责人：寇丽雯 / 李鑫（正式用训练/微调的轻量模型替换规则逻辑）

接口：classify(cleaned_text: str) -> dict   （即 3.3 所需的 profile_data）

返回字段（契约，recommender 只依赖这几个 key）：
    risk_level      : str   # Level_1_Low ~ Level_4_High
    target_issue    : str   # 8 大类压力源之一
    strategy_weights: dict  # relaxation / cognitive / healing / lifestyle
    negative_rules  : dict  # exclude_tags + max_arousal_score
    embedding_prompt: str   # 供 3.2 embedder 编码成 384 维 User Vector
"""

# 8 大类压力源归因（需与寇丽雯/李鑫最终对齐词表）
TARGET_ISSUES = {
    "Academic_Stress":     ["考试", "成绩", "挂科", "作业", "论文", "绩点", "复习", "保研", "考研", "学业"],
    "Appearance_Anxiety":  ["容貌", "长相", "外貌", "颜值", "身材", "太胖", "太瘦", "整容", "照镜子", "不好看", "丑"],
    "Interpersonal":       ["朋友", "室友", "孤立", "吵架", "排挤", "社交", "分手", "冷战", "合不来"],
    "Family":              ["父母", "家人", "家庭", "催婚", "重男轻女", "吵架"],
    "Financial_Stress":    ["没钱", "穷", "兼职", "学费", "生活费", "经济"],
    "Career_Future":       ["找工作", "就业", "实习", "未来", "迷茫", "毕业", "offer"],
    "Emotional_Health":    ["难过", "崩溃", "低落", "孤独", "emo", "抑郁", "焦虑", "压力大", "累"],
    "Sleep_Disorder":      ["失眠", "睡不着", "睡眠", "熬夜", "多梦"],
}

# 自杀倾向关键词（最高危，单独判断）
SUICIDE_KEYWORDS = ["自杀", "不想活", "轻生", "结束生命", "活不下去", "解脱", "跳楼", "割腕"]

# 干预策略 4 类（默认权重，可后续按画像微调）
DEFAULT_STRATEGY_WEIGHTS = {
    "relaxation": 0.35,
    "cognitive": 0.25,
    "healing": 0.25,
    "lifestyle": 0.15,
}

# 负向黑名单规则（默认值）
DEFAULT_NEGATIVE_RULES = {
    "exclude_tags": ["fast_paced", "comparison"],
    "max_arousal_score": 0.4,
}

# 各归因的英文语义模板（embedding_prompt 用）
# ⚠️ 实验结论（设计文档 1.5.2）：干预池是英文语料，embedding_prompt 必须是英文语义描述；
#    拿中文原文直接编码查英文池，Hit@5 会从 97.8% 掉到 13%。
# 规则基线只知道归因类别，所以按类别输出模板；正式版由模型生成贴合具体内容的英文 Prompt。
ENGLISH_ISSUE_PROMPTS = {
    "Academic_Stress":    "Coping with exam pressure and study stress, seeking motivation and calm focus",
    "Appearance_Anxiety": "Seeking self-acceptance and body confidence, coping with anxiety about looks",
    "Interpersonal":      "Resolving friendship and roommate conflict, seeking communication skills and connection",
    "Family":             "Coping with family tension and parents arguing, seeking emotional support",
    "Financial_Stress":   "Coping with money worries and budgeting stress as a student",
    "Career_Future":      "Overcoming job hunting anxiety and career uncertainty, staying hopeful about the future",
    "Emotional_Health":   "Lifting low mood and emptiness, finding hope and emotional balance",
    "Sleep_Disorder":     "Improving sleep quality, relaxation techniques for insomnia relief",
    "General":            "Relaxation and mindfulness, gentle comfort and positive energy",
}


def classify(cleaned_text: str) -> dict:
    """规则基线：关键词命中 → 风险等级 + 压力源归因 + 权重。正式由模型替换。"""
    text = cleaned_text or ""

    # 最高危：自杀倾向
    if any(k in text for k in SUICIDE_KEYWORDS):
        risk_level, issue = "Level_4_High", "Emotional_Health"
    else:
        # 命中压力源，取命中词最多的那类
        hit_counts = {
            issue: sum(1 for k in kws if k in text)
            for issue, kws in TARGET_ISSUES.items()
        }
        hit_counts = {k: v for k, v in hit_counts.items() if v > 0}
        if hit_counts:
            issue = max(hit_counts, key=hit_counts.get)
            # 情绪健康 / 睡眠问题归为较高风险，其余为中等
            if issue in ("Emotional_Health", "Sleep_Disorder"):
                risk_level = "Level_3_High"
            else:
                risk_level = "Level_2_Moderate"
        else:
            issue, risk_level = "Emotional_Health", "Level_1_Low"

    # embedding_prompt：英文语义描述（契约路径）
    # 归因标签打头（与干预池 f"{issue} {text}" 编码同构）+ 该类的英文疏导语义模板
    embedding_prompt = f"{issue} {ENGLISH_ISSUE_PROMPTS[issue]}"

    return {
        "risk_level": risk_level,
        "target_issue": issue,
        "strategy_weights": dict(DEFAULT_STRATEGY_WEIGHTS),
        "negative_rules": dict(DEFAULT_NEGATIVE_RULES),
        "embedding_prompt": embedding_prompt,
    }
