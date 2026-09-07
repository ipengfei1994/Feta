# -*- coding: utf-8 -*-
"""
风险等级模型推理层（英文）。

封装 NLP 同学训练的风险四分类模型（TF-IDF + LogisticRegression），
输出 Normal / Anxiety / Depression / Suicidal 四档之一。

模型文件位于 models/risk_classifier/（不入库，见 .gitignore）：
    risk_tfidf_vec.joblib     TF-IDF 向量化器（10 万词表，纯英文）
    risk_logreg.joblib        LogReg 分类器
    stressor_tfidf_vec.joblib 压力源向量化器（备用，当前主流程未接入）
    stressor_logreg.joblib    压力源分类器（备用，当前主流程未接入）

设计要点：
  - 延迟加载：首次调用时才 joblib.load，lru_cache 保证只加载一次
  - 优雅降级：模型文件缺失或依赖未安装时返回 None，
    classifier 自动回退到关键词规则基线，链路不中断
  - 纯英文：模型在英文语料上训练，中文输入会全部落为零特征并输出恒定值，
    因此调用方需保证输入为英文文本
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "risk_classifier")
)

RISK_VEC_PATH = os.path.join(_MODEL_DIR, "risk_tfidf_vec.joblib")
RISK_CLF_PATH = os.path.join(_MODEL_DIR, "risk_logreg.joblib")
STRESSOR_VEC_PATH = os.path.join(_MODEL_DIR, "stressor_tfidf_vec.joblib")
STRESSOR_CLF_PATH = os.path.join(_MODEL_DIR, "stressor_logreg.joblib")

# 模型输出的四档风险等级（与 classifier 契约一致）
RISK_LEVELS = ("Anxiety", "Depression", "Normal", "Suicidal")


@functools.lru_cache(maxsize=1)
def _load_risk_bundle():
    """加载风险模型（向量化器 + 分类器）。失败返回 None。"""
    try:
        import joblib
    except ImportError:
        return None
    if not (os.path.exists(RISK_VEC_PATH) and os.path.exists(RISK_CLF_PATH)):
        return None
    try:
        return joblib.load(RISK_VEC_PATH), joblib.load(RISK_CLF_PATH)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _load_stressor_bundle():
    """加载压力源模型（备用）。失败返回 None。"""
    try:
        import joblib
    except ImportError:
        return None
    if not (os.path.exists(STRESSOR_VEC_PATH) and os.path.exists(STRESSOR_CLF_PATH)):
        return None
    try:
        return joblib.load(STRESSOR_VEC_PATH), joblib.load(STRESSOR_CLF_PATH)
    except Exception:
        return None


def available() -> bool:
    """风险模型是否可用（文件存在且依赖已装）。"""
    return _load_risk_bundle() is not None


def predict_risk(text: str) -> Optional[Tuple[str, float]]:
    """预测风险等级。返回 (risk_level, confidence)，不可用返回 None。

    confidence 为预测类别的概率值（0~1），可用于阈值过滤或 UI 展示置信度。
    """
    bundle = _load_risk_bundle()
    if bundle is None:
        return None
    vec, clf = bundle
    try:
        x = vec.transform([text or ""])
        label = str(clf.predict(x)[0])
        proba = clf.predict_proba(x)[0]
        classes = [str(c) for c in clf.classes_]
        conf = float(proba[classes.index(label)]) if label in classes else 0.0
        return label, round(conf, 4)
    except Exception:
        return None


def predict_stressor(text: str) -> Optional[Tuple[str, float]]:
    """预测压力源（她们的 8 类枚举，当前主流程未接入，保留备用）。"""
    bundle = _load_stressor_bundle()
    if bundle is None:
        return None
    vec, clf = bundle
    try:
        x = vec.transform([text or ""])
        label = str(clf.predict(x)[0])
        proba = clf.predict_proba(x)[0]
        classes = [str(c) for c in clf.classes_]
        conf = float(proba[classes.index(label)]) if label in classes else 0.0
        return label, round(conf, 4)
    except Exception:
        return None
