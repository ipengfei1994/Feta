# -*- coding: utf-8 -*-
"""
文本清洗模块。

接口：clean(text: str) -> str
输入：原始社交文本（可能含 URL、@提及、emoji、话题标签、口语噪声等）
输出：清洗后文本

契约：调用方只依赖此签名，内部实现可自由替换。
"""

import re


def clean(text: str) -> str:
    """输入原始社交文本，输出清洗后的文本。"""
    if not text:
        return ""
    text = text.strip()
    # 去掉 URL
    text = re.sub(r"https?://\S+", "", text)
    # 去掉 @提及
    text = re.sub(r"@\w+", "", text)
    # 合并连续空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()