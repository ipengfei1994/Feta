# -*- coding: utf-8 -*-
"""
文本清洗模块（接口占位）

负责人：寇丽雯 / 李鑫
接口：clean(text: str) -> str

当前为最小可用实现（去多余空白、统一符号），正式清洗逻辑（emoji、@、URL、
口语噪声等）由负责人在此函数内补全，保持函数签名不变即可。
"""

import re


def clean(text: str) -> str:
    """输入原始社交文本，输出清洗后的文本。

    契约：调用方只依赖此签名，内部实现可自由替换。
    """
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
