# -*- coding: utf-8 -*-
"""
向量 Embedding 层（接口锁定 + 基线实现）

负责人：寇丽雯 / 李鑫（正式用 all-MiniLM-L6-v2 替换 HashingEmbedder）

提供两个后端：
- HashingEmbedder：字符 n-gram 特征哈希（TF-IDF 级基线，零依赖，永远可用）
- SentenceTransformerEmbedder：all-MiniLM-L6-v2（BERT 级，装了 sentence-transformers 自动启用）

对 recommender 而言，只认 `embed() -> np.ndarray(384,)` 这一个接口，
后端怎么换都不影响下游。
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

DIM = 384

# 本地模型目录（从 ModelScope 下载，绕开被代理拦截的 HuggingFace）
_LOCAL_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2")
)
HF_MODEL_NAME = "all-MiniLM-L6-v2"


def _resolve_model_path(model_name: str) -> str:
    """若本地 models/all-MiniLM-L6-v2 目录存在且含权重，优先返回本地路径（免联网）。"""
    if model_name == HF_MODEL_NAME and os.path.isfile(
        os.path.join(_LOCAL_MODEL_DIR, "model.safetensors")
    ):
        return _LOCAL_MODEL_DIR
    return model_name


def _char_grams(text: str) -> list[str]:
    """字符 unigram + bigram，天然适配中文（无需分词）。"""
    chars = [c.lower() for c in text if not c.isspace()]
    grams = chars[:]
    grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return grams


class HashingEmbedder:
    """特征哈希词袋：把字符 n-gram 散列到 384 维，带符号累加后归一化。"""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for g in _char_grams(text):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """BERT 级后端，懒加载（未安装 sentence-transformers 时实例化会报错）。
    默认优先加载本地 models/all-MiniLM-L6-v2（免联网），不存在才回退 HuggingFace。"""

    def __init__(self, model_name: str = HF_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(_resolve_model_path(model_name))

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).astype("float32")

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")


class Embedder:
    """统一入口：优先 BERT，缺失则回退哈希基线。"""

    def __init__(self, backend: str = "auto", dim: int = DIM):
        if backend == "auto":
            try:
                self._impl = SentenceTransformerEmbedder()
                self.name = "sentence-transformers"
            except Exception:
                self._impl = HashingEmbedder(dim)
                self.name = "hashing-baseline"
        elif backend == "hashing":
            self._impl = HashingEmbedder(dim)
            self.name = "hashing-baseline"
        else:
            self._impl = SentenceTransformerEmbedder(backend)
            self.name = backend

    def embed(self, text: str) -> np.ndarray:
        return self._impl.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._impl.embed_batch(texts)


# 兼容旧调用（李鹏飞的 recommender 测试可直接用）
def embed(text: str) -> np.ndarray:
    return HashingEmbedder().embed(text)
