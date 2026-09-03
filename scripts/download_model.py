# -*- coding: utf-8 -*-
"""
下载本地向量模型 all-MiniLM-L6-v2（ModelScope 源，绕开被代理拦截的 HuggingFace）

纯标准库实现（urllib），零第三方依赖，无需安装 modelscope。

用法：
    uv run python scripts/download_model.py

下载到：models/all-MiniLM-L6-v2/
    （config.json / model.safetensors / vocab.txt / tokenizer.json 等 10 个文件）
embedder.py 会自动优先加载该本地目录，存在即免联网。
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo"
REVISION = "master"

# 与本地已验证可加载的目录结构一致（不拉 onnx/openvino 等导出目录）
FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]

LOCAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2")
)


def _download(url: str, dest: str) -> None:
    """分块下载单个文件，带进度；先走默认代理，失败则直连重试一次。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "feta-downloader"})

    for use_proxy in (True, False):
        try:
            opener = urllib.request.build_opener()
            if not use_proxy:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 // total
                            print(f"\r  {os.path.basename(dest)}: {pct:3d}% "
                                  f"({done // 1048576}/{total // 1048576} MB)", end="")
                if total:
                    print()
            return
        except Exception as e:
            if use_proxy:
                continue  # 走直连再试一次
            raise RuntimeError(f"下载失败 {url}: {e}") from e


def main() -> None:
    marker = os.path.join(LOCAL_DIR, "model.safetensors")
    if os.path.isfile(marker):
        print(f"本地模型已就绪：{LOCAL_DIR}")
        print("如需重新下载，请先手动删除该目录。")
        return

    print(f"从 ModelScope 下载 {MODEL_ID} 到 {LOCAL_DIR} ...")
    for rel in FILES:
        url = (f"{BASE_URL}?FilePath={urllib.parse.quote(rel, safe='')}"
               f"&Revision={REVISION}")
        dest = os.path.join(LOCAL_DIR, rel.replace("/", os.sep))
        print(f"下载 {rel} ...")
        _download(url, dest)

    if not os.path.isfile(marker):
        print("错误：model.safetensors 未下载成功，请重试。")
        sys.exit(1)
    print(f"完成。本地模型：{LOCAL_DIR}")
    print("现在可运行：uv run python src/pipeline.py")


if __name__ == "__main__":
    main()
