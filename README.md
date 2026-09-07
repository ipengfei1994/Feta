# feta — 基于社交媒体文本的学生心理预警与干预推荐系统

面向学生群体的心理风险预警与正向疏导内容推荐系统：从社交媒体文本出发，识别心理风险等级与压力源归因，召回并精排正向疏导内容（干预池）。任何风险等级都照常获得推荐（不剥夺使用权）；高危用户额外在后台生成转介记录，供人工/社工等业务线介入。

## 技术栈

- Python 3.13 + [uv](https://docs.astral.sh/uv/) 项目与依赖管理
- `sentence-transformers`（`all-MiniLM-L6-v2`，384 维）语义向量
- `scikit-learn` / `joblib` 风险四分类模型推理（TF-IDF + LogReg，英文）
- `numpy` / `pyyaml` / `streamlit`（界面，规划中）

## 快速开始

```bash
# 1. 安装环境（按 uv.lock 精确还原全部依赖，含 torch）
uv sync

# 2. 下载本地向量模型（从 ModelScope，约 90MB，零依赖脚本，绕开被代理拦截的 HuggingFace）
uv run python scripts/download_model.py

# 3. 构建干预池（从 Sentiment140 正向推文清洗→筛选→打标→向量预计算）
uv run python scripts/build_intervention_pool.py --backend auto

# 4. 跑端到端流水线（规则基线演示）
uv run python src/pipeline.py

# 5. 跑推荐模块独立测试
uv run python src/recommender.py
```

> 无需 uv 的队友可生成传统依赖清单：`uv export --format requirements-txt > requirements.txt`

## 目录结构

```
feta/
├── app/                  # Streamlit 界面
├── src/                  # 核心流水线
│   ├── preprocess.py     # 文本清洗
│   ├── classifier.py     # 风险等级 + 压力源归因 + embedding_prompt
│   ├── risk_model.py     # 风险四分类模型推理封装（英文，缺文件时优雅降级）
│   ├── embedder.py       # 双后端向量编码：MiniLM(BERT) / 哈希(TF-IDF)
│   ├── recommender.py    # 召回与精排推荐层
│   ├── referral.py       # 高危后台转介记录，不拦截推荐；推业务线通道预留
│   └── pipeline.py       # 端到端编排
├── data/                 # 词表 / 危机话术库 / interventions.csv 干预池 / risk_referrals.csv 转介记录 / 原始数据集
├── models/               # 本地模型权重（不入库）
│   ├── all-MiniLM-L6-v2/      # 句向量模型，scripts/download_model.py 下载
│   └── risk_classifier/       # 风险四分类模型，见「模型权重获取」
├── configs/config.yaml   # 权重 / 阈值 / 路径
├── scripts/              # 构建与实验脚本
│   ├── download_model.py             # 从 ModelScope 下载 all-MiniLM-L6-v2（零依赖）
│   ├── build_intervention_pool.py    # Sentiment140 → interventions.csv
│   ├── compare_backends.py           # TF-IDF vs BERT 基线对比
│   └── make_demo_pool.py             # 中文演示池
├── pyproject.toml        # uv 项目定义
└── uv.lock               # 依赖精确锁定
```

## 模型权重获取

`models/` 整体不入库（见 `.gitignore`），需自行放置：

| 模型 | 路径 | 获取方式 | 缺失时行为 |
|---|---|---|---|
| 句向量 | `models/all-MiniLM-L6-v2/` | `uv run python scripts/download_model.py` | 自动回退哈希向量后端 |
| 风险四分类 | `models/risk_classifier/` | 由 NLP 同学提供 4 个 joblib 文件 | 自动回退关键词规则基线 |

`models/risk_classifier/` 需包含：`risk_tfidf_vec.joblib`、`risk_logreg.joblib`（主链路），
可选 `stressor_tfidf_vec.joblib`、`stressor_logreg.joblib`（压力源模型，当前主流程未接入）。

> ⚠️ 该模型在**英文**语料上训练，中文输入会全部落为零特征并输出恒定值。
> classifier 的检测逻辑：文本不含拉丁字母时自动走关键词规则分支，不会误用模型。

## 核心接口契约

各模块只认两个结构，谁先写完谁先联调：

**输入**：`recommend(user_vector: np.ndarray, profile_data: dict)`

```python
profile_data = {
    "risk_level": "Normal",                    # Normal | Anxiety | Depression | Suicidal
    "target_issue": "Academic_Stress",         # 8 大压力源之一
    "strategy_weights": {"relaxation": 0.35, "cognitive": 0.25, "healing": 0.25, "lifestyle": 0.15},
    "negative_rules": {"exclude_tags": ["fast_paced", "comparison"], "max_arousal_score": 0.4},
    # 附加诊断字段（下游不依赖，供调试与 UI 展示）
    "risk_source": "model",                    # "model" | "keyword" 风险等级判定来源
    "risk_confidence": 0.93,                   # 模型置信度；keyword 模式下为 1.0
}
```

**输出**：

```python
{
    "status": "SUCCESS",              # 或 "SUCCESS_WITH_REFERRAL"（高危 + 已转介）
    "risk_level": "Normal",
    "recommendations": [{"id", "text", "target_issue", "similarity_score", "final_score"}, ...],
    "referral": None,                 # 高危时：{timestamp, risk_level, target_issue, reason, logged}
    "crisis_resources": [],           # 高危时附带分级危机话术卡，不作为唯一输出
}
```

高危时 `crisis_resources` 为单元素列表，结构见 `data/crisis_playbook.yaml`：

```python
{
    "id": "crisis-suicidal", "text": "文本呈现明确的自杀意念…",   # text = summary，兼容只渲染文本的 UI
    "target_issue": "Crisis_Intervention",
    "level": "高风险(紧急)", "urgency": 4, "summary": "…",
    "actions": [...],          # 分级干预动作清单
    "monitor": "…",            # 后续监测建议
    "stressor_advice": "…",    # 按 target_issue 给的针对性建议
    "resources": [{"name": "…", "phone": "…"}, ...],   # urgency >= 4 时下发
}
```

## 关键结论（实验支撑）

- **必须走英文 `embedding_prompt`**：中文原文直接编码查英文池会失效（Hit@5 从 97.8% 掉到 13%）。classifier 需输出英文语义描述，详见 `scripts/compare_backends.py`。
- **相似度分数只在同后端内可比**：哈希基线相似度虚高不可靠，MiniLM 低而准。
- **风险模型指标（英文测试集，4845 条）**：accuracy 0.799、macro-F1 0.787；Suicidal recall 0.699。
  因此 classifier 保留自杀关键词安全网——模型漏判时规则兜底升级为 `Suicidal`，宁可假阳不漏检。
- **当前主链路为英文**：风险模型、干预池、embedding_prompt 均为英文语料，验收以英文为准。

## 当前阶段边界

- **英文数据集闭环即可**：整条流水线（classifier 英文 Prompt → 英文干预池 → 语义检索）以英文为验收标准；中文干预池（人工策展中文疏导内容）列为后续阶段，代码复用无需改动。
- **转介只后台落盘**：高危用户的后台转介当前只写入 `data/risk_referrals.csv`，「推给业务线」的自动推送通道（腾讯文档/企业微信/内部 API）为预留项，后续接通道只改 `src/referral.py`。
