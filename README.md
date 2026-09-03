# feta — 基于社交媒体文本的学生心理预警与干预推荐系统

面向学生群体的心理风险预警与正向疏导内容推荐系统：从社交媒体文本出发，识别心理风险等级与压力源归因，召回并精排正向疏导内容（干预池）。任何风险等级都照常获得推荐（不剥夺使用权）；高危用户额外在后台生成转介记录，供人工/社工等业务线介入。

## 技术栈

- Python 3.13 + [uv](https://docs.astral.sh/uv/) 项目与依赖管理
- `sentence-transformers`（`all-MiniLM-L6-v2`，384 维）语义向量
- `numpy` / `pyyaml` / `streamlit`（界面，规划中）

## 快速开始

```bash
# 1. 安装环境（按 uv.lock 精确还原全部依赖，含 torch）
uv sync

# 2. 准备本地向量模型（免联网，从 ModelScope 下载，约 90MB）
#    目录：models/all-MiniLM-L6-v2/
#    需包含：config.json / model.safetensors / vocab.txt / tokenizer.json 等
#    下载地址：https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2

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
├── app/                  # Streamlit 界面（王定祥）
├── src/                  # 核心流水线
│   ├── preprocess.py     # 文本清洗（寇/李）
│   ├── classifier.py     # 风险等级 + 压力源归因 + embedding_prompt（寇/李）
│   ├── embedder.py       # 双后端向量编码：MiniLM(BERT) / 哈希(TF-IDF)（寇/李）
│   ├── recommender.py    # 召回与精排推荐层（李鹏飞）
│   ├── referral.py       # 高危后台转介记录，不拦截推荐；推业务线通道预留（李鹏飞）
│   └── pipeline.py       # 端到端编排（李鹏飞）
├── data/                 # interventions.csv 干预池 + risk_referrals.csv 转介记录 + 原始数据集
├── models/               # all-MiniLM-L6-v2 本地权重（不入库）
├── configs/config.yaml   # 权重 / 阈值 / 路径
├── scripts/              # 构建与实验脚本
│   ├── build_intervention_pool.py   # Sentiment140 → interventions.csv
│   ├── compare_backends.py          # TF-IDF vs BERT 基线对比
│   └── make_demo_pool.py            # 中文演示池
├── pyproject.toml        # uv 项目定义
└── uv.lock               # 依赖精确锁定
```

## 核心接口契约

各模块只认两个结构，谁先写完谁先联调：

**输入**：`recommend(user_vector: np.ndarray, profile_data: dict)`

```python
profile_data = {
    "risk_level": "Level_2_Moderate",     # Level_1_Low ~ Level_4_High
    "target_issue": "Academic_Stress",    # 8 大压力源之一
    "strategy_weights": {"relaxation": 0.35, "cognitive": 0.25, "healing": 0.25, "lifestyle": 0.15},
    "negative_rules": {"exclude_tags": ["fast_paced", "comparison"], "max_arousal_score": 0.4},
}
```

**输出**：

```python
{
    "status": "SUCCESS",              # 或 "SUCCESS_WITH_REFERRAL"（高危 + 已转介）
    "risk_level": "Level_2_Moderate",
    "recommendations": [{"id", "text", "target_issue", "similarity_score", "final_score"}, ...],
    "referral": None,                 # 高危时：{timestamp, risk_level, target_issue, reason, logged}
    "crisis_resources": [],           # 高危时附带危机资源卡片（热线等），不作为唯一输出
}
```

## 关键结论（实验支撑）

- **必须走英文 `embedding_prompt`**：中文原文直接编码查英文池会失效（Hit@5 从 97.8% 掉到 13%）。classifier 需输出英文语义描述，详见 `scripts/compare_backends.py`。
- **相似度分数只在同后端内可比**：哈希基线相似度虚高不可靠，MiniLM 低而准。

## 当前阶段边界

- **英文数据集闭环即可**：整条流水线（classifier 英文 Prompt → 英文干预池 → 语义检索）以英文为验收标准；中文干预池（人工策展中文疏导内容）列为后续阶段，代码复用无需改动。
- **转介只后台落盘**：高危用户的后台转介当前只写入 `data/risk_referrals.csv`，「推给业务线同学」的自动推送通道（腾讯文档/企业微信/内部 API）为预留项，后续接通道只改 `src/referral.py`。

## 分工

| 模块 | 负责人 |
|---|---|
| recommender / pipeline / 整体结构 | 李鹏飞 |
| classifier / embedder / preprocess | 寇丽雯、李鑫 |
| Streamlit 界面 | 王定祥 |
