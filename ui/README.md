# Feta UI 展示层

类微博/推特的学生端信息流 + 运营侧监测后台，**标准库实现、零新增依赖**，直接消费 `src/pipeline.py` 的真实四段式结果。

## 启动

```bash
# 项目根目录
uv run python ui/server.py          # 或双击 start_ui.bat（Windows）
```

- 首次启动需加载本地 MiniLM 与风险四分类模型，并对演示语料逐条跑 pipeline，约 30~60 秒
- 页面：信息流 <http://127.0.0.1:8765/> ｜ 监测后台 <http://127.0.0.1:8765/dashboard>
- 前置条件：`data/interventions.csv` 存在（缺失时执行 `uv run python scripts/make_demo_pool.py` 重建）

## 目录结构

```
ui/
├── server.py            # 标准库 ThreadingHTTPServer：页面 + JSON API
├── mock_data.py         # 演示语料（41 条英文动态，时间戳相对启动时刻分布 30 天）
├── templates/
│   ├── feed.html        # 信息流页（发布入口 + 信息流容器 + 用户侧右栏）
│   └── dashboard.html   # 监测后台（KPI / 图表 / 告警 / 实时分析）
└── static/
    ├── css/style.css    # 设计规范：主色 indigo，风险四档配色，三级阴影，响应式断点 900/600
    └── js/
        ├── feed.js       # 无限滚动、点赞评论、发布、演示高亮
        ├── dashboard.js  # 指标轮询、告警筛选、实时分析
        └── charts.js     # 零依赖 SVG 图表（环图 / 双线趋势 / 条形图）
```

## 两个模块

**模块一 · 信息流（用户视角）**
- 发布入口 → `POST /api/publish`：走真实 pipeline 分析，帖子置顶并紧跟系统推荐
- 动态卡片：头像 / 昵称 / 正文 / 配图区 / 点赞评论互动区
- 时间线倒序 + 无限滚动（IntersectionObserver 触底分页）
- **无痕推荐**：干预内容包装成普通用户推文（虚拟作者、时间、互动数），用户视角零品牌痕迹；数据侧以 `rec-` 前缀区分
- **演示辅助**：`显示风险标签` 展示四档标签与推荐标记；`模拟高危会话` 验证危机卡渲染；自己帖子上的 `高亮相关推荐` 一键圈出「博文 + 跟随推荐」用于讲解

**模块二 · 监测后台（运营视角）**
- 核心指标：动态总数 / 高危动态 / 待处理转介 / 平均置信度
- 图表：风险四档分布、动态趋势双线（总量 vs 高危）、压力源 Top 8
- 告警列表：状态（待处理/已处理）× 级别（焦虑/抑郁/高危）双维筛选，可标记已处理
- 时间范围控件：24h / 7d / 30d；单条文本实时分析（支持强制高危演示）

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/feed?page=N[&crisis=1]` | 分页动态；`crisis=1` 触发危机卡演示 |
| GET | `/api/metrics?range=24h\|7d\|30d` | 指标与图表聚合数据 |
| GET | `/api/alerts?status=&level=` | 告警列表（支持双维筛选） |
| POST | `/api/alerts/resolve` | 标记告警已处理 `{"id": "..."}` |
| POST | `/api/analyze` | 单条文本实时分析 `{"text": "...", "force_risk_level": 可选}` |
| POST | `/api/publish` | 发布动态 `{"text": "..."}`，返回跟随推荐条数 |

## 数据边界与护栏

- 启动时对演示语料逐条跑**真实 pipeline**（MiniLM + 风险模型），指标非假数据
- 演示转介写入 `data/demo_risk_referrals.csv`、告警写入 `data/ui_alerts.csv`，**不污染**真实 `data/risk_referrals.csv`
- 危机卡只下发 `user_facing` 字段；`staff_actions`（内部运营字段）**永不出 API**
- 风险判定与转介叙事只出现在监测后台，信息流一侧只见支持性内容
