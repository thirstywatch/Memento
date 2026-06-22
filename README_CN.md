# Memento

> *"记住。"* — 为 OpenClaw 打造的 Hermes 风格长期记忆引擎。

Memento 让你的 AI 智能体拥有持久、可搜索、自我进化的记忆能力。它能跨会话记住用户偏好、发现记忆矛盾、根据反馈自我训练、让过时信息自动衰减——全部运行在一个 SQLite 文件里。

**4011 行 Python。7 个工具。标准库之外零必需依赖。**

---

## 它做什么

```
每条消息之前              对话之中                  每个会话结束
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ prefetch(查询)    │    │ memory_add()     │    │ auto-extract     │
│ FTS5 + 语义      │    │ memory_search()  │    │ contradict()     │
│ + 实体 + LIKE    │    │ memory_feedback()│    │ reflect()        │
│ → 注入到         │    │ memory_forget()  │    │ → 导出回         │
│   system prompt  │    │ memory_reflect() │    │   markdown 文件  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 核心能力

| 能力 | 如何实现 |
|------|---------|
| **全文搜索** | SQLite FTS5 + BM25 排名 |
| **语义搜索** | Transformer 嵌入，零依赖哈希降级 |
| **实体关联** | 自动提取人名、缩写、驼峰词、引号术语、AKA 别名 |
| **信任评分** | 用户反馈驱动：有用 +0.05，无用 −0.10 |
| **时间衰减** | 90 天半衰期，旧记忆自动降低权重 |
| **三级门控** | 自动写入 / 暂存待确认 / 丢弃 |
| **矛盾检测** | O(n²) 矩阵扫描 + 极性分析 |
| **自动提取** | 正则匹配偏好、决策、失败模式 |
| **Hermes 兼容** | 完整 `MemoryProvider` ABC 生命周期适配 |
| **桥接同步** | 与 OpenClaw markdown 记忆文件双向导入导出 |

---

## 架构

```
OpenClawRuntimeAdapter          ← 你的接入点
  │
  ├── attach_to_context(ctx)    ← 注入工具 + Schema + System Prompt
  │
  ├── build_system_prompt(msg)  ← prefetch + 系统提示块
  │
  ├── handle_tool_call(名称,参数)← 路由全部 7 个工具
  │     │
  │     └── OpenClawMemoryGovernor
  │           │
  │           ├── MemoryWorkflow
  │           │     ├── MemoryStore        (SQLite + FTS5)
  │           │     ├── MemoryScorer       (三级门控)
  │           │     ├── MemoryRetriever    (四层搜索)
  │           │     └── MemoryReflector    (教训 + 技能)
  │           │
  │           ├── EntityExtractor          (5 条正则规则)
  │           ├── EmbeddingBackend         (Transformer + 哈希降级)
  │           ├── OpenClawMemoryBridge     (markdown ↔ SQLite)
  │           ├── Contradiction 扫描器     (实体 + 极性)
  │           └── 自动提取引擎             (偏好/决策/失败 模式)
  │
  └── OpenClawHermesMemoryProvider ← Hermes 插件兼容
```

---

## 快速开始

### 安装

```bash
pip install -e /path/to/memento/plugins
```

### 第一次运行

```python
from openclaw_memory_plugins import build_runtime

# 指向你的 OpenClaw 目录
runtime = build_runtime(
    openclaw_home="~/.openclaw",
    workspace_dir="~/.openclaw/workspace",
    self_improving_dir="~/self-improving",
    proactivity_dir="~/proactivity",
)

# 从 OpenClaw 的 markdown 记忆文件导入历史
runtime.sync_openclaw_memory(import_surface=True)

# 搜索记忆
result = runtime.prefetch("用户喜欢用什么编辑器？")
print(result)
# <memory-context>
# - [user/preference] 用户偏好 nvim > vscode
# </memory-context>

# 添加一条记忆
import json
print(json.loads(runtime.handle_tool_call("memory_add", {
    "target": "user",
    "content": "用户喜欢深色主题。",
    "kind": "preference",
    "confidence": 0.9,
})))

# 反馈一条记忆的质量
print(json.loads(runtime.handle_tool_call("memory_feedback", {
    "record_id": "mem_abc123",
    "helpful": True,
})))

# 检测记忆矛盾
print(json.loads(runtime.handle_tool_call("memory_contradict", {
    "query": "编辑器偏好",
})))
```

---

## 七种工具

| 工具 | 作用 | 关键参数 |
|------|------|---------|
| `memory_add` | 持久化一条事实 | `target`（memory/user），`content`，`kind`，`confidence`，`tags` |
| `memory_search` | 按查询搜索记忆 | `query`，`domain`（user/project/agent），`limit`，`max_chars` |
| `memory_feedback` | 标记记忆有用/无用 | `record_id`，`helpful`，`note`，`weight` |
| `memory_contradict` | 发现自相矛盾的主张 | `query`，`domain`，`threshold`，`limit` |
| `memory_forget` | 移除过时记录 | `fragment`，`domain` |
| `memory_reflect` | 存储教训和可复用技能 | `task_title`，`result_summary`，`lessons`，`skill_steps` |
| `memory_state` | 查看记忆系统状态 | *(无)* |

---

## 数据模型

```
memories                    entities
┌──────────────────┐       ┌──────────────────┐
│ id               │       │ id               │
│ domain (U/P/A)   │       │ name             │
│ kind (pref/fact/ │       │ normalized       │
│   decision/...)  │       │ type             │
│ content          │       │ aliases          │
│ confidence       │       └────────┬─────────┘
│ trust            │                │
│ status           │    memory_entities (M:N)
│ tags             │    ┌───────────┴─────────┐
│ metadata (JSON)  │    │ memory_id           │
│ created_at       │    │ entity_id           │
│ updated_at       │    └─────────────────────┘
└────────┬─────────┘
         │         memory_embeddings
memories_fts (FTS5)┌──────────────────┐
┌──────────────────┐│ memory_id        │
│ content          ││ model_name       │
│ tags             ││ backend          │
│ search_text      ││ dimension        │
└──────────────────┘│ vector (BLOB)    │
                    └──────────────────┘
```

---

## 搜索管道

```
用户查询："帮我配置编辑器"

  第一层：FTS5 MATCH          → BM25 关键词排名
  第二层：Embedding 余弦      → 语义相似度（Transformer 或哈希降级）
  第三层：Entity JOIN         → 与查询实体关联的记忆
  第四层：LIKE 降级           → 子串匹配（FTS5 不可用时）

  打分公式：(关键词重叠 + 置信度 + 信任分 + 类型 + 域) × 时间衰减 + 语义增强

  最终结果："[0.82] 用户偏好 nvim > vscode"
```

---

## 矛盾检测

```
候选池 = FTS5(query) + Entity(query) + 全量（上限 500 条）

for 左, 右 in 候选池 × 候选池:
  实体重叠 = 共享实体 / 全部实体
  if < 0.25: 跳过（不够相关）

  内容相似度 = Jaccard(左词汇集, 右词汇集)
  左极性  = positive / negative / neutral
  右极性  = positive / negative / neutral

  矛盾分 = 实体重叠 × (1 − 内容相似度)
  if 极性冲突:          分 += 0.18
  if 类型不一致:        分 += 0.05
  if 同域内容矛盾:      分 += 0.22

  if 矛盾分 ≥ 0.28: 报告为一对矛盾
```

---

## 与 Hermes 对照

| 能力 | Hermes | Memento |
|------|--------|---------|
| SQLite + FTS5 | ✅ | ✅ |
| 实体提取 + 关联 | ✅ | ✅ |
| 信任评分 + 反馈 | ✅ | ✅ |
| 时间衰减 | ✅ | ✅ |
| 门控评分（自动写入/暂存/丢弃） | ✅ | ✅ |
| 矛盾检测 | ✅ | ✅（增加极性分析） |
| 反思 + 技能提取 | ✅ | ✅ |
| 语义嵌入搜索 | ❌ | ✅ |
| 零依赖降级方案 | ❌ | ✅（哈希嵌入） |
| OpenClaw 原生适配 | ❌ | ✅ |
| Hermes 兼容层 | ✅ | ✅ |
| Markdown 桥接同步 | ❌ | ✅ |
| HRR 全息代数推理 | ✅ | ❌（改用 Embedding 替代） |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENCLAW_MEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | HuggingFace 嵌入模型 |
| `OPENCLAW_MEMORY_EMBEDDING_ALLOW_DOWNLOAD` | `1` | 允许自动下载模型 |
| `OPENCLAW_MEMORY_EMBEDDING_FALLBACK_DIMENSION` | `384` | 哈希嵌入向量维度 |
| `OPENCLAW_MEMORY_EMBEDDING_MAX_TOKENS` | `256` | Transformer 编码最大 token 数 |
| `OPENCLAW_HOME` | *(无)* | OpenClaw 主目录路径 |
| `OPENCLAW_WORKSPACE_DIR` | *(无)* | OpenClaw 工作区路径 |
| `OPENCLAW_SELF_IMPROVING_DIR` | `~/self-improving` | 自我改进记忆目录 |
| `OPENCLAW_PROACTIVITY_DIR` | `~/proactivity` | 主动记忆目录 |

---

## 项目结构

```
plugins/
├── README.md                   ← 英文版
├── README_CN.md                ← 你在这里
├── manifest.json               ← 插件注册表
│
└── openclaw_memory_plugins/    ← Python 包（14 文件，4011 行）
    │
    ├── SKILL.md                技能入口
    ├── AGENT.md                memory-agent 系统提示
    │
    ├── types.py                数据模型（MemoryRecord, MemoryDomain, MemoryKind）
    ├── memory_store.py         SQLite + FTS5 + Entity + Embedding 存储
    ├── memory_retrieve.py      四层搜索（FTS5 → 语义 → 实体 → LIKE）
    ├── memory_score.py         三级门控评分（auto_write / stage / drop）
    ├── memory_entities.py      实体提取（5 条正则规则）
    ├── memory_embeddings.py    双层嵌入后端（Transformer + 哈希降级）
    ├── memory_reflect.py       反思 + 技能候选构建
    ├── memory_workflow.py      编排层（录入 → 检索 → 反思）
    ├── memory_governor.py      生命周期调度（18 个钩子 + 矛盾检测）
    ├── openclaw_adapter.py     OpenClaw 原生运行时适配器
    ├── openclaw_bridge.py      markdown ↔ SQLite 双向同步桥
    ├── hermes_provider.py      Hermes MemoryProvider ABC 兼容层
    ├── register.py             OpenClaw 入口 + 启动引导
    └── __init__.py             公开 API 导出
```

---

## 许可证

MIT

---

## 致谢

受 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 记忆系统架构启发。为 [OpenClaw](https://github.com/nicepkg/openclaw) 而生。
