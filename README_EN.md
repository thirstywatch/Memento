# Memento

> *"Remember."* — A Hermes-inspired long-term memory engine for OpenClaw.

Memento gives your AI agent a durable, searchable, self-improving memory. It remembers user preferences across sessions, discovers contradictions, learns from feedback, and decays stale information — all inside a single SQLite file.

**4011 lines of Python. 7 tools. 0 required dependencies beyond the standard library.**

---

## What It Does

```
Before every message     During conversation        After every session
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ prefetch(query)   │    │ memory_add()     │    │ auto-extract     │
│ FTS5 + Embedding  │    │ memory_search()  │    │ contradict()     │
│ + Entity + LIKE   │    │ memory_feedback()│    │ reflect()        │
│ → inject into     │    │ memory_forget()  │    │ → export back to │
│   system prompt   │    │ memory_reflect() │    │   markdown files │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## Features

| Capability | How |
|------------|-----|
| **Full-text search** | SQLite FTS5 with BM25 ranking |
| **Semantic search** | Transformer embeddings with zero-dependency hashing fallback |
| **Entity linking** | Auto-extract names, acronyms, CamelCase, quoted terms, AKA aliases |
| **Trust scoring** | Feedback-driven: helpful +0.05, unhelpful −0.10 |
| **Temporal decay** | 90-day half-life, old memories fade unless reinforced |
| **Scoring gate** | Three-tier: auto-write / stage / drop |
| **Contradiction detection** | O(n²) matrix scan with polarity analysis |
| **Auto-extraction** | Regex patterns for preferences, decisions, failures |
| **Hermes-compatible** | Full `MemoryProvider` ABC lifecycle adapter |
| **Bridge sync** | Bidirectional import/export with OpenClaw markdown memory files |

---

## Architecture

```
OpenClawRuntimeAdapter          ← Your integration point
  │
  ├── attach_to_context(ctx)    ← Injects tools + schemas + system prompt
  │
  ├── build_system_prompt(msg)  ← prefetch + system prompt block
  │
  ├── handle_tool_call(name,args)← Routes all 7 tools
  │     │
  │     └── OpenClawMemoryGovernor
  │           │
  │           ├── MemoryWorkflow
  │           │     ├── MemoryStore        (SQLite + FTS5)
  │           │     ├── MemoryScorer       (3-tier gate)
  │           │     ├── MemoryRetriever    (4-layer search)
  │           │     └── MemoryReflector    (lessons + skills)
  │           │
  │           ├── EntityExtractor          (5 regex rules)
  │           ├── EmbeddingBackend         (transformers + hash fallback)
  │           ├── OpenClawMemoryBridge     (markdown ↔ SQLite)
  │           ├── Contradiction scanner    (entities + polarity)
  │           └── Auto-extraction engine   (pref / decision / failure patterns)
  │
  └── OpenClawHermesMemoryProvider ← Hermes plugin compatibility
```

---

## Quick Start

### Install

```bash
pip install -e /path/to/memento/plugins
```

### First run

```python
from openclaw_memory_plugins import build_runtime

# Point to your OpenClaw directories
runtime = build_runtime(
    openclaw_home="~/.openclaw",
    workspace_dir="~/.openclaw/workspace",
    self_improving_dir="~/self-improving",
    proactivity_dir="~/proactivity",
)

# Import existing memory from OpenClaw markdown files
runtime.sync_openclaw_memory(import_surface=True)

# Search
result = runtime.prefetch("What editor does the user prefer?")
print(result)
# <memory-context>
# - [user/preference] 用户偏好 nvim > vscode
# </memory-context>

# Add a memory
import json
print(json.loads(runtime.handle_tool_call("memory_add", {
    "target": "user",
    "content": "The user prefers dark themes.",
    "kind": "preference",
    "confidence": 0.9,
})))

# Get feedback on a memory
print(json.loads(runtime.handle_tool_call("memory_feedback", {
    "record_id": "mem_abc123",
    "helpful": True,
})))

# Detect contradictions
print(json.loads(runtime.handle_tool_call("memory_contradict", {
    "query": "editor preference",
})))
```

---

## Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_add` | Persist a durable fact | `target` (memory/user), `content`, `kind`, `confidence`, `tags` |
| `memory_search` | Search memory by query | `query`, `domain` (user/project/agent), `limit`, `max_chars` |
| `memory_feedback` | Rate a memory helpful/unhelpful | `record_id`, `helpful`, `note`, `weight` |
| `memory_contradict` | Find conflicting claims | `query`, `domain`, `threshold`, `limit` |
| `memory_forget` | Remove stale records | `fragment`, `domain` |
| `memory_reflect` | Store lessons + skill candidates | `task_title`, `result_summary`, `lessons`, `skill_steps` |
| `memory_state` | Inspect memory system status | *(none)* |

---

## Data Model

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

## Search Pipeline

```
User query: "帮我配置编辑器"

  Layer 1: FTS5 MATCH          → BM25 keyword ranking
  Layer 2: Embedding cosine    → Semantic similarity (transformers or hash fallback)
  Layer 3: Entity JOIN         → Records linked to same entities
  Layer 4: LIKE fallback       → Substring match (when FTS5 unavailable)

  Scoring: (token_overlap + confidence + trust + kind + domain) × decay + embedding_boost

  Result: "[0.82] user偏好 nvim > vscode"
```

---

## Contradiction Detection

```
pool = FTS5(query) + Entity(query) + all (capped at 500)

for left, right in pool × pool:
  entity_overlap = shared_entities / total_entities
  if < 0.25: skip

  content_similarity = Jaccard(token_set(left), token_set(right))
  polarity_left  = positive / negative / neutral
  polarity_right = positive / negative / neutral

  score = entity_overlap × (1 − content_similarity)
  if polarity conflict:     score += 0.18
  if kind disagreement:     score += 0.05
  if domain-level conflict: score += 0.22

  if score ≥ 0.28: report contradiction
```

---

## vs Hermes

| Capability | Hermes | Memento |
|------------|--------|---------|
| SQLite + FTS5 | ✅ | ✅ |
| Entity extraction + linking | ✅ | ✅ |
| Trust scoring + feedback | ✅ | ✅ |
| Temporal decay | ✅ | ✅ |
| Scoring gate (auto_write/stage/drop) | ✅ | ✅ |
| Contradiction detection | ✅ | ✅ (with polarity analysis) |
| Reflection + skill extraction | ✅ | ✅ |
| Semantic embedding search | ❌ | ✅ |
| Zero-dependency fallback | ❌ | ✅ (hash embedding) |
| OpenClaw native adapter | ❌ | ✅ |
| Hermes provider compat | ✅ | ✅ |
| Markdown bridge sync | ❌ | ✅ |
| HRR holographic algebra | ✅ | ❌ (replaced by embeddings) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCLAW_MEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | HuggingFace model for embeddings |
| `OPENCLAW_MEMORY_EMBEDDING_ALLOW_DOWNLOAD` | `1` | Allow automatic model download |
| `OPENCLAW_MEMORY_EMBEDDING_FALLBACK_DIMENSION` | `384` | Hash embedding vector dimension |
| `OPENCLAW_MEMORY_EMBEDDING_MAX_TOKENS` | `256` | Max tokens for transformer encoding |
| `OPENCLAW_HOME` | *(none)* | Path to OpenClaw home directory |
| `OPENCLAW_WORKSPACE_DIR` | *(none)* | Path to OpenClaw workspace |
| `OPENCLAW_SELF_IMPROVING_DIR` | `~/self-improving` | Self-improving memory directory |
| `OPENCLAW_PROACTIVITY_DIR` | `~/proactivity` | Proactivity memory directory |

---

## Project Structure

```
plugins/
├── README.md                   ← You are here
├── manifest.json               ← Plugin registry
│
└── openclaw_memory_plugins/    ← Python package (14 files, 4011 lines)
    │
    ├── SKILL.md                Skill entry point
    ├── AGENT.md                memory-agent system prompt
    │
    ├── types.py                Data model (MemoryRecord, MemoryDomain, MemoryKind)
    ├── memory_store.py         SQLite + FTS5 + Entity + Embedding storage
    ├── memory_retrieve.py      4-layer search (FTS5 → Semantic → Entity → LIKE)
    ├── memory_score.py         3-tier scoring gate (auto_write / stage / drop)
    ├── memory_entities.py      Entity extraction (5 regex rules)
    ├── memory_embeddings.py    Dual backend (transformers + hash fallback)
    ├── memory_reflect.py       Reflection + skill candidate builder
    ├── memory_workflow.py      Orchestration (ingest → retrieve → reflect)
    ├── memory_governor.py      Lifecycle governor (18 hooks + contradict)
    ├── openclaw_adapter.py     OpenClaw native runtime adapter
    ├── openclaw_bridge.py      Bidirectional markdown ↔ SQLite sync
    ├── hermes_provider.py      Hermes MemoryProvider ABC compatibility
    ├── register.py             OpenClaw entrypoint + bootstrap
    └── __init__.py             Public API exports
```

---

## License

MIT

---

## Credits

Inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent)'s memory system architecture. Built for [OpenClaw](https://github.com/nicepkg/openclaw).
