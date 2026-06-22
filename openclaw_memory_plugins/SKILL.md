# OpenClaw Memory Plugin

Use this plugin pack when OpenClaw needs durable memory with retrieval, scoring, reflection, and contradiction checks.

## Trigger
Load this skill when the task mentions:
- memory, remember, forget, recall, preference, decision, reflection, contradiction
- persistent session state or project memory
- Hermes-compatible memory behavior
- OpenClawMemoryGovernor or OpenClawHermesMemoryProvider

## What It Provides
- SQLite-backed memory storage with FTS5 full-text search
- Entity linking for topic and identity association
- Hybrid recall with fragment search, entity search, and embedding search
- Trust scoring and feedback updates
- Reflection and skill-candidate extraction at session end
- Contradiction detection for conflicting memories

## Main Entry Points
- `openclaw_memory_plugins.register:register`
- `openclaw_memory_plugins.register:bootstrap`
- `openclaw_memory_plugins.register:handle_tool_call`
- `openclaw_memory_plugins.memory_governor:OpenClawMemoryGovernor`
- `openclaw_memory_plugins.hermes_provider:OpenClawHermesMemoryProvider`

## Runtime Flow
1. Prefetch relevant memories before the model answers.
2. Sync turn content after each exchange.
3. Auto-extract notable facts, preferences, decisions, and failures.
4. Reflect at session end to capture lessons and reusable skills.
5. Review contradictions before finalizing the session.

## Tool Surface
- `memory_add`
- `memory_search`
- `memory_feedback`
- `memory_contradict`
- `memory_forget`
- `memory_state`
- `memory_reflect`

## Notes
- Treat `OpenClawMemoryGovernor` as the orchestration layer.
- Treat `OpenClawHermesMemoryProvider` as the compatibility wrapper.
- Use the SQLite store as source of truth; JSONL trails are compatibility artifacts.
- Prefer embedding retrieval for semantic recall, but keep the system functional if embeddings fall back to hashing.
