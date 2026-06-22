from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from openclaw_memory_plugins import MemoryStore, OpenClawHermesMemoryProvider, OpenClawRuntimeAdapter
from openclaw_memory_plugins.register import bootstrap, register


class _FakeContext:
    def __init__(self) -> None:
        self.providers: list[object] = []
        self.memory_provider: object | None = None
        self.tool_schemas: list[dict[str, object]] = []
        self.tool_call = None
        self.system_prompt_block = None
        self.memory_runtime = None
        self.openclaw_memory = None

    def register_memory_provider(self, provider: object) -> None:
        self.providers.append(provider)

    def register_tools(self, tools: list[dict[str, object]]) -> None:
        self.tool_schemas = list(tools)


def run_smoke_test() -> dict[str, object]:
    """Exercise the Hermes-compatible adapter on a temp store."""

    with TemporaryDirectory(prefix="openclaw-hermes-provider-smoke-") as tmp_dir:
        with MemoryStore(tmp_dir) as store:
            with OpenClawHermesMemoryProvider(storage_root=tmp_dir) as provider:
                provider.initialize("smoke-session", hermes_home=tmp_dir, platform="cli")

                seeded = provider.handle_tool_call(
                    "memory_add",
                    {
                        "target": "user",
                        "content": "OpenClawMemoryGovernor integrates with OpenClawMemoryProvider.",
                        "kind": "fact",
                        "confidence": 0.95,
                    },
                )
                provider.handle_tool_call(
                    "memory_add",
                    {
                        "target": "user",
                        "content": "OpenClawMemoryGovernor does not integrate with OpenClawMemoryProvider.",
                        "kind": "fact",
                        "confidence": 0.9,
                    },
                )
                contradict = json.loads(
                    provider.handle_tool_call(
                        "memory_contradict",
                        {"query": "OpenClawMemoryGovernor", "limit": 5},
                    )
                )
                memory_context = provider.prefetch("OpenClawMemoryGovernor", session_id="smoke-session")
                provider.sync_turn(
                    "Remember OpenClawMemoryGovernor integrates with OpenClawMemoryProvider.",
                    "Got it, I will keep it connected.",
                    session_id="smoke-session",
                    messages=[
                        {"role": "user", "content": "OpenClawMemoryGovernor integrates with OpenClawMemoryProvider."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                )
                session_end = provider.on_session_end(
                    [
                        {"role": "user", "content": "OpenClawMemoryGovernor integrates with OpenClawMemoryProvider."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                    task_title="smoke test provider",
                    result_summary="The provider wrapper runs end to end.",
                    lessons="Track contradictions through the governor.",
                )

                bundle = bootstrap()
                registered = register(_FakeContext())
                assert seeded, "expected the adapter to return a tool result"
                assert memory_context.startswith("<memory-context>")
                assert contradict["results"], "expected a contradiction result"
                assert session_end["contradictions"], "expected session end to include contradiction review"
                assert store.records_path.exists()
                assert store.pending_path.exists()
                assert store.reflections_path.exists()
                assert store.feedback_path.exists()
                return {
                    "root_dir": tmp_dir,
                    "bootstrap": {
                        "provider_name": bundle["provider"].name,
                        "tool_names": [schema["name"] for schema in bundle["tool_schemas"]],
                    },
                    "registered": registered,
                    "memory_context": memory_context,
                    "seeded": json.loads(seeded),
                    "contradict": contradict,
                    "session_end": session_end,
                    "counts": {
                        "records": len(store.list_records()),
                        "pending": len(list(store._iter_jsonl(store.pending_path))),
                        "reflections": len(list(store._iter_jsonl(store.reflections_path))),
                        "feedback": len(list(store._iter_jsonl(store.feedback_path))),
                    },
                }


def run_runtime_smoke_test() -> dict[str, object]:
    """Exercise the OpenClaw runtime adapter end to end."""

    with TemporaryDirectory(prefix="openclaw-runtime-smoke-") as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        with MemoryStore(Path(tmp_dir) / "memory") as store:
            runtime = OpenClawRuntimeAdapter(
                governor=None,
                storage_root=store.root_dir,
                session_id="runtime-session",
                platform="openclaw",
                openclaw_home=tmp_dir,
                workspace_dir=workspace_dir,
            )
            try:
                runtime.initialize("runtime-session", openclaw_home=tmp_dir, workspace_dir=workspace_dir, platform="openclaw")
                context = _FakeContext()
                attached = runtime.attach_to_context(context)
                bundle = runtime.build_bundle(message="OpenClaw should wire memory into runtime prompts.", session_id="runtime-session")
                turn = runtime.on_turn_start(1, "Remember OpenClaw runtime memory integration.", domain="project")
                turn_end = runtime.on_turn_end(
                    "Remember OpenClaw runtime memory integration.",
                    "Confirmed.",
                    messages=[
                        {"role": "user", "content": "Remember OpenClaw runtime memory integration."},
                        {"role": "assistant", "content": "Confirmed."},
                    ],
                    task_title="runtime smoke test",
                    result_summary="The runtime adapter wires prompt, tools, and turn hooks.",
                )
                session_end = runtime.on_session_end(
                    [
                        {"role": "user", "content": "Remember OpenClaw runtime memory integration."},
                        {"role": "assistant", "content": "Confirmed."},
                    ],
                    task_title="runtime smoke test",
                    result_summary="The runtime adapter wires prompt, tools, and turn hooks.",
                )
                assert attached["runtime"] == "memento-runtime"
                assert "memory_add" in attached["tool_names"]
                assert bundle.system_prompt_block.startswith("# OpenClaw Memory Governor")
                assert turn["memory_context"].startswith("<memory-context>")
                assert turn_end["memory_context"].startswith("<memory-context>")
                return {
                    "root_dir": tmp_dir,
                    "attached": attached,
                    "bundle": bundle.to_dict(),
                    "turn": turn,
                    "turn_end": turn_end,
                    "session_end": session_end,
                    "context": {
                        "provider": context.memory_provider.name if context.memory_provider else None,
                        "tool_names": [schema["name"] for schema in context.tool_schemas],
                    },
                    "counts": {
                        "records": len(store.list_records()),
                        "reflections": len(list(store._iter_jsonl(store.reflections_path))),
                    },
                }
            finally:
                runtime.close()


def main() -> int:
    result = {
        "provider_smoke": run_smoke_test(),
        "runtime_smoke": run_runtime_smoke_test(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
