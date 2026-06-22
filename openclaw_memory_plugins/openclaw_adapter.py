from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hermes_provider import OpenClawHermesMemoryProvider
from .memory_governor import OpenClawMemoryGovernor
from .memory_store import MemoryStore


@dataclass(slots=True)
class OpenClawRuntimeBundle:
    provider: OpenClawHermesMemoryProvider
    governor: OpenClawMemoryGovernor
    system_prompt_block: str
    tool_schemas: list[dict[str, Any]]
    memory_context: str = ""
    session: dict[str, Any] | None = None
    bridge_sync: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "system_prompt_block": self.system_prompt_block,
            "tool_names": [schema.get("name", "") for schema in self.tool_schemas],
            "memory_context": self.memory_context,
            "session": dict(self.session or {}),
            "bridge_sync": self.bridge_sync,
        }


class OpenClawRuntimeAdapter:
    """OpenClaw-native orchestration surface for the memory plugin."""

    def __init__(
        self,
        governor: OpenClawMemoryGovernor | None = None,
        *,
        storage_root: str | Path | None = None,
        session_id: str = "",
        platform: str = "openclaw",
        auto_extract: bool = True,
        auto_mirror: bool = True,
        openclaw_home: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        self_improving_dir: str | Path | None = None,
        proactivity_dir: str | Path | None = None,
    ) -> None:
        self._storage_root = Path(storage_root).expanduser() if storage_root else None
        self._auto_extract = auto_extract
        self._auto_mirror = auto_mirror
        self._governor = governor
        self._openclaw_home = Path(openclaw_home).expanduser() if openclaw_home else None
        self._workspace_dir = Path(workspace_dir).expanduser() if workspace_dir else None
        self._self_improving_dir = Path(self_improving_dir).expanduser() if self_improving_dir else None
        self._proactivity_dir = Path(proactivity_dir).expanduser() if proactivity_dir else None
        if self._governor is not None:
            if session_id:
                self._governor.session_id = session_id
            self._governor.platform = platform or self._governor.platform

    @property
    def name(self) -> str:
        return "openclaw-memory-runtime"

    @property
    def governor(self) -> OpenClawMemoryGovernor:
        return self._ensure_governor()

    @property
    def provider(self) -> OpenClawHermesMemoryProvider:
        return OpenClawHermesMemoryProvider(governor=self._ensure_governor())

    def _ensure_governor(self) -> OpenClawMemoryGovernor:
        if self._governor is None:
            store = MemoryStore(self._storage_root) if self._storage_root else None
            self._governor = OpenClawMemoryGovernor(
                store=store,
                platform="openclaw",
                auto_extract=self._auto_extract,
                auto_mirror=self._auto_mirror,
                openclaw_home=self._openclaw_home,
                workspace_dir=self._workspace_dir,
                self_improving_dir=self._self_improving_dir,
                proactivity_dir=self._proactivity_dir,
            )
        return self._governor

    def _storage_path(self) -> Path:
        return Path(self._ensure_governor().workflow.store.root_dir).expanduser()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._ensure_governor().initialize(session_id, **kwargs)

    def system_prompt_block(self) -> str:
        return self._ensure_governor().system_prompt_block()

    def prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
        domain: str | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        governor = self._ensure_governor()
        if session_id:
            governor.session_id = session_id
        return governor.prefetch(query, domain=domain, limit=limit, max_chars=max_chars)

    def prefetch_turn(
        self,
        message: str,
        *,
        session_id: str = "",
        turn_number: int = 0,
        domain: str | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> dict[str, Any]:
        governor = self._ensure_governor()
        if session_id:
            governor.session_id = session_id
        if turn_number:
            governor.on_turn_start(turn_number, message)
        memory_context = governor.prefetch(message, domain=domain, limit=limit, max_chars=max_chars)
        return {
            "memory_context": memory_context,
            "system_prompt_block": governor.system_prompt_block(),
            "session": governor._sync_session_snapshot(),
            "recall": governor.last_recall.to_dict() if governor.last_recall else None,
        }

    def build_system_prompt(
        self,
        message: str,
        *,
        session_id: str = "",
        turn_number: int = 0,
        domain: str | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        bundle = self.prefetch_turn(
            message,
            session_id=session_id,
            turn_number=turn_number,
            domain=domain,
            limit=limit,
            max_chars=max_chars,
        )
        parts = [bundle["system_prompt_block"]]
        memory_context = bundle.get("memory_context") or ""
        if memory_context:
            parts.append(memory_context)
        return "\n\n".join(part for part in parts if part)

    def queue_prefetch(
        self,
        query: str,
        *,
        session_id: str = "",
        domain: str | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> None:
        governor = self._ensure_governor()
        if session_id:
            governor.session_id = session_id
        governor.queue_prefetch(query, domain=domain, limit=limit, max_chars=max_chars)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        domain: str | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        governor = self._ensure_governor()
        if session_id:
            governor.session_id = session_id
        return governor.sync_turn(
            user_content,
            assistant_content,
            domain=domain,
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            skill_steps=skill_steps,
            messages=messages,
        ).to_dict()

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> dict[str, Any]:
        governor = self._ensure_governor()
        governor.on_turn_start(turn_number, message, **kwargs)
        memory_context = governor.prefetch(
            message,
            domain=kwargs.get("domain"),
            limit=int(kwargs.get("limit") or 5),
            max_chars=int(kwargs.get("max_chars") or 800),
        )
        return {
            "memory_context": memory_context,
            "system_prompt_block": governor.system_prompt_block(),
            "session": governor._sync_session_snapshot(),
        }

    def on_turn_end(
        self,
        user_content: str,
        assistant_content: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        domain: str | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        return self.sync_turn(
            user_content,
            assistant_content,
            session_id=session_id,
            messages=messages,
            domain=domain,
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            skill_steps=skill_steps,
        )

    def on_session_end(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return self._ensure_governor().on_session_end(messages=messages, **kwargs)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._ensure_governor().on_session_switch(
            new_session_id,
            parent_session_id=parent_session_id,
            reset=reset,
            rewound=rewound,
            **kwargs,
        )

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return self._ensure_governor().on_pre_compress(messages)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> dict[str, Any]:
        return self._ensure_governor().on_delegation(task, result, child_session_id=child_session_id, **kwargs)

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._ensure_governor().on_memory_write(action, target, content, metadata)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self._ensure_governor().get_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        return self._ensure_governor().handle_tool_call(tool_name, args, **kwargs)

    def sync_openclaw_memory(
        self,
        *,
        import_surface: bool = True,
        export_surface: bool = False,
        reason: str = "manual",
    ) -> dict[str, Any]:
        return self._ensure_governor().sync_openclaw_memory(
            import_surface=import_surface,
            export_surface=export_surface,
            reason=reason,
        )

    def build_bundle(self, *, message: str = "", session_id: str = "", turn_number: int = 0, domain: str | None = None) -> OpenClawRuntimeBundle:
        governor = self._ensure_governor()
        provider = OpenClawHermesMemoryProvider(governor=governor)
        memory_context = ""
        if message:
            memory_context = governor.prefetch(message, domain=domain)
        return OpenClawRuntimeBundle(
            provider=provider,
            governor=governor,
            system_prompt_block=governor.system_prompt_block(),
            tool_schemas=governor.get_tool_schemas(),
            memory_context=memory_context,
            session=governor._sync_session_snapshot(),
            bridge_sync=governor.last_bridge_sync,
        )

    def attach_to_context(self, ctx: Any) -> dict[str, Any]:
        provider = self.provider
        tools = self.get_tool_schemas()
        prompt = self.system_prompt_block()
        attached: list[str] = []
        if hasattr(ctx, "register_memory_provider"):
            ctx.register_memory_provider(provider)
            attached.append("register_memory_provider")
        if hasattr(ctx, "memory_provider"):
            ctx.memory_provider = provider
            attached.append("memory_provider")
        if hasattr(ctx, "register_tools"):
            ctx.register_tools(tools)
            attached.append("register_tools")
        if hasattr(ctx, "tool_schemas"):
            ctx.tool_schemas = tools
            attached.append("tool_schemas")
        if hasattr(ctx, "tool_call"):
            ctx.tool_call = self.handle_tool_call
            attached.append("tool_call")
        if hasattr(ctx, "system_prompt_block"):
            ctx.system_prompt_block = lambda: prompt
            attached.append("system_prompt_block")
        if hasattr(ctx, "memory_runtime"):
            ctx.memory_runtime = self
            attached.append("memory_runtime")
        if hasattr(ctx, "openclaw_memory"):
            ctx.openclaw_memory = self
            attached.append("openclaw_memory")
        if not attached:
            raise AttributeError("Context does not expose any recognized OpenClaw memory integration hooks.")
        return {
            "provider": provider.name,
            "runtime": self.name,
            "attached": attached,
            "tool_names": [schema["name"] for schema in tools],
            "system_prompt_block": prompt,
            "storage_root": str(self._storage_path()),
        }

    def shutdown(self) -> None:
        governor = self._governor
        if governor is not None:
            governor.shutdown()

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "OpenClawRuntimeAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
