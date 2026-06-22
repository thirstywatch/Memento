from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hermes_provider import OpenClawHermesMemoryProvider
from .memory_governor import OpenClawMemoryGovernor
from .memory_store import MemoryStore
from .openclaw_adapter import OpenClawRuntimeAdapter

_DEFAULT_STORAGE_ROOT = Path("~/.memento").expanduser()
_RUNTIME: OpenClawRuntimeAdapter | None = None


def build_governor(
    *,
    storage_root: str | Path | None = None,
    session_id: str = "",
    platform: str = "cli",
    auto_extract: bool = True,
    auto_mirror: bool = True,
    openclaw_home: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    self_improving_dir: str | Path | None = None,
    proactivity_dir: str | Path | None = None,
) -> OpenClawMemoryGovernor:
    store = MemoryStore(storage_root) if storage_root is not None else MemoryStore(_DEFAULT_STORAGE_ROOT)
    return OpenClawMemoryGovernor(
        store=store,
        session_id=session_id,
        platform=platform,
        auto_extract=auto_extract,
        auto_mirror=auto_mirror,
        openclaw_home=openclaw_home,
        workspace_dir=workspace_dir,
        self_improving_dir=self_improving_dir,
        proactivity_dir=proactivity_dir,
    )


def build_runtime(
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
) -> OpenClawRuntimeAdapter:
    governor = build_governor(
        storage_root=storage_root,
        session_id=session_id,
        platform=platform,
        auto_extract=auto_extract,
        auto_mirror=auto_mirror,
        openclaw_home=openclaw_home,
        workspace_dir=workspace_dir,
        self_improving_dir=self_improving_dir,
        proactivity_dir=proactivity_dir,
    )
    return OpenClawRuntimeAdapter(governor=governor)


def build_provider(
    *,
    storage_root: str | Path | None = None,
    session_id: str = "",
    platform: str = "cli",
    auto_extract: bool = True,
    auto_mirror: bool = True,
    openclaw_home: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    self_improving_dir: str | Path | None = None,
    proactivity_dir: str | Path | None = None,
) -> OpenClawHermesMemoryProvider:
    runtime = build_runtime(
        storage_root=storage_root,
        session_id=session_id,
        platform=platform,
        auto_extract=auto_extract,
        auto_mirror=auto_mirror,
        openclaw_home=openclaw_home,
        workspace_dir=workspace_dir,
        self_improving_dir=self_improving_dir,
        proactivity_dir=proactivity_dir,
    )
    return runtime.provider


def get_runtime() -> OpenClawRuntimeAdapter:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = build_runtime()
    return _RUNTIME


def get_provider() -> OpenClawHermesMemoryProvider:
    return get_runtime().provider


def get_tool_schemas() -> list[dict[str, Any]]:
    return get_runtime().get_tool_schemas()


def handle_tool_call(tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
    return get_runtime().handle_tool_call(tool_name, args, **kwargs)


def register(ctx: Any) -> dict[str, Any]:
    """OpenClaw entrypoint for the memory plugin pack."""

    runtime = get_runtime()
    result = runtime.attach_to_context(ctx)
    result["runtime"] = runtime.name
    return result


def bootstrap() -> dict[str, Any]:
    """Return the runtime bundle for hosts that prefer explicit loading."""

    runtime = get_runtime()
    return {
        "runtime": runtime,
        "provider": runtime.provider,
        "governor": runtime.governor,
        "tool_schemas": runtime.get_tool_schemas(),
        "tool_call": handle_tool_call,
        "system_prompt_block": runtime.system_prompt_block(),
    }
