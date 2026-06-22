from __future__ import annotations



from pathlib import Path

from typing import Any



from .memory_governor import OpenClawMemoryGovernor

from .memory_store import MemoryStore



try:  # Hermes runtime provides the real MemoryProvider ABC.

    from agent.memory_provider import MemoryProvider as HermesMemoryProvider

except Exception:  # pragma: no cover - local OpenClaw-only runs do not ship Hermes.

    class HermesMemoryProvider:  # type: ignore[too-many-ancestors]

        pass





class OpenClawHermesMemoryProvider(HermesMemoryProvider):

    """Hermes-compatible wrapper around OpenClawMemoryGovernor.



    The governor stays the concrete implementation. This adapter is the thin

    compatibility layer that Hermes' plugin loader can discover and register.

    """



    def __init__(
        self,
        governor: OpenClawMemoryGovernor | None = None,
        *,
        storage_root: str | Path | None = None,
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


    @property

    def name(self) -> str:

        return "memento"



    def _ensure_governor(self) -> OpenClawMemoryGovernor:

        if self._governor is None:
            store = MemoryStore(self._storage_root) if self._storage_root else None
            self._governor = OpenClawMemoryGovernor(
                store=store,
                auto_extract=self._auto_extract,
                auto_mirror=self._auto_mirror,
                openclaw_home=self._openclaw_home,
                workspace_dir=self._workspace_dir,
                self_improving_dir=self._self_improving_dir,
                proactivity_dir=self._proactivity_dir,
            )
        return self._governor


    def _storage_path(self) -> Path:

        governor = self._ensure_governor()

        return Path(governor.workflow.store.root_dir).expanduser()



    def is_available(self) -> bool:

        return True



    def initialize(self, session_id: str, **kwargs: Any) -> None:

        storage_root = kwargs.pop("storage_root", None) or kwargs.pop("memory_root", None)
        openclaw_home = kwargs.pop("openclaw_home", None) or kwargs.pop("hermes_home", None) or self._openclaw_home
        workspace_dir = kwargs.pop("workspace_dir", None) or self._workspace_dir
        self_improving_dir = kwargs.pop("self_improving_dir", None) or self._self_improving_dir
        proactivity_dir = kwargs.pop("proactivity_dir", None) or self._proactivity_dir
        if openclaw_home is not None:
            self._openclaw_home = Path(openclaw_home).expanduser()
        if workspace_dir is not None:
            self._workspace_dir = Path(workspace_dir).expanduser()
        if self_improving_dir is not None:
            self._self_improving_dir = Path(self_improving_dir).expanduser()
        if proactivity_dir is not None:
            self._proactivity_dir = Path(proactivity_dir).expanduser()
        if storage_root:
            root_path = Path(storage_root).expanduser()
            if self._governor is None or root_path != self._storage_path():
                self._storage_root = root_path
                self._governor = OpenClawMemoryGovernor(
                    store=MemoryStore(root_path),
                    auto_extract=self._auto_extract,
                    auto_mirror=self._auto_mirror,
                    openclaw_home=self._openclaw_home,
                    workspace_dir=self._workspace_dir,
                    self_improving_dir=self._self_improving_dir,
                    proactivity_dir=self._proactivity_dir,
                )
        governor = self._ensure_governor()
        governor.initialize(session_id, **kwargs)


    def system_prompt_block(self) -> str:

        return self._ensure_governor().system_prompt_block()



    def prefetch(self, query: str, *, session_id: str = "") -> str:

        governor = self._ensure_governor()

        if session_id:

            governor.session_id = session_id

        return governor.prefetch(query)



    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:

        governor = self._ensure_governor()

        if session_id:

            governor.session_id = session_id

        governor.queue_prefetch(query)



    def sync_turn(

        self,

        user_content: str,

        assistant_content: str,

        *,

        session_id: str = "",

        messages: list[dict[str, Any]] | None = None,

    ) -> None:

        governor = self._ensure_governor()

        if session_id:

            governor.session_id = session_id

        governor.sync_turn(user_content, assistant_content, messages=messages)



    def reflect(

        self,

        *,

        task_title: str,

        result_summary: str,

        lessons: str | None = None,

        skill_steps: list[str] | None = None,

        domain: str | None = None,

        ingest_skill_candidate: bool = True,

    ) -> dict[str, Any]:

        outcome = self._ensure_governor().reflect(

            task_title=task_title,

            result_summary=result_summary,

            lessons=lessons,

            skill_steps=skill_steps,

            domain=domain,

            ingest_skill_candidate=ingest_skill_candidate,

        )

        return outcome.to_dict()



    def run_cycle(

        self,

        *,

        query: str,

        candidate: Any | None = None,

        task_title: str | None = None,

        result_summary: str | None = None,

        lessons: str | None = None,

        skill_steps: list[str] | None = None,

        domain: str | None = None,

        limit: int = 5,

        max_chars: int = 800,

    ) -> dict[str, Any]:

        packet = self._ensure_governor().run_cycle(

            query=query,

            candidate=candidate,

            task_title=task_title,

            result_summary=result_summary,

            lessons=lessons,

            skill_steps=skill_steps,

            domain=domain,

            limit=limit,

            max_chars=max_chars,

        )

        return packet.to_dict()



    def get_tool_schemas(self) -> list[dict[str, Any]]:

        return self._ensure_governor().get_tool_schemas()



    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:

        return self._ensure_governor().handle_tool_call(tool_name, args, **kwargs)



    def shutdown(self) -> None:

        governor = self._governor

        if governor is not None:

            governor.shutdown()



    def close(self) -> None:

        self.shutdown()



    def __enter__(self) -> "OpenClawHermesMemoryProvider":

        return self



    def __exit__(self, exc_type, exc, tb) -> None:

        self.shutdown()



    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:

        self._ensure_governor().on_turn_start(turn_number, message, **kwargs)



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

    ) -> None:

        self._ensure_governor().on_session_switch(

            new_session_id,

            parent_session_id=parent_session_id,

            reset=reset,

            rewound=rewound,

            **kwargs,

        )



    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:

        return self._ensure_governor().on_pre_compress(messages)



    def on_delegation(

        self,

        task: str,

        result: str,

        *,

        child_session_id: str = "",

        **kwargs: Any,

    ) -> None:

        self._ensure_governor().on_delegation(

            task,

            result,

            child_session_id=child_session_id,

            **kwargs,

        )



    def on_memory_write(

        self,

        action: str,

        target: str,

        content: str,

        metadata: dict[str, Any] | None = None,

    ) -> None:

        self._ensure_governor().on_memory_write(action, target, content, metadata)



    def get_config_schema(self) -> list[dict[str, Any]]:

        return []



    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:

        return None



    def backup_paths(self) -> list[str]:

        path = self._storage_path()

        return [str(path)] if path.exists() else []





def register(ctx: Any) -> None:

    """Hermes plugin registration entrypoint."""



    ctx.register_memory_provider(OpenClawHermesMemoryProvider())

