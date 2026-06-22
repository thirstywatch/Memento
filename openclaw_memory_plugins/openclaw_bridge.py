from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory_store import MemoryStore
from .types import MemoryRecord, utc_now_iso

_BRIDGE_BEGIN = "<!-- OPENCLAW_MEMORY_BRIDGE_BEGIN -->"
_BRIDGE_END = "<!-- OPENCLAW_MEMORY_BRIDGE_END -->"


@dataclass(slots=True)
class BridgeSource:
    path: Path
    domain: str
    kind: str
    label: str


class OpenClawMemoryBridge:
    """Bridge OpenClaw markdown memory files to the plugin store.

    The plugin store stays the source of truth. This bridge imports the host's
    markdown memory surfaces as warm-start context and writes back a compact
    generated digest into the host workspace so OpenClaw can see the plugin's
    durable state on the next turn.
    """

    def __init__(
        self,
        *,
        openclaw_home: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        self_improving_dir: str | Path | None = None,
        proactivity_dir: str | Path | None = None,
        bridge_filename: str = "openclaw-memory-bridge.md",
    ) -> None:
        self.openclaw_home = Path(openclaw_home).expanduser() if openclaw_home else None
        self.workspace_dir = Path(workspace_dir).expanduser() if workspace_dir else None
        self.self_improving_dir = Path(self_improving_dir).expanduser() if self_improving_dir else None
        self.proactivity_dir = Path(proactivity_dir).expanduser() if proactivity_dir else None
        self.bridge_filename = bridge_filename

    @classmethod
    def from_env(cls) -> "OpenClawMemoryBridge":
        return cls(
            openclaw_home=os.environ.get("OPENCLAW_HOME"),
            workspace_dir=os.environ.get("OPENCLAW_WORKSPACE_DIR"),
            self_improving_dir=os.environ.get("OPENCLAW_SELF_IMPROVING_DIR") or (Path.home() / "self-improving"),
            proactivity_dir=os.environ.get("OPENCLAW_PROACTIVITY_DIR") or (Path.home() / "proactivity"),
        )

    def _root(self) -> Path | None:
        if self.openclaw_home is not None:
            return self.openclaw_home
        env_root = os.environ.get("OPENCLAW_HOME")
        if env_root:
            return Path(env_root).expanduser()
        return None

    def _workspace(self) -> Path | None:
        if self.workspace_dir is not None:
            return self.workspace_dir
        root = self._root()
        if root is not None:
            return root / "workspace"
        env_workspace = os.environ.get("OPENCLAW_WORKSPACE_DIR")
        if env_workspace:
            return Path(env_workspace).expanduser()
        return None

    def _self_improving(self) -> Path | None:
        if self.self_improving_dir is not None:
            return self.self_improving_dir
        env_path = os.environ.get("OPENCLAW_SELF_IMPROVING_DIR")
        if env_path:
            return Path(env_path).expanduser()
        return Path.home() / "self-improving"

    def _proactivity(self) -> Path | None:
        if self.proactivity_dir is not None:
            return self.proactivity_dir
        env_path = os.environ.get("OPENCLAW_PROACTIVITY_DIR")
        if env_path:
            return Path(env_path).expanduser()
        return Path.home() / "proactivity"

    @staticmethod
    def _stable_id(seed: str) -> str:
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"bridge_{digest}"

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _compact_markdown(text: str, *, max_lines: int = 18, max_chars: int = 1200) -> str:
        lines: list[str] = []
        in_code = False
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped == _BRIDGE_BEGIN or stripped == _BRIDGE_END:
                continue
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped.startswith("|"):
                continue
            if len(stripped) > 220:
                stripped = stripped[:217] + "..."
            lines.append(stripped)
            if len(lines) >= max_lines:
                break
        compact = "\n".join(lines).strip()
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3] + "..."
        return compact

    @staticmethod
    def _strip_bridge_block(text: str) -> str:
        lines = text.splitlines()
        output: list[str] = []
        skipping = False
        for line in lines:
            stripped = line.strip()
            if stripped == _BRIDGE_BEGIN:
                skipping = True
                continue
            if stripped == _BRIDGE_END:
                skipping = False
                continue
            if not skipping:
                output.append(line)
        return "\n".join(output).strip()

    @staticmethod
    def _first_heading(text: str, default: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
        return default

    def _source_specs(self) -> list[BridgeSource]:
        specs: list[BridgeSource] = []
        root = self._root()
        workspace = self._workspace()
        # Prefer the live workspace surface first so the bridge follows the real OpenClaw profile.
        if workspace is not None:
            specs.extend([
                BridgeSource(workspace / "AGENTS.md", "agent", "note", "Workspace AGENTS"),
                BridgeSource(workspace / "SOUL.md", "agent", "pattern", "Workspace SOUL"),
                BridgeSource(workspace / "HEARTBEAT.md", "agent", "note", "Workspace HEARTBEAT"),
                BridgeSource(workspace / "MEMORY.md", "project", "fact", "Workspace MEMORY"),
            ])
        elif root is not None:
            specs.extend([
                BridgeSource(root / "AGENTS.md", "agent", "note", "OpenClaw AGENTS"),
                BridgeSource(root / "SOUL.md", "agent", "pattern", "OpenClaw SOUL"),
                BridgeSource(root / "HEARTBEAT.md", "agent", "note", "OpenClaw HEARTBEAT"),
                BridgeSource(root / "workspace" / "MEMORY.md", "project", "fact", "Workspace MEMORY"),
            ])
        self_improving = self._self_improving()
        if self_improving is not None:
            specs.extend([
                BridgeSource(self_improving / "memory.md", "agent", "pattern", "Self-improving memory"),
                BridgeSource(self_improving / "corrections.md", "agent", "failure", "Self-improving corrections"),
                BridgeSource(self_improving / "heartbeat-state.md", "agent", "note", "Self-improving heartbeat state"),
            ])
        proactivity = self._proactivity()
        if proactivity is not None:
            specs.extend([
                BridgeSource(proactivity / "memory.md", "agent", "pattern", "Proactivity memory"),
                BridgeSource(proactivity / "session-state.md", "project", "note", "Proactivity session state"),
                BridgeSource(proactivity / "heartbeat.md", "agent", "note", "Proactivity heartbeat"),
                BridgeSource(proactivity / "patterns.md", "agent", "pattern", "Proactivity patterns"),
                BridgeSource(proactivity / "log.md", "agent", "note", "Proactivity log"),
            ])
        return specs

    def discover_sources(self) -> list[BridgeSource]:
        return [spec for spec in self._source_specs() if spec.path.exists() and spec.path.is_file()]

    def _record_from_source(self, source: BridgeSource) -> MemoryRecord | None:
        try:
            raw = source.path.read_text(encoding="utf-8")
        except OSError:
            return None
        stripped = self._strip_bridge_block(raw)
        compact = self._compact_markdown(stripped)
        if not compact:
            return None
        title = self._first_heading(stripped, source.label)
        seed = f"{source.path.as_posix()}::{self._text_hash(stripped)}"
        metadata = {
            "bridge": True,
            "source_path": str(source.path),
            "source_label": source.label,
            "content_hash": self._text_hash(stripped),
            "imported_at": utc_now_iso(),
        }
        return MemoryRecord(
            id=self._stable_id(seed),
            domain=source.domain,  # type: ignore[arg-type]
            kind=source.kind,  # type: ignore[arg-type]
            content=f"{title}: {compact}" if title and title not in compact else compact,
            confidence=0.76 if source.kind in {"fact", "decision", "preference"} else 0.68,
            source="openclaw_markdown",
            tags=["openclaw", "bridge", source.path.stem.lower()],
            metadata=metadata,
        )

    def import_surface(self, store: MemoryStore) -> dict[str, Any]:
        imported: list[dict[str, Any]] = []
        skipped: list[str] = []
        sources = self.discover_sources()
        for source in sources:
            record = self._record_from_source(source)
            if record is None:
                skipped.append(str(source.path))
                continue
            stored = store.upsert_record(record)
            imported.append(
                {
                    "path": str(source.path),
                    "record_id": stored.id,
                    "domain": stored.domain,
                    "kind": stored.kind,
                }
            )
        return {
            "imported": imported,
            "skipped": skipped,
            "source_count": len(sources),
        }

    def _top_records(self, store: MemoryStore, *, domain: str, limit: int) -> list[MemoryRecord]:
        records = store.list_records(domain=domain, status="active", limit=None)
        records.sort(
            key=lambda record: (
                float(record.metadata.get("trust", record.confidence)) if isinstance(record.metadata, dict) else record.confidence,
                record.updated_at,
                record.confidence,
            ),
            reverse=True,
        )
        return records[:limit]

    def build_snapshot(self, store: MemoryStore, *, session_snapshot: dict[str, Any] | None = None) -> str:
        lines = [
            _BRIDGE_BEGIN,
            "# OpenClaw Memory Bridge",
            f"Generated: {utc_now_iso()}",
            "",
            "This block is generated by the OpenClawMemoryBridge. It is safe to keep in",
            "OpenClaw's workspace memory because the bridge strips it back out on import.",
            "",
        ]

        for domain, heading in (("user", "## User Memory"), ("project", "## Project Memory"), ("agent", "## Agent Memory")):
            records = self._top_records(store, domain=domain, limit=8)
            if not records:
                continue
            lines.append(heading)
            for record in records:
                lines.append(f"- [{record.kind}] {record.content}")
            lines.append("")

        if session_snapshot:
            lines.append("## Session State")
            for key in ("session_id", "parent_session_id", "turn_number", "agent_context", "agent_identity", "agent_workspace", "platform", "user_id"):
                value = session_snapshot.get(key)
                if value not in (None, ""):
                    lines.append(f"- {key}: {value}")
            counts = session_snapshot.get("counts") or {}
            if isinstance(counts, dict) and counts:
                lines.append(f"- counts: {json.dumps(counts, ensure_ascii=False, sort_keys=True)}")
            lines.append("")

        lines.append(f"{_BRIDGE_END}")
        return "\n".join(lines).rstrip() + "\n"

    def export_snapshot(self, store: MemoryStore, *, session_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        workspace = self._workspace()
        if workspace is None:
            return {"skipped": True, "reason": "workspace_dir_missing", "exported": []}

        bridge_dir = workspace / "memory"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        bridge_path = bridge_dir / self.bridge_filename
        bridge_snapshot = self.build_snapshot(store, session_snapshot=session_snapshot)
        bridge_path.write_text(bridge_snapshot, encoding="utf-8")

        memory_path = workspace / "MEMORY.md"
        existing = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
        existing = self._strip_bridge_block(existing)
        merged = existing.rstrip()
        if merged:
            merged += "\n\n"
        merged += bridge_snapshot
        tmp_memory_path = memory_path.with_suffix(".md.tmp")
        tmp_memory_path.write_text(merged, encoding="utf-8")
        tmp_memory_path.replace(memory_path)
        return {
            "skipped": False,
            "exported": [str(bridge_path), str(memory_path)],
        }

    def sync(self, store: MemoryStore, *, session_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        imported = self.import_surface(store)
        exported = self.export_snapshot(store, session_snapshot=session_snapshot)
        return {
            "imported": imported,
            "exported": exported,
        }
