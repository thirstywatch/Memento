from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

MemoryDomain = Literal["user", "project", "agent"]
MemoryKind = Literal["preference", "fact", "decision", "pattern", "failure", "skill", "note"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_memory_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class MemoryRecord:
    id: str
    domain: MemoryDomain
    kind: MemoryKind
    content: str
    confidence: float = 0.5
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "kind": self.kind,
            "content": self.content,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "tags": list(self.tags),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=str(data.get("id") or new_memory_id()),
            domain=data.get("domain", "project"),
            kind=data.get("kind", "note"),
            content=str(data.get("content", "")).strip(),
            confidence=float(data.get("confidence", 0.5)),
            source=str(data.get("source", "conversation")),
            tags=list(data.get("tags", [])),
            status=str(data.get("status", "active")),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            metadata=dict(data.get("metadata", {})),
        )

    def touch(self) -> None:
        self.updated_at = utc_now_iso()
