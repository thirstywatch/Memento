from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

MemoryDomain = Literal["user", "project", "agent"]
MemoryKind = Literal["preference", "fact", "decision", "pattern", "failure", "skill", "note"]
MemoryStatus = Literal["active", "staged", "superseded", "removed", "disputed"]
AdjudicationAction = Literal["write", "stage", "supersede", "reject"]

CONSERVATIVE_KINDS: frozenset[MemoryKind] = frozenset({"fact", "preference"})
AGGRESSIVE_MERGE_KINDS: frozenset[MemoryKind] = frozenset({"skill", "pattern"})

ADJUDICATION_CONFIDENCE_BOOST: float = 0.08
ADJUDICATION_TRUST_BOOST: float = 0.10


@dataclass(slots=True)
class AdjudicationRule:
    name: str
    description: str
    priority: int
    left_kinds: frozenset[MemoryKind] | None = None
    right_kinds: frozenset[MemoryKind] | None = None
    recommended_action: AdjudicationAction = "write"
    demote_old: bool = False
    merge_content: bool = False


ADJUDICATION_RULES: list[AdjudicationRule] = [
    AdjudicationRule(
        name="conservative-preserve",
        description="Keep conservative facts/preferences staged rather than auto-replacing history.",
        priority=100,
        left_kinds=CONSERVATIVE_KINDS,
        right_kinds=CONSERVATIVE_KINDS,
        recommended_action="stage",
    ),
    AdjudicationRule(
        name="decision-replacement",
        description="Decision-to-decision conflicts should replace the older decision.",
        priority=95,
        left_kinds={"decision"},
        right_kinds={"decision"},
        recommended_action="supersede",
        demote_old=True,
    ),
    AdjudicationRule(
        name="decision-over-fact",
        description="A decision can supersede a conflicting fact when the decision is the stronger signal.",
        priority=93,
        left_kinds={"decision"},
        right_kinds={"fact"},
        recommended_action="supersede",
        demote_old=True,
    ),
    AdjudicationRule(
        name="fact-against-decision",
        description="A conflicting fact should be staged when it collides with an existing decision.",
        priority=92,
        left_kinds={"fact"},
        right_kinds={"decision"},
        recommended_action="stage",
    ),
    AdjudicationRule(
        name="aggressive-replace",
        description="Skill and pattern records can overwrite each other.",
        priority=90,
        left_kinds=AGGRESSIVE_MERGE_KINDS,
        right_kinds=AGGRESSIVE_MERGE_KINDS,
        recommended_action="supersede",
        demote_old=True,
        merge_content=True,
    ),
    AdjudicationRule(
        name="skilled-over-raw",
        description="A skill or pattern can replace a weaker note-like record.",
        priority=80,
        left_kinds=AGGRESSIVE_MERGE_KINDS,
        right_kinds=None,
        recommended_action="supersede",
        demote_old=True,
    ),
    AdjudicationRule(
        name="preference-dominates-fact",
        description="A preference can supersede a conflicting fact when the preference is the stronger signal.",
        priority=70,
        left_kinds={"preference"},
        right_kinds={"fact"},
        recommended_action="supersede",
        demote_old=True,
    ),
]


@dataclass(slots=True)
class ConflictRelation:
    candidate_id: str
    existing_id: str
    score: float
    entity_overlap: float
    content_similarity: float
    polarity_left: str
    polarity_right: str
    shared_entities: list[str]
    rule_applied: str = ""
    suggested_action: AdjudicationAction = "write"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "existing_id": self.existing_id,
            "left": {"id": self.candidate_id},
            "right": {"id": self.existing_id},
            "score": round(self.score, 3),
            "entity_overlap": round(self.entity_overlap, 3),
            "content_similarity": round(self.content_similarity, 3),
            "polarity": [self.polarity_left, self.polarity_right],
            "shared_entities": list(self.shared_entities),
            "rule_applied": self.rule_applied,
            "suggested_action": self.suggested_action,
        }


@dataclass(slots=True)
class ConsistencyEntry:
    category: str
    topic: str
    left_id: str
    right_id: str
    resolution: str
    adjudication: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "topic": self.topic,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "resolution": self.resolution,
            "adjudication": self.adjudication,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_memory_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple | set):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class MemoryRecord:
    id: str
    domain: MemoryDomain
    kind: MemoryKind
    content: str
    confidence: float = 0.5
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    status: MemoryStatus = "active"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    supersedes_ids: list[str] = field(default_factory=list)
    contradicts_ids: list[str] = field(default_factory=list)
    adjudication: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if self.supersedes_ids:
            payload["supersedes_ids"] = list(self.supersedes_ids)
        if self.contradicts_ids:
            payload["contradicts_ids"] = list(self.contradicts_ids)
        if self.adjudication:
            payload["adjudication"] = self.adjudication
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        metadata = _as_dict(data.get("metadata"))
        return cls(
            id=str(data.get("id") or new_memory_id()),
            domain=str(data.get("domain") or "project"),
            kind=str(data.get("kind") or "note"),
            content=str(data.get("content") or "").strip(),
            confidence=float(data.get("confidence", 0.5)),
            source=str(data.get("source") or "conversation"),
            tags=_as_str_list(data.get("tags")),
            status=str(data.get("status") or "active"),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            metadata=metadata,
            supersedes_ids=_as_str_list(data.get("supersedes_ids")),
            contradicts_ids=_as_str_list(data.get("contradicts_ids")),
            adjudication=str(data.get("adjudication") or ""),
        )

    def touch(self) -> None:
        self.updated_at = utc_now_iso()
