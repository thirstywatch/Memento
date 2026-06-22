from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .memory_reflect import MemoryReflector, ReflectionBundle
from .memory_retrieve import MemoryRetriever, RetrievedMemory
from .memory_score import MemoryScorer, ScoreDecision
from .memory_store import MemoryStore
from .memory_workflow import IngestOutcome, MemoryWorkflow, RecallOutcome, ReflectionOutcome
from .openclaw_bridge import OpenClawMemoryBridge
from .types import MemoryDomain, MemoryKind, MemoryRecord, new_memory_id, utc_now_iso

_VALID_ROLES = {"system", "user", "assistant", "tool"}
_PREF_PATTERNS = (
    re.compile(r"\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)", re.I),
    re.compile(r"\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)", re.I),
    re.compile(r"\bI\s+(?:always|never|usually)\s+(.+)", re.I),
)
_DECISION_PATTERNS = (
    re.compile(r"\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)", re.I),
    re.compile(r"\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)", re.I),
    re.compile(r"\bwe\s+should\s+(.+)", re.I),
)
_FAILURE_PATTERNS = (re.compile(r"\b(?:failed|blocked|error|issue|bug)\b", re.I),)
_POSITIVE_CLAIM_PATTERNS = (
    re.compile(r"\b(?:prefer|like|love|use|want|need|always|must|should|enabled|true|can)\b", re.I),
)
_NEGATIVE_CLAIM_PATTERNS = (
    re.compile(r"\b(?:avoid|dislike|hate|never|cannot|can't|won't|disabled|false|shouldn't|mustn't|no|not)\b", re.I),
)


@dataclass(slots=True)
class GovernorPacket:
    recall: RecallOutcome
    memory_context: str
    ingest: IngestOutcome | None = None
    reflection: ReflectionOutcome | None = None
    session: dict[str, Any] = field(default_factory=dict)
    extracted: list[dict[str, Any]] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    pre_compress: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall.to_dict(),
            "memory_context": self.memory_context,
            "ingest": self.ingest.to_dict() if self.ingest else None,
            "reflection": self.reflection.to_dict() if self.reflection else None,
            "session": dict(self.session),
            "extracted": list(self.extracted),
            "writes": list(self.writes),
            "pre_compress": self.pre_compress,
        }


class OpenClawMemoryGovernor:
    """Lifecycle adapter that gives the plugin pack Hermes-style governance."""

    def __init__(
        self,
        *,
        workflow: MemoryWorkflow | None = None,
        store: MemoryStore | None = None,
        scorer: MemoryScorer | None = None,
        retriever: MemoryRetriever | None = None,
        reflector: MemoryReflector | None = None,
        default_domain: MemoryDomain = "project",
        session_id: str = "",
        parent_session_id: str = "",
        agent_context: str = "primary",
        agent_identity: str = "",
        agent_workspace: str = "",
        platform: str = "cli",
        user_id: str = "",
        auto_extract: bool = True,
        auto_mirror: bool = True,
        bridge: OpenClawMemoryBridge | None = None,
        openclaw_home: str | Any | None = None,
        workspace_dir: str | Any | None = None,
        self_improving_dir: str | Any | None = None,
        proactivity_dir: str | Any | None = None,
    ) -> None:
        self.workflow = workflow or MemoryWorkflow(
            store=store,
            scorer=scorer,
            retriever=retriever,
            reflector=reflector,
        )
        self.default_domain = default_domain
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.agent_context = agent_context
        self.agent_identity = agent_identity
        self.agent_workspace = agent_workspace
        self.platform = platform
        self.user_id = user_id
        self.auto_extract = auto_extract
        self.auto_mirror = auto_mirror
        self.bridge = bridge or OpenClawMemoryBridge(
            openclaw_home=openclaw_home,
            workspace_dir=workspace_dir,
            self_improving_dir=self_improving_dir,
            proactivity_dir=proactivity_dir,
        )
        self.last_bridge_sync: dict[str, Any] | None = None
        self.last_recall: RecallOutcome | None = None
        self.last_ingest: IngestOutcome | None = None
        self.last_reflection: ReflectionOutcome | None = None
        self.last_packet: GovernorPacket | None = None
        self.last_write: dict[str, Any] | None = None
        self.last_pre_compress: str = ""
        self.last_extracted: list[dict[str, Any]] = []
        self.last_session_end: dict[str, Any] | None = None
        self.turn_number: int = 0
        self.last_turn_user: str = ""
        self.last_turn_assistant: str = ""
        self._prefetch_cache_key: tuple[str, str | None, int, int] | None = None
        self._prefetch_cache_value: str = ""
        self._prefetch_cache_recall: RecallOutcome | None = None
        self._queued_prefetch: tuple[str, str | None, int, int] | None = None
        self._queued_prefetch_value: str = ""

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self.session_id = session_id
        self.parent_session_id = str(kwargs.get("parent_session_id") or self.parent_session_id)
        self.agent_context = str(kwargs.get("agent_context") or self.agent_context)
        self.agent_identity = str(kwargs.get("agent_identity") or self.agent_identity)
        self.agent_workspace = str(kwargs.get("agent_workspace") or self.agent_workspace)
        self.platform = str(kwargs.get("platform") or self.platform)
        self.user_id = str(kwargs.get("user_id") or self.user_id)
        if bool(kwargs.pop("sync_openclaw_memory", True)):
            self.sync_openclaw_memory(import_surface=True, export_surface=False, reason="initialize")

    def shutdown(self) -> None:
        self._queued_prefetch = None
        self._queued_prefetch_value = ""
        try:
            self.sync_openclaw_memory(import_surface=False, export_surface=True, reason="shutdown")
        finally:
            if hasattr(self.workflow.store, "close"):
                self.workflow.store.close()

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "OpenClawMemoryGovernor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def sync_openclaw_memory(
        self,
        *,
        import_surface: bool = True,
        export_surface: bool = False,
        reason: str = "manual",
    ) -> dict[str, Any]:
        if self.bridge is None:
            payload = {"skipped": True, "reason": "bridge_unavailable", "imported": None, "exported": None}
            self.last_bridge_sync = payload
            return payload
        snapshot = self._sync_session_snapshot()
        payload: dict[str, Any] = {"skipped": False, "reason": reason}
        payload["imported"] = self.bridge.import_surface(self.workflow.store) if import_surface else None
        payload["exported"] = self.bridge.export_snapshot(self.workflow.store, session_snapshot=snapshot) if export_surface else None
        self.last_bridge_sync = payload
        return payload

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            if "text" in value:
                return str(value.get("text") or "").strip()
            if "content" in value:
                return str(value.get("content") or "").strip()
        return str(value).strip()

    @staticmethod
    def _safe_role(message: dict[str, Any]) -> str:
        role = str(message.get("role") or "").strip().lower()
        return role if role in _VALID_ROLES else "user"

    @staticmethod
    def _infer_kind(text: str, role: str) -> MemoryKind:
        lowered = text.lower()
        if role == "user":
            if any(pattern.search(text) for pattern in _PREF_PATTERNS):
                return "preference"
            if any(pattern.search(text) for pattern in _FAILURE_PATTERNS):
                return "failure"
            return "note"
        if any(pattern.search(text) for pattern in _FAILURE_PATTERNS):
            return "failure"
        if any(pattern.search(text) for pattern in _DECISION_PATTERNS):
            return "decision"
        if "skill" in lowered or "pattern" in lowered:
            return "skill"
        return "pattern"

    @staticmethod
    def _infer_domain(text: str, role: str, default_domain: MemoryDomain = "project") -> MemoryDomain:
        lowered = text.lower()
        if role == "user":
            if any(pattern.search(text) for pattern in _PREF_PATTERNS):
                return "user"
            if "my " in lowered or "i " in lowered:
                return "user"
            return default_domain
        if any(pattern.search(text) for pattern in _DECISION_PATTERNS):
            return "project"
        return "project" if default_domain != "user" else default_domain

    @staticmethod
    def _score_text(text: str, role: str) -> float:
        score = 0.4
        lowered = text.lower()
        if role == "user":
            score += 0.08
        if any(pattern.search(text) for pattern in _PREF_PATTERNS):
            score += 0.28
        if any(pattern.search(text) for pattern in _DECISION_PATTERNS):
            score += 0.18
        if any(pattern.search(text) for pattern in _FAILURE_PATTERNS):
            score += 0.16
        if len(lowered) > 120:
            score += 0.04
        if len(lowered) < 18:
            score -= 0.08
        return max(0.0, min(score, 1.0))

    def _candidate_from_message(self, message: dict[str, Any]) -> MemoryRecord | None:
        role = self._safe_role(message)
        content = self._normalize_text(message.get("content"))
        if not content or role == "tool":
            return None
        domain = self._infer_domain(content, role, self.default_domain)
        kind = self._infer_kind(content, role)
        score = self._score_text(content, role)
        metadata = {"source_role": role, "origin": "auto_extract", "turn_number": self.turn_number}
        if role == "assistant" and kind == "pattern" and len(content) > 220:
            kind = "fact"
        return MemoryRecord(
            id=new_memory_id(),
            domain=domain,
            kind=kind,
            content=content,
            confidence=score,
            source="conversation" if role == "user" else "task_result",
            tags=[f"role:{role}", "auto-extract"],
            metadata=metadata,
        )

    def _extract_candidates_from_messages(self, messages: Iterable[dict[str, Any]]) -> list[MemoryRecord]:
        candidates: list[MemoryRecord] = []
        seen: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            candidate = self._candidate_from_message(message)
            if candidate is None:
                continue
            key = candidate.content.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _ingest_candidates(
        self,
        candidates: Iterable[MemoryRecord],
        *,
        force_domain: MemoryDomain | None = None,
        source: str = "auto_extract",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            if force_domain is not None:
                candidate = MemoryRecord(
                    id=candidate.id,
                    domain=force_domain,
                    kind=candidate.kind,
                    content=candidate.content,
                    confidence=candidate.confidence,
                    source=candidate.source,
                    tags=list(candidate.tags),
                    status=candidate.status,
                    created_at=candidate.created_at,
                    updated_at=candidate.updated_at,
                    metadata=dict(candidate.metadata),
                )
            outcome = self.workflow.ingest(
                candidate,
                domain=force_domain,
                kind=candidate.kind,
                source=source,
                confidence=candidate.confidence,
                tags=candidate.tags,
                metadata=candidate.metadata,
            )
            self.last_ingest = outcome
            results.append(outcome.to_dict())
        return results

    def _sync_session_snapshot(self) -> dict[str, Any]:
        store = self.workflow.store
        return {
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "turn_number": self.turn_number,
            "agent_context": self.agent_context,
            "agent_identity": self.agent_identity,
            "agent_workspace": self.agent_workspace,
            "platform": self.platform,
            "user_id": self.user_id,
            "counts": {
                "records": store.count_records(status=None) if hasattr(store, "count_records") else len(store.list_records(status=None)),
                "active_records": store.count_records(status="active") if hasattr(store, "count_records") else len(store.list_records(status="active")),
                "pending": len(list(store._iter_jsonl(store.pending_path))) if hasattr(store, "_iter_jsonl") else 0,
                "reflections": len(list(store._iter_jsonl(store.reflections_path))) if hasattr(store, "_iter_jsonl") else 0,
            },
        }

    def _maybe_return_cached_prefetch(self, query: str, domain: MemoryDomain | None, limit: int, max_chars: int) -> str | None:
        cache_key = (query, domain, limit, max_chars)
        if self._queued_prefetch == cache_key and self._queued_prefetch_value:
            self._prefetch_cache_key = cache_key
            self._prefetch_cache_value = self._queued_prefetch_value
            self._queued_prefetch = None
            self._queued_prefetch_value = ""
            return self._prefetch_cache_value
        if self._prefetch_cache_key == cache_key:
            return self._prefetch_cache_value
        return None

    @staticmethod
    def build_memory_context(memory_pack: str) -> str:
        pack = (memory_pack or "").strip()
        if not pack:
            return ""
        return (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new user input. "
            "Treat as authoritative reference data. This is the agent's persistent memory.]\n\n"
            f"{pack}\n"
            "</memory-context>"
        )

    def system_prompt_block(self) -> str:
        snapshot = self._sync_session_snapshot()
        store = self.workflow.store
        user_count = store.count_records(domain="user", status="active") if hasattr(store, "count_records") else len(store.list_records(domain="user", status="active"))
        project_count = store.count_records(domain="project", status="active") if hasattr(store, "count_records") else len(store.list_records(domain="project", status="active"))
        agent_count = store.count_records(domain="agent", status="active") if hasattr(store, "count_records") else len(store.list_records(domain="agent", status="active"))
        return (
            "# OpenClaw Memory Governor\n"
            f"Session: {snapshot['session_id'] or 'unset'}\n"
            f"Mode: {snapshot['agent_context']} | Platform: {snapshot['platform']}\n"
            f"Active memory: user={user_count}, project={project_count}, agent={agent_count}\n"
            "Use memory_add to persist durable facts, memory_search to recall, "
            "memory_feedback to raise or lower trust, memory_contradict to spot conflicts, and memory_forget to remove stale entries.\n"
            "User preferences live in the user domain; project decisions and task state live in project."
        )

    def route_domain(self, candidate: MemoryRecord | dict[str, Any]) -> MemoryDomain:
        return self.workflow.route_domain(candidate)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Za-z0-9_\-]+", text)}

    def _entity_names(self, record: MemoryRecord) -> set[str]:
        store = self.workflow.store
        names: set[str] = set()
        if hasattr(store, "get_record_entities"):
            for entity in store.get_record_entities(record.id):
                normalized = str(entity.get("normalized") or "").strip().lower()
                if normalized:
                    names.add(normalized)
                name = str(entity.get("name") or "").strip().lower()
                if name:
                    names.add(name)
        if not names and hasattr(store, "_entity_extractor"):
            extractor = getattr(store, "_entity_extractor")
            if hasattr(extractor, "extract_normalized"):
                metadata_text = json.dumps(record.metadata, ensure_ascii=False, sort_keys=True)
                names.update(
                    str(item).strip().lower()
                    for item in extractor.extract_normalized([record.content, " ".join(record.tags), metadata_text])
                    if str(item).strip()
                )
        return names

    @classmethod
    def _claim_polarity(cls, text: str) -> str:
        if any(pattern.search(text) for pattern in _POSITIVE_CLAIM_PATTERNS):
            return "positive"
        if any(pattern.search(text) for pattern in _NEGATIVE_CLAIM_PATTERNS):
            return "negative"
        return "neutral"

    @classmethod
    def _content_similarity(cls, left: MemoryRecord, right: MemoryRecord) -> float:
        left_tokens = cls._token_set(" ".join([left.content, " ".join(left.tags)]))
        right_tokens = cls._token_set(" ".join([right.content, " ".join(right.tags)]))
        if not left_tokens or not right_tokens:
            return 0.0
        shared = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return shared / union if union else 0.0

    def contradict(
        self,
        query: str = "",
        *,
        domain: MemoryDomain | None = None,
        threshold: float = 0.28,
        limit: int = 5,
    ) -> dict[str, Any]:
        store = self.workflow.store
        seed_query = query.strip()
        pool: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        def _append(records: Iterable[MemoryRecord]) -> None:
            for record in records:
                if record.id in seen_ids:
                    continue
                seen_ids.add(record.id)
                pool.append(record)

        if seed_query:
            _append(store.find_matching_records(seed_query, domain=domain, limit=max(limit * 4, 12), status=None))
            if hasattr(store, "_entity_extractor"):
                entity_names = store._entity_extractor.extract_names(seed_query)
                if entity_names:
                    _append(store.find_records_for_entities(entity_names, domain=domain, limit=max(limit * 4, 12), status=None))
        else:
            _append(store.list_records(domain=domain, status=None, limit=max(limit * 6, 24)))

        contradictions: list[dict[str, Any]] = []
        for index, left in enumerate(pool[:500]):
            left_entities = self._entity_names(left)
            if not left_entities:
                continue
            for right in pool[index + 1 : 500]:
                right_entities = self._entity_names(right)
                if not right_entities:
                    continue
                shared = left_entities & right_entities
                union = left_entities | right_entities
                if not shared or not union:
                    continue
                entity_overlap = len(shared) / len(union)
                if entity_overlap < 0.25:
                    continue
                content_similarity = self._content_similarity(left, right)
                polarity_left = self._claim_polarity(left.content)
                polarity_right = self._claim_polarity(right.content)
                contradiction_score = entity_overlap * (1.0 - content_similarity)
                if polarity_left != "neutral" and polarity_right != "neutral" and polarity_left != polarity_right:
                    contradiction_score += 0.18
                if left.kind != right.kind and {left.kind, right.kind} & {"decision", "preference", "fact", "failure"}:
                    contradiction_score += 0.05
                if contradiction_score < threshold:
                    continue
                contradictions.append(
                    {
                        "score": round(min(1.0, contradiction_score), 3),
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_similarity, 3),
                        "polarity": [polarity_left, polarity_right],
                        "left": left.to_dict(),
                        "right": right.to_dict(),
                        "shared_entities": sorted(shared),
                    }
                )
        contradictions.sort(key=lambda item: (item["score"], item["entity_overlap"]), reverse=True)
        return {"query": seed_query, "domain": domain, "results": contradictions[:limit]}

    def prefetch(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        cached = self._maybe_return_cached_prefetch(query, domain, limit, max_chars)
        if cached is not None:
            if self._prefetch_cache_recall is not None:
                self.last_recall = self._prefetch_cache_recall
            return cached
        recall = self.workflow.retrieve(query, domain=domain, limit=limit, max_chars=max_chars)
        memory_context = self.build_memory_context(recall.memory_pack)
        self.last_recall = recall
        self._prefetch_cache_key = (query, domain, limit, max_chars)
        self._prefetch_cache_value = memory_context
        self._prefetch_cache_recall = recall
        return memory_context

    def queue_prefetch(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> None:
        self._queued_prefetch = (query, domain, limit, max_chars)
        self._queued_prefetch_value = self.prefetch(query, domain=domain, limit=limit, max_chars=max_chars)

    def recall(self, query: str, *, domain: MemoryDomain | None = None, limit: int = 5, max_chars: int = 800) -> RecallOutcome:
        recall = self.workflow.retrieve(query, domain=domain, limit=limit, max_chars=max_chars)
        self.last_recall = recall
        return recall

    def ingest(
        self,
        candidate: MemoryRecord | dict[str, Any] | str,
        *,
        domain: MemoryDomain | None = None,
        kind: MemoryKind = "note",
        source: str = "conversation",
        confidence: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestOutcome:
        outcome = self.workflow.ingest(
            candidate,
            domain=domain,
            kind=kind,
            source=source,
            confidence=confidence,
            tags=tags,
            metadata=metadata,
        )
        self.last_ingest = outcome
        return outcome

    def reflect(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        domain: MemoryDomain | None = None,
        ingest_skill_candidate: bool = True,
    ) -> ReflectionOutcome:
        outcome = self.workflow.reflect(
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            skill_steps=skill_steps,
            domain=domain or self.default_domain,
            ingest_skill_candidate=ingest_skill_candidate,
        )
        self.last_reflection = outcome
        return outcome

    def _mirror_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        domain = "user" if target == "user" else self.default_domain
        kind = str(metadata.get("kind") or "note")
        source = str(metadata.get("source") or "memory_write")
        confidence = float(metadata.get("confidence") or 0.6)
        tags = list(metadata.get("tags") or [])
        record_id = str(metadata.get("record_id") or "").strip()
        fragment = str(metadata.get("old_text") or "").strip()
        if action == "add":
            record = self.workflow.normalize_candidate(
                content,
                domain=domain,
                kind=kind,
                source=source,
                confidence=confidence,
                tags=tags,
                metadata=metadata,
            )
            stored = self.workflow.store.save_record(record)
            payload = {"action": action, "stored": stored.to_dict()}
            self.last_write = payload
            return payload
        if action in {"replace", "remove"} and not (record_id or fragment):
            matches = self.workflow.store.find_matching_records(content, domain=domain, limit=1, status=None)
            fragment = matches[0].id if matches else content[:48]
        if action == "replace":
            target_record = self.workflow.store.find_record(record_id) if record_id else None
            if target_record is None and fragment:
                matches = self.workflow.store.find_matching_records(fragment, domain=domain, limit=1, status=None)
                target_record = matches[0] if matches else None
            if target_record is None and not fragment and content:
                matches = self.workflow.store.find_matching_records(content, domain=domain, limit=1, status=None)
                target_record = matches[0] if matches else None
            if target_record is None:
                payload = {"action": action, "error": f"No record matched '{record_id or fragment or content[:48]}'."}
                self.last_write = payload
                return payload
            target_record.content = content
            target_record.kind = kind  # type: ignore[assignment]
            target_record.source = source
            target_record.confidence = confidence
            target_record.tags = tags
            target_record.metadata = metadata
            updated = self.workflow.store.upsert_record(target_record)
            payload = {"action": action, "stored": updated.to_dict()}
            self.last_write = payload
            return payload
        if action == "remove":
            removed = self.workflow.store.deactivate_matching_records(fragment, domain=domain, status="removed", limit=5)
            payload = {"action": action, "removed": [record.to_dict() for record in removed]}
            self.last_write = payload
            return payload
        payload = {"action": action, "error": f"Unknown memory write action '{action}'."}
        self.last_write = payload
        return payload

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.auto_mirror:
            self.last_write = {"action": action, "skipped": True}
            return
        self._mirror_write(action, target, content, metadata)

    def record_feedback(
        self,
        record_id: str,
        *,
        helpful: bool,
        note: str = "",
        weight: float = 0.05,
    ) -> dict[str, Any]:
        updated = self.workflow.store.record_feedback(record_id, helpful=helpful, note=note, weight=weight)
        return {
            "record_id": record_id,
            "helpful": helpful,
            "note": note,
            "weight": round(float(weight), 3),
            "updated": updated.to_dict() if updated else None,
        }

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        self.turn_number = turn_number
        self.last_turn_user = message.strip()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        domain: MemoryDomain | None = None,
        candidate: MemoryRecord | dict[str, Any] | str | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> GovernorPacket:
        self.turn_number += 1
        self.last_turn_user = user_content
        self.last_turn_assistant = assistant_content
        recall = self.workflow.retrieve(user_content, domain=domain, limit=5, max_chars=800)
        memory_context = self.build_memory_context(recall.memory_pack)
        ingest_outcome = None
        reflection_outcome = None
        writes: list[dict[str, Any]] = []
        if candidate is None and user_content.strip():
            candidate = user_content
        if candidate is not None:
            ingest_outcome = self.workflow.ingest(candidate, domain=domain)
            self.last_ingest = ingest_outcome
        if messages and self.auto_extract:
            extracted = self._extract_candidates_from_messages(messages)
            self.last_extracted = [candidate.to_dict() for candidate in extracted]
            self._ingest_candidates(extracted, force_domain=domain)
        if task_title and result_summary:
            reflection_outcome = self.workflow.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain or self.default_domain,
            )
            self.last_reflection = reflection_outcome
        packet = GovernorPacket(
            recall=recall,
            memory_context=memory_context,
            ingest=ingest_outcome,
            reflection=reflection_outcome,
            session=self._sync_session_snapshot(),
            extracted=list(self.last_extracted),
            writes=writes,
        )
        self.last_recall = recall
        self.last_packet = packet
        self._prefetch_cache_key = (user_content, domain, 5, 800)
        self._prefetch_cache_value = memory_context
        return packet

    def run_cycle(
        self,
        *,
        query: str,
        candidate: MemoryRecord | dict[str, Any] | str | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> GovernorPacket:
        recall = self.workflow.retrieve(query, domain=domain, limit=limit, max_chars=max_chars)
        memory_context = self.build_memory_context(recall.memory_pack)
        ingest_outcome = None
        reflection_outcome = None
        if candidate is not None:
            ingest_outcome = self.workflow.ingest(candidate, domain=domain)
            self.last_ingest = ingest_outcome
        if task_title and result_summary:
            reflection_outcome = self.workflow.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain or self.default_domain,
            )
            self.last_reflection = reflection_outcome
        packet = GovernorPacket(
            recall=recall,
            memory_context=memory_context,
            ingest=ingest_outcome,
            reflection=reflection_outcome,
            session=self._sync_session_snapshot(),
        )
        self.last_recall = recall
        self.last_packet = packet
        self._prefetch_cache_key = (query, domain, limit, max_chars)
        self._prefetch_cache_value = memory_context
        return packet

    def sync_all(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: list[dict[str, Any]] | None = None) -> GovernorPacket:
        if session_id:
            self.session_id = session_id
        return self.sync_turn(user_content, assistant_content, messages=messages)

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        if session_id:
            self.session_id = session_id
        return self.prefetch(query)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        if session_id:
            self.session_id = session_id
        self.queue_prefetch(query)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> dict[str, Any]:
        content = f"Delegation task: {task.strip()} | Result: {result.strip()}"
        candidate = MemoryRecord(
            id=new_memory_id("del"),
            domain="project",
            kind="pattern",
            content=content,
            confidence=0.78,
            source="delegation",
            tags=["delegation", child_session_id or self.session_id],
            metadata={"child_session_id": child_session_id, **kwargs},
        )
        outcome = self.workflow.ingest(candidate, domain="project", kind="pattern", source="delegation", confidence=0.78, tags=candidate.tags, metadata=candidate.metadata)
        self.last_ingest = outcome
        return {"candidate": candidate.to_dict(), "outcome": outcome.to_dict()}

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        previous = self.session_id
        self.session_id = new_session_id
        self.parent_session_id = parent_session_id or self.parent_session_id
        if reset or rewound:
            self.turn_number = 0
            self._prefetch_cache_key = None
            self._prefetch_cache_value = ""
            self._queued_prefetch = None
            self._queued_prefetch_value = ""
            self.sync_openclaw_memory(import_surface=True, export_surface=False, reason="session_switch")
        payload = {
            "previous_session_id": previous,
            "session_id": new_session_id,
            "parent_session_id": self.parent_session_id,
            "reset": reset,
            "rewound": rewound,
            **kwargs,
        }
        self.last_session_end = payload
        return payload

    def _compression_summary(self, messages: Iterable[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        extracted = self._extract_candidates_from_messages(messages)
        self.last_extracted = [candidate.to_dict() for candidate in extracted]
        if not extracted:
            return "", []
        lines = ["[CONTEXT SUMMARY]:"]
        for candidate in extracted[:5]:
            lines.append(f"- [{candidate.domain}/{candidate.kind}] {candidate.content}")
        return "\n".join(lines), self.last_extracted

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        summary, extracted = self._compression_summary(messages)
        if extracted and self.auto_extract:
            self._ingest_candidates([MemoryRecord.from_dict(item) for item in extracted], force_domain=None, source="pre_compress")
        self.last_pre_compress = summary
        return summary

    def on_session_end(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        domain: MemoryDomain | None = None,
        ingest_skill_candidate: bool = True,
    ) -> dict[str, Any]:
        bridge_sync = self.sync_openclaw_memory(import_surface=False, export_surface=True, reason="session_end")
        extracted: list[dict[str, Any]] = []
        reflection: ReflectionOutcome | None = None
        if messages and self.auto_extract:
            candidates = self._extract_candidates_from_messages(messages)
            extracted = self._ingest_candidates(candidates, force_domain=domain, source="session_end")
            self.last_extracted = [candidate.to_dict() for candidate in candidates]
        if task_title and result_summary:
            reflection = self.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain,
                ingest_skill_candidate=ingest_skill_candidate,
            )
        contradiction_parts: list[str] = [part for part in (task_title, result_summary, self.last_turn_user, self.last_turn_assistant) if part]
        if messages:
            contradiction_parts.extend(self._normalize_text(message.get("content")) for message in messages if isinstance(message, dict))
        contradiction_query = " ".join(part for part in contradiction_parts if part).strip()
        contradictions = self.contradict(contradiction_query, domain=domain, limit=5)["results"] if contradiction_query else []
        payload = {
            "session": self._sync_session_snapshot(),
            "bridge_sync": bridge_sync,
            "extracted": extracted,
            "reflection": reflection.to_dict() if reflection else None,
            "contradictions": contradictions,
            "last_pre_compress": self.last_pre_compress,
            "last_write": self.last_write,
        }
        self.last_session_end = payload
        return payload

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_add",
                "description": "Add a durable memory record without going through scoring.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "enum": ["memory", "user"]},
                        "content": {"type": "string"},
                        "kind": {"type": "string", "enum": ["preference", "fact", "decision", "pattern", "failure", "skill", "note"]},
                        "confidence": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                    },
                    "required": ["target", "content"],
                },
            },
            {
                "name": "memory_search",
                "description": "Search memory by query and return recalled context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": "string", "enum": ["user", "project", "agent"]},
                        "limit": {"type": "integer"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_feedback",
                "description": "Rate a stored memory as helpful or unhelpful.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "helpful": {"type": "boolean"},
                        "note": {"type": "string"},
                        "weight": {"type": "number"},
                    },
                    "required": ["record_id", "helpful"],
                },
            },
            {
                "name": "memory_contradict",
                "description": "Find memories that make conflicting claims about the same entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": "string", "enum": ["user", "project", "agent"]},
                        "threshold": {"type": "number"},
                        "limit": {"type": "integer"},
                    },
                },
            },
            {
                "name": "memory_forget",
                "description": "Remove memory records matching a fragment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "domain": {"type": "string", "enum": ["user", "project", "agent"]},
                    },
                    "required": ["fragment"],
                },
            },
            {
                "name": "memory_state",
                "description": "Inspect the current memory governor state.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "memory_reflect",
                "description": "Store a reflection and optional skill candidate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_title": {"type": "string"},
                        "result_summary": {"type": "string"},
                        "lessons": {"type": "string"},
                        "skill_steps": {"type": "array", "items": {"type": "string"}},
                        "domain": {"type": "string", "enum": ["user", "project", "agent"]},
                    },
                    "required": ["task_title", "result_summary"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if not isinstance(args, dict):
            args = {}
        if tool_name == "memory_add":
            payload = self._mirror_write(
                "add",
                str(args.get("target") or "memory"),
                str(args.get("content") or ""),
                {
                    "kind": args.get("kind") or "note",
                    "confidence": args.get("confidence") or 0.6,
                    "tags": args.get("tags") or [],
                    "metadata": args.get("metadata") or {},
                },
            )
            return json.dumps(payload, ensure_ascii=False)
        if tool_name == "memory_search":
            recall = self.recall(
                str(args.get("query") or ""),
                domain=args.get("domain") if args.get("domain") in {"user", "project", "agent"} else None,
                limit=int(args.get("limit") or 5),
                max_chars=int(args.get("max_chars") or 800),
            )
            return json.dumps(recall.to_dict(), ensure_ascii=False)
        if tool_name == "memory_feedback":
            payload = self.record_feedback(
                str(args.get("record_id") or ""),
                helpful=bool(args.get("helpful")),
                note=str(args.get("note") or ""),
                weight=float(args.get("weight") or 0.05),
            )
            return json.dumps(payload, ensure_ascii=False)
        if tool_name == "memory_contradict":
            payload = self.contradict(
                str(args.get("query") or ""),
                domain=args.get("domain") if args.get("domain") in {"user", "project", "agent"} else None,
                threshold=float(args.get("threshold") or 0.28),
                limit=int(args.get("limit") or 5),
            )
            return json.dumps(payload, ensure_ascii=False)
        if tool_name == "memory_forget":
            removed = self.workflow.store.deactivate_matching_records(
                str(args.get("fragment") or ""),
                domain=args.get("domain") if args.get("domain") in {"user", "project", "agent"} else None,
            )
            return json.dumps({"removed": [record.to_dict() for record in removed]}, ensure_ascii=False)
        if tool_name == "memory_state":
            return json.dumps(self._sync_session_snapshot(), ensure_ascii=False)
        if tool_name == "memory_reflect":
            reflection = self.reflect(
                task_title=str(args.get("task_title") or ""),
                result_summary=str(args.get("result_summary") or ""),
                lessons=str(args.get("lessons") or "") or None,
                skill_steps=list(args.get("skill_steps") or []) or None,
                domain=args.get("domain") if args.get("domain") in {"user", "project", "agent"} else None,
            )
            return json.dumps(reflection.to_dict(), ensure_ascii=False)
        raise ValueError(f"Unknown memory tool '{tool_name}'.")

    def on_session_end_summary(self) -> dict[str, Any]:
        return self.last_session_end or {}
