from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .memory_reflect import CorrectionTarget, MemoryReflector, ReflectionBundle
from .memory_retrieve import MemoryRetriever, RetrievedMemory
from .memory_score import MemoryScorer, ScoreDecision
from .memory_store import MemoryStore
from .memory_workflow import IngestOutcome, MemoryWorkflow, RecallOutcome, ReflectionOutcome
from .types import ConflictRelation, MemoryDomain, MemoryKind, MemoryRecord, new_memory_id


@dataclass(slots=True)
class GovernorPacket:
    session_id: str
    action: str
    query: str = ""
    summary: str = ""
    recall: RecallOutcome | None = None
    ingest: IngestOutcome | None = None
    reflection: ReflectionOutcome | None = None
    consistency_report: dict[str, Any] | None = None
    bridge_sync: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "action": self.action,
            "query": self.query,
            "summary": self.summary,
            "messages": list(self.messages),
            "metadata": dict(self.metadata),
        }
        if self.recall is not None:
            payload["recall"] = self.recall.to_dict()
        if self.ingest is not None:
            payload["ingest"] = self.ingest.to_dict()
        if self.reflection is not None:
            payload["reflection"] = self.reflection.to_dict()
        if self.consistency_report is not None:
            payload["consistency_report"] = self.consistency_report
        if self.bridge_sync is not None:
            payload["bridge_sync"] = self.bridge_sync
        return payload


class OpenClawMemoryGovernor:
    """Lifecycle adapter that gives the plugin pack Hermes-style governance."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        session_id: str = "",
        platform: str = "openclaw",
        auto_extract: bool = True,
        auto_mirror: bool = True,
        openclaw_home: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        self_improving_dir: str | Path | None = None,
        proactivity_dir: str | Path | None = None,
    ) -> None:
        self.workflow = MemoryWorkflow(store=store)
        self.store = self.workflow.store
        self.scorer: MemoryScorer = self.workflow.scorer
        self.retriever: MemoryRetriever = self.workflow.retriever
        self.reflector: MemoryReflector = self.workflow.reflector

        self.session_id = session_id
        self.platform = platform
        self.auto_extract = auto_extract
        self.auto_mirror = auto_mirror
        self.openclaw_home = Path(openclaw_home).expanduser() if openclaw_home else None
        self.workspace_dir = Path(workspace_dir).expanduser() if workspace_dir else None
        self.self_improving_dir = Path(self_improving_dir).expanduser() if self_improving_dir else None
        self.proactivity_dir = Path(proactivity_dir).expanduser() if proactivity_dir else None

        self._lock = threading.RLock()
        self._closed = False
        self._session_started_at = self._utc_now()
        self._turn_number = 0
        self._last_turn_number = 0
        self._prefetch_cache: dict[tuple[str, str | None, int, int], tuple[str, str]] = {}
        self._pending_prefetch: list[tuple[str, str | None, int, int]] = []

        self.last_recall: RecallOutcome | None = None
        self.last_ingest: IngestOutcome | None = None
        self.last_reflection: ReflectionOutcome | None = None
        self.last_bridge_sync: dict[str, Any] | None = None
        self.last_contradictions: list[ConflictRelation] = []
        self.last_consistency_report: dict[str, Any] | None = None
        self.last_session_summary: str = ""

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _safe_role(self, message: dict[str, Any] | str | None) -> str:
        if isinstance(message, dict):
            return str(message.get("role") or "").strip().lower()
        return ""

    def _message_text(self, message: dict[str, Any] | str | None) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return self._normalize_text(message)
        for key in ("content", "text", "body", "message"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return self._normalize_text(value)
        return ""

    def _kind_from_text(self, text: str, *, role: str = "", fallback: MemoryKind = "note") -> MemoryKind:
        lowered = text.lower()
        if any(word in lowered for word in ("prefer", "always", "never", "like", "dislike")):
            return "preference"
        if any(word in lowered for word in ("decided", "decision", "choose", "replace", "instead")):
            return "decision"
        if any(word in lowered for word in ("fail", "failed", "error", "issue", "blocked")):
            return "failure"
        if any(word in lowered for word in ("pattern", "repeat", "lesson", "works", "how to")):
            return "pattern"
        if role == "assistant" and len(lowered) > 120:
            return "fact"
        return fallback

    def _candidate_from_message(
        self,
        message: dict[str, Any] | str,
        *,
        domain: MemoryDomain | None = None,
    ) -> MemoryRecord | None:
        role = self._safe_role(message)
        text = self._message_text(message)
        if not text:
            return None
        source = "conversation"
        if role in {"assistant", "tool"}:
            source = "task_result"
        kind = self._kind_from_text(text, role=role, fallback="note")
        payload: dict[str, Any] = {
            "id": new_memory_id("mem"),
            "domain": domain or self.workflow.route_domain({"content": text, "kind": kind, "source": source}),
            "kind": kind,
            "content": text,
            "source": source,
            "confidence": 0.62 if kind != "note" else 0.48,
            "metadata": {"role": role} if role else {},
        }
        return self.workflow.normalize_candidate(payload, domain=domain)

    def _extract_candidates_from_messages(
        self,
        messages: list[dict[str, Any]] | None,
        *,
        domain: MemoryDomain | None = None,
    ) -> list[MemoryRecord]:
        if not messages:
            return []
        candidates: list[MemoryRecord] = []
        seen: set[str] = set()
        for message in messages:
            candidate = self._candidate_from_message(message, domain=domain)
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
        domain: MemoryDomain | None = None,
    ) -> list[IngestOutcome]:
        outcomes: list[IngestOutcome] = []
        for candidate in candidates:
            outcome = self.ingest(candidate, domain=domain or candidate.domain)
            outcomes.append(outcome)
        return outcomes

    def _append_prefetch(self, query: str, domain: str | None, limit: int, max_chars: int) -> None:
        key = (query.strip().lower(), domain, int(limit), int(max_chars))
        if key not in self._pending_prefetch:
            self._pending_prefetch.append(key)

    def _maybe_return_cached_prefetch(self, query: str, *, domain: MemoryDomain | None = None, limit: int = 5, max_chars: int = 800) -> str | None:
        key = (query.strip().lower(), domain, int(limit), int(max_chars))
        cached = self._prefetch_cache.get(key)
        if cached is None:
            return None
        return cached[1]

    def _sync_session_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "platform": self.platform,
                "turn_number": self._turn_number,
                "last_turn_number": self._last_turn_number,
                "started_at": self._session_started_at,
                "record_counts": {
                    "total": self.store.count_records(status=None),
                    "active": self.store.count_records(status="active"),
                    "staged": self.store.count_records(status="staged"),
                    "superseded": self.store.count_records(status="superseded"),
                    "disputed": self.store.count_records(status="disputed"),
                    "removed": self.store.count_records(status="removed"),
                },
                "last_recall": self.last_recall.to_dict() if self.last_recall else None,
                "last_ingest": self.last_ingest.to_dict() if self.last_ingest else None,
                "last_reflection": self.last_reflection.to_dict() if self.last_reflection else None,
                "last_bridge_sync": dict(self.last_bridge_sync or {}),
            }

    def _format_recall(self, recall: RecallOutcome) -> str:
        return recall.memory_pack

    def _topic_label(self, left: MemoryRecord | None, right: MemoryRecord | None, fallback: str = "") -> str:
        for record in (left, right):
            if record is None:
                continue
            content = self._normalize_text(record.content)
            if content:
                return content[:120]
        return fallback or ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self.session_id = session_id or self.session_id
        self.platform = str(kwargs.get("platform") or self.platform)
        openclaw_home = kwargs.get("openclaw_home") or kwargs.get("hermes_home") or self.openclaw_home
        workspace_dir = kwargs.get("workspace_dir") or self.workspace_dir
        self_improving_dir = kwargs.get("self_improving_dir") or self.self_improving_dir
        proactivity_dir = kwargs.get("proactivity_dir") or self.proactivity_dir
        self.openclaw_home = Path(openclaw_home).expanduser() if openclaw_home else None
        self.workspace_dir = Path(workspace_dir).expanduser() if workspace_dir else None
        self.self_improving_dir = Path(self_improving_dir).expanduser() if self_improving_dir else None
        self.proactivity_dir = Path(proactivity_dir).expanduser() if proactivity_dir else None
        self._session_started_at = self._utc_now()
        self._turn_number = 0
        self._last_turn_number = 0
        self._prefetch_cache.clear()
        self._pending_prefetch.clear()
        self.last_bridge_sync = {
            "event": "initialize",
            "session_id": self.session_id,
            "platform": self.platform,
            "started_at": self._session_started_at,
        }

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.store.close()
            self._closed = True

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "OpenClawMemoryGovernor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Prompt and retrieval
    # ------------------------------------------------------------------
    def system_prompt_block(self) -> str:
        return (
            "Memory governor active. Prefetch relevant memory before answering, "
            "prefer recall before write, and surface contradictions instead of silently overwriting them."
        )

    def build_memory_context(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        return self.retriever.build_memory_pack(query, domain=domain, limit=limit, max_chars=max_chars)

    def prefetch(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        cached = self._maybe_return_cached_prefetch(query, domain=domain, limit=limit, max_chars=max_chars)
        if cached is not None:
            return cached
        recall = self.retrieve(query, domain=domain, limit=limit, max_chars=max_chars)
        memory_context = recall.memory_pack
        key = (query.strip().lower(), domain, int(limit), int(max_chars))
        self._prefetch_cache[key] = (self._utc_now(), memory_context)
        self.last_bridge_sync = {
            "event": "prefetch",
            "query": query,
            "domain": domain,
            "limit": limit,
            "max_chars": max_chars,
        }
        return memory_context

    def queue_prefetch(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> None:
        self._append_prefetch(query, domain, limit, max_chars)

    def prefetch_all(
        self,
        queries: Iterable[str],
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> str:
        packs = [self.prefetch(query, domain=domain, limit=limit, max_chars=max_chars) for query in queries if query.strip()]
        return "\n\n".join(pack for pack in packs if pack)

    def queue_prefetch_all(
        self,
        queries: Iterable[str],
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> None:
        for query in queries:
            if query.strip():
                self.queue_prefetch(query, domain=domain, limit=limit, max_chars=max_chars)

    def retrieve(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        limit: int = 5,
        max_chars: int = 800,
    ) -> RecallOutcome:
        resolved_domain = domain or None
        matches = self.retriever.search(query, domain=resolved_domain, limit=limit)
        memory_pack = self.retriever.build_memory_pack(
            query,
            domain=resolved_domain,
            limit=limit,
            max_chars=max_chars,
        )
        recall = RecallOutcome(query=query, domain=resolved_domain, matches=matches, memory_pack=memory_pack)
        self.last_recall = recall
        return recall

    # ------------------------------------------------------------------
    # Ingest and reflection
    # ------------------------------------------------------------------
    def normalize_candidate(
        self,
        candidate: MemoryRecord | dict[str, Any] | str,
        *,
        domain: MemoryDomain | None = None,
        kind: MemoryKind = "note",
        source: str = "conversation",
        confidence: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if isinstance(candidate, MemoryRecord):
            record = candidate
        elif isinstance(candidate, str):
            record = MemoryRecord.from_dict(
                {
                    "content": candidate,
                    "domain": domain or self.route_domain({"content": candidate, "kind": kind, "source": source}),
                    "kind": kind,
                    "source": source,
                    "confidence": confidence,
                    "tags": tags or [],
                    "metadata": metadata or {},
                }
            )
        else:
            payload = dict(candidate)
            payload.setdefault("content", "")
            payload.setdefault("kind", kind)
            payload.setdefault("source", source)
            payload.setdefault("confidence", confidence)
            payload.setdefault("tags", tags or [])
            payload.setdefault("metadata", metadata or {})
            payload["domain"] = domain or self.workflow.route_domain(payload)
            record = MemoryRecord.from_dict(payload)

        if domain:
            record.domain = domain
        elif not record.domain:
            record.domain = self.workflow.route_domain(record)
        if not record.kind:
            record.kind = kind
        if not record.source:
            record.source = source
        if tags is not None and not record.tags:
            record.tags = list(tags)
        if metadata and not record.metadata:
            record.metadata = dict(metadata)
        return record

    def route_domain(self, candidate: MemoryRecord | dict[str, Any]) -> MemoryDomain:
        return self.workflow.route_domain(candidate)

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
        record = self.normalize_candidate(
            candidate,
            domain=domain,
            kind=kind,
            source=source,
            confidence=confidence,
            tags=tags,
            metadata=metadata,
        )
        decision = self.scorer.classify(record)
        if decision.action == "auto_write":
            stored = self.store.save_record(record)
            outcome = IngestOutcome(candidate=record, decision=decision, stored=stored)
        elif decision.action == "stage":
            staged = self.store.stage_record(record, reason="score gate", score=decision.score)
            outcome = IngestOutcome(candidate=record, decision=decision, staged=staged)
        else:
            outcome = IngestOutcome(candidate=record, decision=decision)
        self.last_ingest = outcome
        return outcome

    def reflect(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        domain: MemoryDomain = "project",
        ingest_skill_candidate: bool = True,
        correction_targets: list[CorrectionTarget] | None = None,
        corrects_ids: list[str] | None = None,
    ) -> ReflectionOutcome:
        bundle = self.reflector.bundle(
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            skill_steps=skill_steps,
            domain=domain,
            correction_targets=correction_targets,
            corrects_ids=corrects_ids,
        )
        reflection_saved = self.store.save_reflection(bundle.summary)
        skill_saved = None
        skill_outcome = None
        correction_results: list[dict[str, Any]] = []

        for target in bundle.correction_targets:
            if target.action == "supersede":
                self.store.supersede_record(
                    target.record_id,
                    reflection_saved.id,
                    adjudication=target.reason or "superseded by reflection",
                )
                correction_results.append({"record_id": target.record_id, "action": "supersede", "by": reflection_saved.id})
            elif target.action == "dispute":
                rec = self.store.find_record(target.record_id)
                if rec is not None:
                    rec.status = "disputed"
                    rec.touch()
                    meta = dict(rec.metadata)
                    meta["disputed_by_reflection"] = reflection_saved.id
                    meta["dispute_reason"] = target.reason
                    rec.metadata = meta
                    self.store.upsert_record(rec, _skip_contradiction=True)
                    correction_results.append({"record_id": target.record_id, "action": "dispute"})
            elif target.action == "decay":
                rec = self.store.find_record(target.record_id)
                if rec is not None and target.confidence > 0:
                    meta = dict(rec.metadata)
                    meta["trust"] = max(0.1, float(meta.get("trust", 0.5)) - 0.2)
                    rec.metadata = meta
                    rec.touch()
                    self.store.upsert_record(rec, _skip_contradiction=True)
                    correction_results.append({"record_id": target.record_id, "action": "decay", "new_trust": meta["trust"]})

        if bundle.skill_candidate and ingest_skill_candidate:
            skill_outcome = self.ingest(
                bundle.skill_candidate,
                domain=domain,
                kind="skill",
                source="reflection",
                confidence=bundle.skill_candidate.confidence,
                tags=bundle.skill_candidate.tags,
                metadata=bundle.skill_candidate.metadata,
            )
            skill_saved = skill_outcome.stored or skill_outcome.staged
        elif bundle.skill_candidate:
            skill_saved = bundle.skill_candidate

        outcome = ReflectionOutcome(
            bundle=bundle,
            reflection_saved=reflection_saved,
            skill_saved=skill_saved,
            skill_outcome=skill_outcome,
            correction_results=correction_results,
        )
        self.last_reflection = outcome
        return outcome

    # ------------------------------------------------------------------
    # Contradiction and consistency
    # ------------------------------------------------------------------
    def contradict(
        self,
        query: MemoryRecord | dict[str, Any] | str,
        *,
        domain: MemoryDomain | None = None,
        threshold: float = 0.28,
        limit: int = 5,
    ) -> list[ConflictRelation]:
        candidate = self.normalize_candidate(query, domain=domain)
        conflicts = self.store.contradiction_check(candidate, domain=domain or candidate.domain, threshold=threshold, limit=limit)
        self.last_contradictions = list(conflicts)
        return conflicts

    def build_consistency_report(
        self,
        *,
        contradictions: list[ConflictRelation] | None = None,
        correction_results: list[dict[str, Any]] | None = None,
        task_title: str = "",
        result_summary: str = "",
        domain: MemoryDomain | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        contradictions = list(contradictions or [])
        correction_results = list(correction_results or [])
        entries: list[dict[str, Any]] = []

        resolved = 0
        pending = 0
        strong_conflicts = 0

        for result in correction_results:
            record_id = str(result.get("record_id") or "")
            action = str(result.get("action") or "")
            if action in {"supersede", "dispute", "decay"}:
                resolved += 1
                record = self.store.find_record(record_id)
                entries.append(
                    {
                        "category": "resolved",
                        "topic": self._topic_label(record, None, fallback=record_id),
                        "left_id": record_id,
                        "right_id": str(result.get("by") or ""),
                        "resolution": action,
                        "adjudication": str(result.get("by") or action),
                    }
                )

        for conflict in contradictions:
            left = self.store.find_record(conflict.candidate_id)
            right = self.store.find_record(conflict.existing_id)
            topic = self._topic_label(left, right, fallback=conflict.candidate_id)
            if conflict.score >= 0.6:
                strong_conflicts += 1
                entries.append(
                    {
                        "category": "strong_conflicts",
                        "topic": topic,
                        "left_id": conflict.candidate_id,
                        "right_id": conflict.existing_id,
                        "resolution": conflict.suggested_action,
                        "adjudication": conflict.rule_applied or conflict.suggested_action,
                    }
                )
            else:
                pending += 1
                entries.append(
                    {
                        "category": "pending",
                        "topic": topic,
                        "left_id": conflict.candidate_id,
                        "right_id": conflict.existing_id,
                        "resolution": conflict.suggested_action,
                        "adjudication": conflict.rule_applied or conflict.suggested_action,
                    }
                )

        if not entries and (task_title or result_summary or messages):
            entries.append(
                {
                    "category": "pending",
                    "topic": self._normalize_text(task_title or result_summary or "session audit")[:120],
                    "left_id": "",
                    "right_id": "",
                    "resolution": "review",
                    "adjudication": "session-end",
                }
            )
            pending = 1

        summary = {
            "resolved": resolved,
            "pending": pending,
            "strong_conflicts": strong_conflicts,
            "total_entries": len(entries),
            "task_title": task_title,
            "domain": domain,
        }
        return {
            "summary": summary,
            "resolved": [entry for entry in entries if entry["category"] == "resolved"],
            "pending": [entry for entry in entries if entry["category"] == "pending"],
            "strong_conflicts": [entry for entry in entries if entry["category"] == "strong_conflicts"],
            "entries": entries,
            "generated_at": self._utc_now(),
            "session_id": self.session_id,
        }

    def _write_session_ledger(self, payload: dict[str, Any]) -> Path:
        ledger_path = self.store.root_dir / "session_ledger.jsonl"
        ledger_entry = {
            "session_id": self.session_id,
            "platform": self.platform,
            "generated_at": self._utc_now(),
            **payload,
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger_entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return ledger_path

    # ------------------------------------------------------------------
    # Session events
    # ------------------------------------------------------------------
    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        self._turn_number = int(turn_number or 0)
        self._last_turn_number = self._turn_number
        if message and self.auto_extract:
            self.prefetch(message, domain=kwargs.get("domain"), limit=int(kwargs.get("limit") or 5), max_chars=int(kwargs.get("max_chars") or 800))

    def on_turn_end(
        self,
        user_content: str,
        assistant_content: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        domain: MemoryDomain | None = None,
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
        ).to_dict()

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
        self.session_id = new_session_id or self.session_id
        if reset:
            self._prefetch_cache.clear()
            self._pending_prefetch.clear()
        self.last_bridge_sync = {
            "event": "session_switch",
            "previous_session_id": previous,
            "new_session_id": self.session_id,
            "parent_session_id": parent_session_id,
            "reset": reset,
            "rewound": rewound,
        }
        return {
            "session_id": self.session_id,
            "previous_session_id": previous,
            "snapshot": self._sync_session_snapshot(),
        }

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> dict[str, Any]:
        reflection = self.reflect(
            task_title=task,
            result_summary=result,
            lessons=kwargs.get("lessons"),
            skill_steps=kwargs.get("skill_steps"),
            domain=kwargs.get("domain") or "project",
            ingest_skill_candidate=bool(kwargs.get("ingest_skill_candidate", True)),
        )
        packet = {
            "event": "delegation",
            "task": task,
            "child_session_id": child_session_id,
            "reflection": reflection.to_dict(),
        }
        self.last_bridge_sync = packet
        return packet

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "id": new_memory_id("mem"),
            "domain": self.route_domain({"content": content, "kind": target, "source": action, "metadata": metadata or {}}),
            "kind": target if target in {"preference", "fact", "decision", "pattern", "failure", "skill", "note"} else "note",
            "content": content,
            "source": action,
            "metadata": metadata or {},
        }
        if action in {"forget", "remove", "delete"}:
            self.store.deactivate_matching_records(content, domain=payload["domain"], status="removed", limit=3)
            return
        self.ingest(payload, domain=payload["domain"], kind=payload["kind"], source=action, metadata=metadata)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        domain: MemoryDomain | None = None,
        task_title: str | None = None,
        result_summary: str | None = None,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
    ) -> GovernorPacket:
        if session_id:
            self.session_id = session_id
        self._turn_number += 1
        self._last_turn_number = self._turn_number

        recall = self.retrieve(user_content or assistant_content, domain=domain, limit=5, max_chars=800)
        candidates: list[MemoryRecord] = []
        if self.auto_extract:
            candidates.extend(self._extract_candidates_from_messages(messages, domain=domain))
            for text in (user_content, assistant_content):
                if text.strip():
                    candidate = self._candidate_from_message(text, domain=domain)
                    if candidate is not None:
                        candidates.append(candidate)
        ingest_outcome = None
        if candidates:
            ingest_outcome = self._ingest_candidates(candidates, domain=domain)[-1]
        reflection_outcome = None
        if task_title and result_summary:
            reflection_outcome = self.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain or "project",
            )
            refreshed = self.retrieve(user_content or assistant_content, domain=domain, limit=5, max_chars=800)
            recall = refreshed

        self.last_recall = recall
        if ingest_outcome is not None:
            self.last_ingest = ingest_outcome
        if reflection_outcome is not None:
            self.last_reflection = reflection_outcome
        self.last_bridge_sync = {
            "event": "sync_turn",
            "session_id": self.session_id,
            "turn_number": self._turn_number,
        }
        return GovernorPacket(
            session_id=self.session_id,
            action="sync_turn",
            query=user_content or assistant_content,
            recall=recall,
            ingest=ingest_outcome,
            reflection=reflection_outcome,
            bridge_sync=self.last_bridge_sync,
            messages=list(messages or []),
            metadata={"domain": domain, "task_title": task_title},
        )

    def sync_openclaw_memory(
        self,
        *,
        import_surface: bool = True,
        export_surface: bool = False,
        reason: str = "manual",
    ) -> GovernorPacket:
        payload: dict[str, Any] = {
            "event": "sync_openclaw_memory",
            "reason": reason,
            "import_surface": import_surface,
            "export_surface": export_surface,
        }
        if import_surface and self._pending_prefetch:
            query, domain, limit, max_chars = self._pending_prefetch.pop(0)
            payload["prefetch"] = self.prefetch(query, domain=domain, limit=limit, max_chars=max_chars)
        if export_surface:
            payload["state"] = self._sync_session_snapshot()
        self.last_bridge_sync = payload
        return GovernorPacket(
            session_id=self.session_id,
            action="sync_openclaw_memory",
            summary=reason,
            bridge_sync=payload,
            metadata={"import_surface": import_surface, "export_surface": export_surface},
        )

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        candidates = self._extract_candidates_from_messages(messages)
        if candidates and self.auto_extract:
            self._ingest_candidates(candidates, domain=None)
        return self.prefetch_all(self._message_text(message) for message in messages if self._message_text(message))

    def on_session_end(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        domain = kwargs.get("domain") or None
        task_title = str(kwargs.get("task_title") or "").strip()
        result_summary = str(kwargs.get("result_summary") or "").strip()
        lessons = kwargs.get("lessons")
        skill_steps = kwargs.get("skill_steps")
        ingest_skill_candidate = bool(kwargs.get("ingest_skill_candidate", True))

        reflection = None
        has_content = bool(task_title or result_summary or lessons or skill_steps)
        has_messages_with_text = any(self._message_text(m) for m in (messages or []))
        if has_content or has_messages_with_text:
            fallback_summary = self._message_text(messages[-1]) if messages else ""
            if not (task_title or result_summary) and not fallback_summary:
                reflection = None
            else:
                reflection = self.reflect(
                    task_title=task_title or "session end",
                    result_summary=result_summary or fallback_summary,
                    lessons=lessons,
                    skill_steps=skill_steps,
                    domain=domain or "project",
                    ingest_skill_candidate=ingest_skill_candidate,
                    correction_targets=kwargs.get("correction_targets"),
                    corrects_ids=kwargs.get("corrects_ids"),
                )
        contradictions: list[ConflictRelation] = []
        if reflection is not None:
            contradictions = self.contradict(reflection.reflection_saved, domain=domain or reflection.reflection_saved.domain, limit=5)
        elif messages:
            candidates = self._extract_candidates_from_messages(messages, domain=domain)
            for candidate in candidates:
                contradictions.extend(self.contradict(candidate, domain=domain or candidate.domain, limit=3))
                break
        report = self.build_consistency_report(
            contradictions=contradictions,
            correction_results=reflection.correction_results if reflection else [],
            task_title=task_title or "session end",
            result_summary=result_summary,
            domain=domain,
            messages=messages,
        )
        ledger_path = self._write_session_ledger(
            {
                "consistency_report": report,
                "reflection": reflection.to_dict() if reflection else None,
                "messages": messages,
            }
        )
        self.last_consistency_report = report
        self.last_session_summary = self.build_session_end_summary(report)
        self.last_bridge_sync = {
            "event": "session_end",
            "ledger_path": str(ledger_path),
            "summary": self.last_session_summary,
        }
        return {
            "session_id": self.session_id,
            "consistency_report": report,
            "reflection": reflection.to_dict() if reflection else None,
            "ledger_path": str(ledger_path),
            "summary": self.last_session_summary,
            "snapshot": self._sync_session_snapshot(),
        }

    # ------------------------------------------------------------------
    # Tool surface
    # ------------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_add",
                "description": "Write a durable memory record.",
                "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "domain": {"type": "string"}, "kind": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["content"]},
            },
            {
                "name": "memory_search",
                "description": "Search memory with status-aware recall.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "domain": {"type": "string"}, "limit": {"type": "integer"}, "max_chars": {"type": "integer"}}, "required": ["query"]},
            },
            {
                "name": "memory_feedback",
                "description": "Adjust trust for a memory record.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}, "helpful": {"type": "boolean"}, "note": {"type": "string"}}, "required": ["record_id", "helpful"]},
            },
            {
                "name": "memory_contradict",
                "description": "Return contradictions for a candidate memory.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "domain": {"type": "string"}, "threshold": {"type": "number"}, "limit": {"type": "integer"}}, "required": ["query"]},
            },
            {
                "name": "memory_forget",
                "description": "Mark matching memories as removed.",
                "parameters": {"type": "object", "properties": {"fragment": {"type": "string"}, "domain": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["fragment"]},
            },
            {
                "name": "memory_reflect",
                "description": "Create a reflection record and apply correction targets.",
                "parameters": {"type": "object", "properties": {"task_title": {"type": "string"}, "result_summary": {"type": "string"}, "lessons": {"type": "string"}, "skill_steps": {"type": "array", "items": {"type": "string"}}, "domain": {"type": "string"}}, "required": ["task_title", "result_summary"]},
            },
            {
                "name": "memory_state",
                "description": "Return a compact snapshot of the current memory state.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def memory_state(self) -> dict[str, Any]:
        return self._sync_session_snapshot()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        tool_name = str(tool_name or "").strip()
        args = dict(args or {})
        if tool_name == "memory_add":
            outcome = self.ingest(
                args.get("content", ""),
                domain=args.get("domain"),
                kind=str(args.get("kind") or "note"),
                confidence=float(args.get("confidence", 0.5)),
                metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
            )
            return json.dumps(outcome.to_dict(), ensure_ascii=False)
        if tool_name == "memory_search":
            recall = self.retrieve(
                str(args.get("query") or ""),
                domain=args.get("domain"),
                limit=int(args.get("limit") or 5),
                max_chars=int(args.get("max_chars") or 800),
            )
            return json.dumps(recall.to_dict(), ensure_ascii=False)
        if tool_name == "memory_feedback":
            record = self.store.record_feedback(
                str(args.get("record_id") or ""),
                helpful=bool(args.get("helpful", True)),
                note=str(args.get("note") or ""),
            )
            return json.dumps(record.to_dict() if record else None, ensure_ascii=False)
        if tool_name == "memory_contradict":
            conflicts = self.contradict(
                str(args.get("query") or ""),
                domain=args.get("domain"),
                threshold=float(args.get("threshold", 0.28)),
                limit=int(args.get("limit") or 5),
            )
            return json.dumps([conflict.to_dict() for conflict in conflicts], ensure_ascii=False)
        if tool_name == "memory_forget":
            records = self.store.deactivate_matching_records(
                str(args.get("fragment") or ""),
                domain=args.get("domain"),
                limit=int(args.get("limit") or 5),
            )
            return json.dumps([record.to_dict() for record in records], ensure_ascii=False)
        if tool_name == "memory_reflect":
            outcome = self.reflect(
                task_title=str(args.get("task_title") or "session reflection"),
                result_summary=str(args.get("result_summary") or ""),
                lessons=args.get("lessons"),
                skill_steps=list(args.get("skill_steps") or []),
                domain=str(args.get("domain") or "project"),
            )
            return json.dumps(outcome.to_dict(), ensure_ascii=False)
        if tool_name == "memory_state":
            return json.dumps(self.memory_state(), ensure_ascii=False)
        raise ValueError(f"Unknown memory tool: {tool_name}")

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
        recall = self.retrieve(query, domain=domain, limit=limit, max_chars=max_chars)
        ingest_outcome = self.ingest(candidate, domain=domain) if candidate is not None else None
        reflection_outcome = None
        if task_title and result_summary:
            reflection_outcome = self.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain or "project",
            )
        packet = GovernorPacket(
            session_id=self.session_id,
            action="run_cycle",
            query=query,
            recall=recall,
            ingest=ingest_outcome,
            reflection=reflection_outcome,
            bridge_sync={"event": "run_cycle"},
            metadata={"domain": domain},
        )
        self.last_bridge_sync = packet.bridge_sync
        return packet

    def on_session_end_summary(self, *args: Any, **kwargs: Any) -> str:
        if args and isinstance(args[0], dict):
            return self.build_session_end_summary(args[0])
        report = self.build_consistency_report(*args, **kwargs)
        return self.build_session_end_summary(report)

    def build_session_end_summary(self, report: dict[str, Any]) -> str:
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        return (
            f"resolved={summary.get('resolved', 0)}, "
            f"pending={summary.get('pending', 0)}, "
            f"strong_conflicts={summary.get('strong_conflicts', 0)}"
        )

    def sync_all(self, *args: Any, **kwargs: Any) -> GovernorPacket:
        return self.sync_openclaw_memory(*args, **kwargs)
