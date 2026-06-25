from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory_reflect import CorrectionTarget, MemoryReflector, ReflectionBundle
from .memory_retrieve import MemoryRetriever, RetrievedMemory
from .memory_score import MemoryScorer, ScoreDecision
from .memory_store import MemoryStore
from .types import MemoryDomain, MemoryKind, MemoryRecord

_VALID_DOMAINS: set[str] = {"user", "project", "agent"}
_VALID_KINDS: set[str] = {
    "preference",
    "fact",
    "decision",
    "pattern",
    "failure",
    "skill",
    "note",
}


@dataclass(slots=True)
class IngestOutcome:
    candidate: MemoryRecord
    decision: ScoreDecision
    stored: MemoryRecord | None = None
    staged: MemoryRecord | None = None

    @property
    def action(self) -> str:
        return self.decision.action

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "decision": self.decision.to_dict(),
            "action": self.action,
            "stored": self.stored.to_dict() if self.stored else None,
            "staged": self.staged.to_dict() if self.staged else None,
        }


@dataclass(slots=True)
class RecallOutcome:
    query: str
    domain: MemoryDomain | None
    matches: list[RetrievedMemory]
    memory_pack: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "domain": self.domain,
            "matches": [item.to_dict() for item in self.matches],
            "memory_pack": self.memory_pack,
        }


@dataclass(slots=True)
class ReflectionOutcome:
    bundle: ReflectionBundle
    reflection_saved: MemoryRecord
    skill_saved: MemoryRecord | None = None
    skill_outcome: IngestOutcome | None = None
    correction_results: list[dict[str, Any]] = field(default_factory=list)  # ── Phase 3 ──

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bundle": {
                "summary": self.bundle.summary.to_dict(),
                "skill_candidate": self.bundle.skill_candidate.to_dict() if self.bundle.skill_candidate else None,
                "correction_targets": [t.to_dict() for t in self.bundle.correction_targets],
            },
            "reflection_saved": self.reflection_saved.to_dict(),
            "skill_saved": self.skill_saved.to_dict() if self.skill_saved else None,
            "skill_outcome": self.skill_outcome.to_dict() if self.skill_outcome else None,
        }
        if self.correction_results:
            d["correction_results"] = list(self.correction_results)
        return d


class MemoryWorkflow:
    """High-level orchestration for Hermes-style memory handling.

    The workflow keeps the low-level components separate while providing a
    single place for routing, gating, retrieval, and reflection.
    """

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        scorer: MemoryScorer | None = None,
        retriever: MemoryRetriever | None = None,
        reflector: MemoryReflector | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.scorer = scorer or MemoryScorer()
        self.retriever = retriever or MemoryRetriever(self.store)
        self.reflector = reflector or MemoryReflector()

    def route_domain(self, candidate: MemoryRecord | dict[str, Any]) -> MemoryDomain:
        payload = candidate.to_dict() if isinstance(candidate, MemoryRecord) else dict(candidate)
        domain = str(payload.get("domain") or "").strip().lower()
        if domain in _VALID_DOMAINS:
            return domain  # type: ignore[return-value]

        kind = str(payload.get("kind") or "note").strip().lower()
        if kind not in _VALID_KINDS:
            kind = "note"

        source = str(payload.get("source") or "conversation").strip().lower()
        content = str(payload.get("content") or "").strip().lower()
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            explicit_scope = str(metadata.get("scope") or "").strip().lower()
            if explicit_scope in _VALID_DOMAINS:
                return explicit_scope  # type: ignore[return-value]

        if kind == "preference":
            return "user"
        if source in {"agent", "system", "tool"}:
            return "agent"
        if kind in {"skill", "failure", "pattern"}:
            return "project"
        if kind in {"decision", "fact", "note"}:
            if any(
                hint in content
                for hint in (
                    "i prefer",
                    "my preference",
                    "remember my",
                    "for this project",
                    "project state",
                    "task state",
                    "decision",
                    "we should",
                )
            ):
                return "user" if "my" in content else "project"
            return "project"
        return "project"

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
            payload["domain"] = domain or self.route_domain(payload)
            record = MemoryRecord.from_dict(payload)

        if domain:
            record.domain = domain
        elif not record.domain or record.domain not in _VALID_DOMAINS:
            record.domain = self.route_domain(record)

        if not record.kind:
            record.kind = kind
        if not record.source:
            record.source = source
        if tags is not None and not record.tags:
            record.tags = list(tags)
        if metadata and not record.metadata:
            record.metadata = dict(metadata)

        return record

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
            return IngestOutcome(candidate=record, decision=decision, stored=stored)
        if decision.action == "stage":
            staged = self.store.stage_record(record, reason="score gate", score=decision.score)
            return IngestOutcome(candidate=record, decision=decision, staged=staged)
        return IngestOutcome(candidate=record, decision=decision)

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
        return RecallOutcome(query=query, domain=resolved_domain, matches=matches, memory_pack=memory_pack)

    def reflect(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        skill_steps: list[str] | None = None,
        domain: MemoryDomain = "project",
        ingest_skill_candidate: bool = True,
        correction_targets: list[CorrectionTarget] | None = None,  # ── Phase 3 ──
        corrects_ids: list[str] | None = None,  # ── Phase 3 ──
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

        # ── Phase 3: 执行修正 ──
        correction_results: list[dict[str, Any]] = []
        for target in bundle.correction_targets:
            if target.action == "supersede":
                self.store.supersede_record(
                    target.record_id, reflection_saved.id,
                    adjudication=target.reason or "superseded by reflection",
                )
                correction_results.append({
                    "record_id": target.record_id, "action": "supersede", "by": reflection_saved.id,
                })
            elif target.action == "dispute":
                rec = self.store.find_record(target.record_id)
                if rec:
                    rec.status = "disputed"
                    rec.touch()
                    meta = dict(rec.metadata)
                    meta["disputed_by_reflection"] = reflection_saved.id
                    meta["dispute_reason"] = target.reason
                    rec.metadata = meta
                    self.store.upsert_record(rec, _skip_contradiction=True)
                    correction_results.append({
                        "record_id": target.record_id, "action": "dispute",
                    })
            elif target.action == "decay":
                rec = self.store.find_record(target.record_id)
                if rec and target.confidence > 0:
                    meta = dict(rec.metadata)
                    meta["trust"] = max(0.1, float(meta.get("trust", 0.5)) - 0.2)
                    rec.metadata = meta
                    rec.touch()
                    self.store.upsert_record(rec, _skip_contradiction=True)
                    correction_results.append({
                        "record_id": target.record_id, "action": "decay",
                        "new_trust": meta["trust"],
                    })

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
        return ReflectionOutcome(
            bundle=bundle,
            reflection_saved=reflection_saved,
            skill_saved=skill_saved,
            skill_outcome=skill_outcome,
            correction_results=correction_results,
        )

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
    ) -> dict[str, Any]:
        packet: dict[str, Any] = {
            "recall": self.retrieve(query, domain=domain, limit=limit, max_chars=max_chars).to_dict(),
        }
        if candidate is not None:
            packet["ingest"] = self.ingest(candidate, domain=domain).to_dict()
        if task_title and result_summary:
            packet["reflection"] = self.reflect(
                task_title=task_title,
                result_summary=result_summary,
                lessons=lessons,
                skill_steps=skill_steps,
                domain=domain or "project",
            ).to_dict()
        return packet
