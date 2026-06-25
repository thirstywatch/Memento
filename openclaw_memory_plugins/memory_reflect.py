from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .types import MemoryRecord, new_memory_id


@dataclass(slots=True)
class CorrectionTarget:
    """一条反思指向的修正目标。"""
    record_id: str
    action: str  # "supersede" | "dispute" | "decay"
    reason: str = ""
    confidence: float = 0.0  # 新置信度（decay 时用）

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "action": self.action,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
        }


@dataclass(slots=True)
class ReflectionBundle:
    summary: MemoryRecord
    skill_candidate: MemoryRecord | None = None
    # ── Phase 3: 修正指向 ──
    correction_targets: list[CorrectionTarget] = field(default_factory=list)


class MemoryReflector:
    def build_reflection(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        domain: str = "project",
        source: str = "reflection",
        confidence: float = 0.78,
        corrects_ids: list[str] | None = None,  # ── Phase 3 ──
    ) -> MemoryRecord:
        pieces = [f"Task: {task_title.strip()}", f"Result: {result_summary.strip()}"]
        if lessons:
            pieces.append(f"Lessons: {lessons.strip()}")
        content = " | ".join(piece for piece in pieces if piece)
        kind = "failure" if any(word in result_summary.lower() for word in ("fail", "blocked", "error", "issue")) else "pattern"
        metadata: dict[str, Any] = {"task_title": task_title.strip()}
        if corrects_ids:
            metadata["corrects_ids"] = list(corrects_ids)
        return MemoryRecord(
            id=new_memory_id("ref"),
            domain=domain,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content,
            confidence=confidence,
            source=source,
            tags=["reflection", task_title.strip().replace(" ", "-").lower()],
            metadata=metadata,
        )

    def build_skill_candidate(
        self,
        *,
        task_title: str,
        steps: Iterable[str],
        domain: str = "project",
        source: str = "reflection",
        confidence: float = 0.72,
    ) -> MemoryRecord:
        step_list = [step.strip() for step in steps if step.strip()]
        step_text = "; ".join(step_list)
        content = f"Reusable skill candidate from {task_title.strip()}: {step_text}"
        return MemoryRecord(
            id=new_memory_id("skill"),
            domain=domain,  # type: ignore[arg-type]
            kind="skill",
            content=content,
            confidence=confidence,
            source=source,
            tags=["skill-candidate", task_title.strip().replace(" ", "-").lower()],
            metadata={"task_title": task_title.strip(), "steps": step_list},
        )

    def extract_corrections(
        self,
        result_summary: str,
        lessons: str | None = None,
    ) -> list[CorrectionTarget]:
        """从反思文本中提取修正指向。

        启发式检测 "之前记错了/actually/纠正" 等模式。
        完整的修正指向由调用方（governor）提供结构化数据。
        """
        combined = f"{result_summary} {lessons or ''}".lower()
        targets: list[CorrectionTarget] = []
        # 检测明确的纠错标记 — 由 governor/system 注入
        if "<<correct:" in combined:
            for segment in combined.split("<<"):
                if segment.startswith("correct:"):
                    parts = segment.split(">>")[0].split(":")
                    if len(parts) >= 2:
                        targets.append(CorrectionTarget(
                            record_id=parts[1].strip(),
                            action="supersede",
                            reason="explicit correction in reflection",
                        ))
        return targets

    def bundle(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        skill_steps: Iterable[str] | None = None,
        domain: str = "project",
        correction_targets: list[CorrectionTarget] | None = None,  # ── Phase 3 ──
        corrects_ids: list[str] | None = None,  # ── Phase 3 ──
    ) -> ReflectionBundle:
        summary = self.build_reflection(
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            domain=domain,
            corrects_ids=corrects_ids,
        )
        skill_candidate = None
        if skill_steps:
            skill_candidate = self.build_skill_candidate(
                task_title=task_title,
                steps=skill_steps,
                domain=domain,
            )
        extracted = self.extract_corrections(result_summary, lessons)
        all_targets = list(correction_targets or []) + extracted
        return ReflectionBundle(
            summary=summary,
            skill_candidate=skill_candidate,
            correction_targets=all_targets,
        )
