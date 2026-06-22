from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import MemoryRecord, new_memory_id


@dataclass(slots=True)
class ReflectionBundle:
    summary: MemoryRecord
    skill_candidate: MemoryRecord | None = None


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
    ) -> MemoryRecord:
        pieces = [f"Task: {task_title.strip()}", f"Result: {result_summary.strip()}"]
        if lessons:
            pieces.append(f"Lessons: {lessons.strip()}")
        content = " | ".join(piece for piece in pieces if piece)
        kind = "failure" if any(word in result_summary.lower() for word in ("fail", "blocked", "error", "issue")) else "pattern"
        return MemoryRecord(
            id=new_memory_id("ref"),
            domain=domain,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content,
            confidence=confidence,
            source=source,
            tags=["reflection", task_title.strip().replace(" ", "-").lower()],
            metadata={"task_title": task_title.strip()},
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

    def bundle(
        self,
        *,
        task_title: str,
        result_summary: str,
        lessons: str | None = None,
        skill_steps: Iterable[str] | None = None,
        domain: str = "project",
    ) -> ReflectionBundle:
        summary = self.build_reflection(
            task_title=task_title,
            result_summary=result_summary,
            lessons=lessons,
            domain=domain,
        )
        skill_candidate = None
        if skill_steps:
            skill_candidate = self.build_skill_candidate(
                task_title=task_title,
                steps=skill_steps,
                domain=domain,
            )
        return ReflectionBundle(summary=summary, skill_candidate=skill_candidate)
