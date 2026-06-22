from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from .types import MemoryRecord, new_memory_id

DecisionAction = Literal["auto_write", "stage", "drop"]


@dataclass(slots=True)
class ScoreDecision:
    score: float
    action: DecisionAction
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "action": self.action,
            "reasons": list(self.reasons),
        }


class MemoryScorer:
    KIND_BASE = {
        "preference": 0.84,
        "decision": 0.8,
        "fact": 0.7,
        "pattern": 0.76,
        "failure": 0.74,
        "skill": 0.78,
        "note": 0.42,
    }

    HIGH_VALUE_HINTS = (
        "always",
        "never",
        "prefer",
        "remember",
        "important",
        "must",
        "should",
        "rule",
        "decision",
        "fail",
        "error",
        "bug",
        "lesson",
        "works",
        "repeat",
        "use",
    )

    TRANSIENT_HINTS = (
        "today",
        "now",
        "right now",
        "maybe",
        "probably",
        "might",
        "temporary",
        "just",
        "moment",
    )

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def classify(self, candidate: MemoryRecord | dict) -> ScoreDecision:
        record = candidate if isinstance(candidate, MemoryRecord) else MemoryRecord.from_dict(candidate)
        text = self._clean_text(record.content)
        score = self.KIND_BASE.get(record.kind, 0.5)
        reasons: list[str] = [f"kind base: {record.kind}"]

        if record.domain == "user":
            score += 0.08
            reasons.append("user domain bonus")
        elif record.domain == "project":
            score += 0.05
            reasons.append("project domain bonus")

        for hint in self.HIGH_VALUE_HINTS:
            if hint in text:
                score += 0.05
                reasons.append(f"high-value hint: {hint}")

        for hint in self.TRANSIENT_HINTS:
            if hint in text:
                score -= 0.07
                reasons.append(f"transient hint: {hint}")

        if len(text) < 24:
            score -= 0.12
            reasons.append("too short")
        elif len(text) > 320:
            score += 0.06
            reasons.append("rich detail")

        if record.source in {"conversation", "task_result", "reflection"}:
            score += 0.03
            reasons.append(f"source bonus: {record.source}")

        if record.confidence >= 0.9:
            score += 0.05
            reasons.append("confidence bonus")
        elif record.confidence <= 0.35:
            score -= 0.08
            reasons.append("low-confidence penalty")

        if record.kind in {"decision", "preference"} and any(hint in text for hint in ("prefer", "always", "never", "must")):
            score += 0.05
            reasons.append("explicit rule bonus")

        if record.kind == "failure" and any(hint in text for hint in ("failed", "error", "bug", "blocked")):
            score += 0.08
            reasons.append("failure learning bonus")

        score = max(0.0, min(score, 1.0))
        if score >= 0.8:
            action: DecisionAction = "auto_write"
        elif score >= 0.55:
            action = "stage"
        else:
            action = "drop"
        return ScoreDecision(score=score, action=action, reasons=reasons)

    def score(self, candidate: MemoryRecord | dict) -> float:
        return self.classify(candidate).score

    def build_candidate(
        self,
        *,
        content: str,
        domain: str = "project",
        kind: str = "note",
        source: str = "conversation",
        confidence: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        return MemoryRecord(
            id=new_memory_id(),
            domain=domain,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content.strip(),
            confidence=confidence,
            source=source,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
