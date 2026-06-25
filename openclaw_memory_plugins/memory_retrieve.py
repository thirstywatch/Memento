from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from .memory_store import MemoryStore
from .types import MemoryRecord, MemoryStatus

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass(slots=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float
    reasons: list[str]

    # ── Phase 2: 冲突感知字段 ──
    contradicts: list[dict[str, Any]] = field(default_factory=list)
    superseded_by: str = ""
    status_snapshot: str = "active"
    trust_snapshot: float = 0.5

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "record": self.record.to_dict(),
            "status": self.status_snapshot,
            "trust": round(self.trust_snapshot, 3),
        }
        if self.contradicts:
            d["contradicts"] = list(self.contradicts)
        if self.superseded_by:
            d["superseded_by"] = self.superseded_by
        return d


class MemoryRetriever:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()

    def _tokenize(self, text: str) -> Counter[str]:
        tokens = [token.lower() for token in _WORD_RE.findall(text)]
        return Counter(tokens)

    def _parse_timestamp(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _temporal_decay(self, updated_at: str, *, half_life_days: float = 90.0) -> tuple[float, str | None]:
        if half_life_days <= 0:
            return 1.0, None
        timestamp = self._parse_timestamp(updated_at)
        if timestamp is None:
            return 1.0, None
        now = datetime.now(timezone.utc)
        age_days = max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / half_life_days)
        return max(0.15, min(decay, 1.0)), f"time decay: {decay:.2f}"

    def _trust_value(self, record: MemoryRecord) -> float:
        trust = 0.5
        if isinstance(record.metadata, dict):
            try:
                trust = float(record.metadata.get("trust", trust))
            except (TypeError, ValueError):
                trust = 0.5
        return max(0.0, min(trust, 1.0))

    def _score_record(self, query_tokens: Counter[str], record: MemoryRecord) -> RetrievedMemory:
        content_tokens = self._tokenize(record.content)
        tags_tokens = self._tokenize(" ".join(record.tags))
        all_tokens = content_tokens + tags_tokens
        overlap = set(query_tokens) & set(all_tokens)
        reasons: list[str] = []

        score = 0.0
        if overlap:
            overlap_bonus = min(len(overlap) * 0.18, 0.54)
            score += overlap_bonus
            reasons.append(f"token overlap: {', '.join(sorted(overlap))}")

        confidence_bonus = max(0.0, min(float(record.confidence), 1.0)) * 0.16
        score += confidence_bonus
        if confidence_bonus:
            reasons.append(f"confidence bonus: {confidence_bonus:.2f}")

        trust = self._trust_value(record)
        trust_snapshot = trust
        trust_bonus = (trust - 0.5) * 0.3
        score += trust_bonus
        if trust_bonus:
            reasons.append(f"trust bonus: {trust_bonus:.2f}")

        if record.kind in {"decision", "preference", "pattern"}:
            score += 0.12
            reasons.append(f"kind bonus: {record.kind}")

        if record.domain == "user":
            score += 0.08
            reasons.append("user domain bonus")
        elif record.domain == "project" and trust >= 0.75:
            score += 0.05
            reasons.append("project trust bonus")

        decay, decay_reason = self._temporal_decay(record.updated_at)
        score *= decay
        if decay_reason:
            reasons.append(decay_reason)

        # ── Phase 2: 状态感知降权 ──
        status_snapshot: MemoryStatus = record.status
        if record.status == "superseded":
            score *= 0.2
            reasons.append("superseded penalty (0.2x)")
        elif record.status == "disputed":
            score *= 0.4
            reasons.append("disputed penalty (0.4x)")
        elif record.status == "removed":
            score *= 0.1
            reasons.append("removed penalty (0.1x)")
        elif record.status == "staged":
            score *= 0.7
            reasons.append("staged penalty (0.7x)")
        elif record.status != "active":
            score *= 0.5
            reasons.append("inactive penalty")

        # ── Phase 2: 加载冲突关系 ──
        contradicts_data: list[dict[str, Any]] = []
        superseded_by = ""
        if record.contradicts_ids:
            for cid in record.contradicts_ids:
                c_record = self.store.find_record(cid)
                if c_record:
                    contradicts_data.append({
                        "record_id": cid,
                        "status": c_record.status,
                        "content": c_record.content[:80],
                    })
        if record.supersedes_ids:
            for sid in record.supersedes_ids:
                s_record = self.store.find_record(sid)
                if s_record:
                    contradicts_data.append({
                        "record_id": sid,
                        "status": s_record.status,
                        "superseded": True,
                        "content": s_record.content[:80],
                    })
        if record.status == "superseded" and record.metadata:
            superseded_by = str(record.metadata.get("superseded_by", "") or "")

        score = max(0.0, min(score, 1.0))
        return RetrievedMemory(
            record=record,
            score=score,
            reasons=reasons,
            contradicts=contradicts_data,
            superseded_by=superseded_by,
            status_snapshot=status_snapshot,
            trust_snapshot=trust_snapshot,
        )

    def _candidate_pool(self, query: str, *, domain: str | None = None, limit: int = 5) -> tuple[list[MemoryRecord], list[tuple[MemoryRecord, float]]]:
        fragment_hits = self.store.find_matching_records(query, domain=domain, limit=max(limit * 3, 8), status=None)
        candidates: list[MemoryRecord] = list(fragment_hits)
        seen = {record.id for record in candidates}

        semantic_hits: list[tuple[MemoryRecord, float]] = []
        if hasattr(self.store, "find_semantic_records"):
            semantic_hits = self.store.find_semantic_records(query, domain=domain, limit=max(limit * 3, 8), status=None)
            for record, _similarity in semantic_hits:
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)

        entity_names = self.store._entity_extractor.extract_names(query) if hasattr(self.store, "_entity_extractor") else []
        if entity_names and hasattr(self.store, "find_records_for_entities"):
            for record in self.store.find_records_for_entities(entity_names, domain=domain, limit=max(limit * 2, 6), status=None):
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)

        if len(candidates) < limit:
            for record in self.store.list_records(domain=domain, status=None, limit=max(limit * 4, 12)):
                if record.id in seen:
                    continue
                seen.add(record.id)
                candidates.append(record)
                if len(candidates) >= max(limit * 4, 12):
                    break
        return candidates, semantic_hits

    def search(self, query: str, *, domain: str | None = None, limit: int = 5) -> list[RetrievedMemory]:
        query_tokens = self._tokenize(query)
        candidates, semantic_hits = self._candidate_pool(query, domain=domain, limit=limit)
        ranked = [self._score_record(query_tokens, record) for record in candidates]
        semantic_hits = {record.id: similarity for record, similarity in semantic_hits}
        for item in ranked:
            similarity = semantic_hits.get(item.record.id)
            if similarity is not None:
                item.score = max(item.score, min(1.0, 0.5 + similarity * 0.5))
                if similarity > 0.65:
                    item.reasons.append(f"embedding similarity: {similarity:.2f}")
        ranked.sort(key=lambda item: (item.score, item.record.updated_at, item.record.confidence), reverse=True)
        return ranked[:limit]

    def build_memory_pack(self, query: str, *, domain: str | None = None, limit: int = 3, max_chars: int = 800) -> str:
        matches = self.search(query, domain=domain, limit=limit)
        lines: list[str] = []
        for item in matches:
            record = item.record
            prefix = "[active]"
            if item.status_snapshot == "superseded":
                prefix = "[superseded]"
            elif item.status_snapshot == "disputed":
                prefix = "[disputed]"
            elif item.status_snapshot == "staged":
                prefix = "[staged]"
            elif item.status_snapshot == "removed":
                continue  # 不展示已删除的
            lines.append(f"- {prefix} [{record.domain}/{record.kind}] {record.content}")

            # ── Phase 2: 冲突语义 ──
            if item.contradicts:
                for cx in item.contradicts[:2]:
                    cx_status = cx.get("status", "")
                    if cx.get("superseded"):
                        lines.append(f"  [已覆盖] 旧记录 {cx['record_id'][:8]} ({cx_status})")
                    else:
                        lines.append(f"  [冲突] 与 {cx['record_id'][:8]} ({cx_status}) 存在冲突 — {cx.get('content','')[:50]}")
            if item.superseded_by:
                lines.append(f"  [已被覆盖] 此记录已被 {item.superseded_by[:8]} 替代")

            if len(lines) >= limit * 3:
                break

        if not lines:
            return ""

        pack = "\n".join(lines)
        if len(pack) > max_chars:
            return pack[: max_chars - 3] + "..."
        return pack

    def search_many(self, queries: Iterable[str], *, domain: str | None = None, limit: int = 5) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        seen: set[str] = set()
        for query in queries:
            for item in self.search(query, domain=domain, limit=limit):
                if item.record.id in seen:
                    continue
                seen.add(item.record.id)
                results.append(item)
        results.sort(key=lambda item: (item.score, item.record.updated_at), reverse=True)
        return results[:limit]
