from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable

from .memory_store import MemoryStore
from .types import MemoryRecord

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass(slots=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "record": self.record.to_dict(),
        }


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

        if record.status != "active":
            score *= 0.5
            reasons.append("inactive penalty")

        score = max(0.0, min(score, 1.0))
        return RetrievedMemory(record=record, score=score, reasons=reasons)

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
            lines.append(f"- [{record.domain}/{record.kind}] {record.content}")
            if len(lines) >= limit:
                break
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
