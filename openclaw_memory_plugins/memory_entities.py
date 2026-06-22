from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_QUOTED_RE = re.compile(r'"([^"]{2,80})"')
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b")
_CAMEL_WORD_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,10}\b")
_AKA_RE = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9_\- ]{1,60}?)\s+(?:aka|also known as|alias|called)\s+([A-Za-z0-9][A-Za-z0-9_\- ]{1,60}?)\b", re.I)


@dataclass(slots=True)
class EntityMention:
    name: str
    normalized: str
    entity_type: str = "entity"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "normalized": self.normalized,
            "entity_type": self.entity_type,
        }


class EntityExtractor:
    """Lightweight entity extractor for Hermes-style memory association."""

    def _normalize_name(self, value: str) -> str:
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        text = text.replace("_", " ").replace("/", " ")
        text = re.sub(r"\s+", " ", text).strip(" \t\r\n,.;:!?()[]{}<>")
        return text

    def _entity_type(self, name: str) -> str:
        lowered = name.lower()
        if lowered.startswith("openclaw") or "memory" in lowered or "plugin" in lowered or "governor" in lowered:
            return "component"
        if name.isupper() and len(name) <= 10:
            return "acronym"
        if any(part[0].isupper() for part in name.split() if part):
            return "name"
        return "entity"

    def _add(self, items: list[EntityMention], seen: set[str], raw: str) -> None:
        cleaned = self._normalize_name(raw)
        if len(cleaned) < 2:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        items.append(EntityMention(name=cleaned, normalized=key, entity_type=self._entity_type(cleaned)))

    def extract(self, text: str | Iterable[str]) -> list[EntityMention]:
        if isinstance(text, str):
            source = text
        else:
            source = "\n".join(part for part in text if part)
        if not source.strip():
            return []

        mentions: list[EntityMention] = []
        seen: set[str] = set()

        for match in _QUOTED_RE.findall(source):
            self._add(mentions, seen, match)
        for left, right in _AKA_RE.findall(source):
            self._add(mentions, seen, left)
            self._add(mentions, seen, right)
        for match in _CAPITALIZED_RE.findall(source):
            self._add(mentions, seen, match)
        for match in _CAMEL_WORD_RE.findall(source):
            self._add(mentions, seen, match)
        for match in _ACRONYM_RE.findall(source):
            self._add(mentions, seen, match)

        return mentions

    def extract_names(self, text: str | Iterable[str]) -> list[str]:
        return [mention.name for mention in self.extract(text)]

    def extract_normalized(self, text: str | Iterable[str]) -> list[str]:
        return [mention.normalized for mention in self.extract(text)]
