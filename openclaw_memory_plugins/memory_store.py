from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from .memory_embeddings import EmbeddingBackend, cosine_similarity, pack_vector, unpack_vector
from .memory_entities import EntityExtractor, EntityMention
from .types import (
    ADJUDICATION_RULES,
    AGGRESSIVE_MERGE_KINDS,
    CONSERVATIVE_KINDS,
    ConflictRelation,
    MemoryRecord,
    new_memory_id,
    utc_now_iso,
)


class MemoryStore:
    """SQLite-backed persistence for OpenClaw memory records.

    The store keeps a small JSONL compatibility trail for the current plugin
    surface while using SQLite as the source of truth. The implementation is
    intentionally compact but keeps the current caller contracts stable.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir).expanduser() if root_dir is not None else Path.home() / ".openclaw-memory"
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.root_dir / "memories.sqlite3"
        self.records_path = self.root_dir / "records.jsonl"
        self.pending_path = self.root_dir / "pending.jsonl"
        self.reflections_path = self.root_dir / "reflections.jsonl"
        self.feedback_path = self.root_dir / "feedback.jsonl"

        for path in (self.records_path, self.pending_path, self.reflections_path, self.feedback_path):
            path.touch(exist_ok=True)

        self._lock = threading.RLock()
        self._closed = False
        self._fts_enabled = False
        self._embedding_backend = EmbeddingBackend()
        self._entity_extractor = EntityExtractor()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.commit()
            self._conn.close()
            self._closed = True

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def _iter_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return iter(())

        def _reader() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload

        return _reader()

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    # ------------------------------------------------------------------
    # SQLite bootstrap
    # ------------------------------------------------------------------
    def _bootstrap(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    domain TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    trust REAL NOT NULL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    search_text TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_trust ON memories(trust)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at)")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL DEFAULT 'entity',
                    aliases TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, entity_id),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_name ON memory_entities(entity_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_entities_memory_id ON memory_entities(memory_id)")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model ON memory_embeddings(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_backend ON memory_embeddings(backend)")

            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        content,
                        tags,
                        search_text,
                        content='memories',
                        content_rowid='rowid'
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content, tags, search_text)
                        VALUES (new.rowid, new.content, new.tags, new.search_text);
                    END
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, tags, search_text)
                        VALUES ('delete', old.rowid, old.content, old.tags, old.search_text);
                    END
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, tags, search_text)
                        VALUES ('delete', old.rowid, old.content, old.tags, old.search_text);
                        INSERT INTO memories_fts(rowid, content, tags, search_text)
                        VALUES (new.rowid, new.content, new.tags, new.search_text);
                    END
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            self._conn.commit()
            if self._fts_enabled:
                self._rebuild_fts()

    def _rebuild_fts(self) -> None:
        if not self._fts_enabled:
            return
        self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_record(record: MemoryRecord | dict[str, Any] | str, *, default_domain: str = "project") -> MemoryRecord:
        if isinstance(record, MemoryRecord):
            return record
        if isinstance(record, str):
            return MemoryRecord.from_dict({"content": record, "domain": default_domain})
        return MemoryRecord.from_dict(record)

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> MemoryRecord | None:
        if row is None:
            return None
        try:
            tags = list(json.loads(row["tags"]) if row["tags"] else [])
        except (TypeError, ValueError, json.JSONDecodeError):
            tags = []
        try:
            metadata = dict(json.loads(row["metadata"]) if row["metadata"] else {})
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        metadata.setdefault("trust", float(row["trust"]))
        supersedes_ids = list(metadata.pop("supersedes_ids", []) or [])
        contradicts_ids = list(metadata.pop("contradicts_ids", []) or [])
        adjudication = str(metadata.pop("adjudication", "") or "")
        return MemoryRecord(
            id=str(row["id"]),
            domain=str(row["domain"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            confidence=float(row["confidence"]),
            source=str(row["source"]),
            tags=tags,
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            metadata=metadata,
            supersedes_ids=supersedes_ids,
            contradicts_ids=contradicts_ids,
            adjudication=adjudication,
        )

    @staticmethod
    def _record_to_payload(record: MemoryRecord) -> dict[str, Any]:
        payload = record.to_dict()
        payload["search_text"] = " ".join(
            part
            for part in (
                record.content,
                " ".join(record.tags),
                json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
            )
            if part
        ).strip()
        return payload

    @staticmethod
    def _normalize_fragment(fragment: str) -> str:
        text = fragment.strip()
        if not text:
            return ""
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text)
        if not tokens:
            safe = text.replace('"', '""')
            return f'"{safe}"'
        return " OR ".join(tokens)

    @staticmethod
    def _status_matches(record_status: str, status: str | None) -> bool:
        return status is None or record_status == status

    def _rowid_for_id(self, record_id: str) -> int | None:
        row = self._conn.execute("SELECT rowid FROM memories WHERE id = ?", (record_id,)).fetchone()
        return int(row[0]) if row is not None else None

    # ------------------------------------------------------------------
    # Phase 1: 冲突检测与裁决
    # ------------------------------------------------------------------

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Za-z0-9_\-]+", text)}

    @staticmethod
    def _topic_tokens(text: str) -> set[str]:
        """Extract content words used to compare replacement-style decisions."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
            "decided", "does", "do", "done", "for", "from", "had", "has", "have", "i", "if", "in",
            "is", "it", "its", "just", "may", "might", "need", "needs", "not", "of", "on", "or",
            "our", "out", "should", "so", "the", "then", "there", "this", "to", "use", "used", "using",
            "we", "were", "will", "with", "would", "want", "wanted", "instead", "choose", "chose", "agreed",
        }
        tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_\-]+", text)}
        return {token for token in tokens if token not in stopwords and len(token) > 1}

    @classmethod
    def _claim_polarity(cls, text: str) -> str:
        """判断一段文本的断言极性。"""
        positive = re.compile(r"\b(?:prefer|like|love|use|want|need|always|must|should|enabled|true|can)\b", re.I)
        negative = re.compile(r"\b(?:avoid|dislike|hate|never|cannot|can't|won't|disabled|false|shouldn't|mustn't|no|not)\b", re.I)
        if positive.search(text):
            return "positive"
        if negative.search(text):
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

    def _load_entities_for(self, record: MemoryRecord) -> set[str]:
        """加载一条记录的所有实体名。"""
        names: set[str] = set()
        for entity in self.get_record_entities(record.id):
            normalized = str(entity.get("normalized") or "").strip().lower()
            if normalized:
                names.add(normalized)
            name = str(entity.get("name") or "").strip().lower()
            if name and name != normalized:
                names.add(name)
        if not names:
            names.update(
                str(item).strip().lower()
                for item in self._entity_extractor.extract_normalized([record.content, " ".join(record.tags)])
                if str(item).strip()
            )
        return names

    def _match_adjudication_rule(self, candidate_kind: str, existing_kind: str) -> str | None:
        """匹配适用的裁决原则名称，返回 None 表示无规则匹配。"""
        for rule in ADJUDICATION_RULES:
            left_ok = rule.left_kinds is None or candidate_kind in rule.left_kinds
            right_ok = rule.right_kinds is None or existing_kind in rule.right_kinds
            if left_ok and right_ok:
                return rule.name
        return None

    def contradiction_check(
        self,
        candidate: MemoryRecord,
        *,
        domain: str | None = None,
        threshold: float = 0.28,
        limit: int = 5,
    ) -> list[ConflictRelation]:
        """在写入前检测候选记录与已有记录的冲突。

        返回 ConflictRelation 列表，按冲突分数降序排列。
        """
        seed = candidate.content.strip()
        if not seed:
            return []

        pool: list[MemoryRecord] = []
        seen: set[str] = set()

        def _append(records: list[MemoryRecord]) -> None:
            for record in records:
                if record.id in seen or record.id == candidate.id:
                    continue
                seen.add(record.id)
                pool.append(record)

        with self._lock:
            _append(self.find_matching_records(seed, domain=domain, limit=limit * 4, status=None))
            entity_names = self._entity_extractor.extract_names(seed)
            if entity_names:
                _append(self.find_records_for_entities(entity_names, domain=domain, limit=limit * 4, status=None))
            if len(pool) < limit:
                _append(self.list_records(domain=domain, status=None, limit=limit * 6))

        results: list[ConflictRelation] = []
        candidate_entities = self._load_entities_for(candidate)
        candidate_topics = self._topic_tokens(candidate.content)

        for existing in pool[:500]:
            existing_entities = self._load_entities_for(existing)
            existing_topics = self._topic_tokens(existing.content)
            shared = candidate_entities & existing_entities
            union = candidate_entities | existing_entities
            topic_shared: set[str] = set()
            topic_overlap = 0.0
            if candidate.kind == existing.kind == "decision":
                topic_shared = candidate_topics & existing_topics
                topic_union = candidate_topics | existing_topics
                if topic_union:
                    topic_overlap = len(topic_shared) / len(topic_union)

            if not shared and not (candidate.kind == existing.kind == "decision" and topic_overlap > 0.0):
                continue

            entity_overlap = len(shared) / len(union) if union else 0.0
            effective_overlap = entity_overlap
            if candidate.kind == existing.kind == "decision":
                effective_overlap = max(entity_overlap, topic_overlap)

            content_sim = self._content_similarity(candidate, existing)
            polarity_c = self._claim_polarity(candidate.content)
            polarity_e = self._claim_polarity(existing.content)

            contradiction_score = effective_overlap * (1.0 - content_sim)
            if candidate.kind == existing.kind == "decision" and topic_overlap > 0.0:
                contradiction_score = max(contradiction_score, 0.35 + (topic_overlap * 0.25))
            if polarity_c != "neutral" and polarity_e != "neutral" and polarity_c != polarity_e:
                contradiction_score += 0.18
            if (
                candidate.kind != existing.kind
                and {candidate.kind, existing.kind} & {"decision", "preference", "fact", "failure"}
            ):
                contradiction_score += 0.05

            if effective_overlap < 0.25 and not (candidate.kind == existing.kind == "decision" and topic_overlap > 0.0):
                continue

            if contradiction_score < threshold and not (candidate.kind == existing.kind == "decision" and topic_overlap > 0.0):
                continue

            rule_name = self._match_adjudication_rule(candidate.kind, existing.kind) or ""
            suggested = self._adjudicate_action(candidate, existing, rule_name, contradiction_score)

            results.append(ConflictRelation(
                candidate_id=candidate.id,
                existing_id=existing.id,
                score=min(1.0, contradiction_score),
                entity_overlap=effective_overlap,
                content_similarity=content_sim,
                polarity_left=polarity_c,
                polarity_right=polarity_e,
                shared_entities=sorted(shared or topic_shared),
                rule_applied=rule_name,
                suggested_action=suggested,
            ))

        results.sort(key=lambda r: (r.score, r.entity_overlap), reverse=True)
        return results[:limit]

    def _adjudicate_action(
        self,
        candidate: MemoryRecord,
        existing: MemoryRecord,
        rule_name: str,
        score: float,
    ) -> str:
        """根据裁决原则和信任/置信度决定写入动作: write/stage/supersede/reject。

        优先级:
          1. 匹配到的 AdjudicationRule
          2. 分数+信任组合: 高冲突+低信任旧记录 → supersede
          3. 默认保守: stage
        """
        # 检查是否有 explicit 裁决规则匹配
        for rule in ADJUDICATION_RULES:
            if rule.name == rule_name:
                if rule.recommended_action == "supersede":
                    return "supersede"
                if rule.recommended_action == "stage":
                    return "stage"
                return rule.recommended_action

        # 基于信任和置信度判断
        existing_trust = float(existing.metadata.get("trust", 0.5)) if isinstance(existing.metadata, dict) else 0.5
        candidate_is_preferred = candidate.confidence >= 0.75 and existing.confidence < 0.6
        existing_is_stale = existing_trust < 0.3

        if score >= 0.4 and candidate.kind == "decision" and existing.kind == "fact":
            return "supersede"
        if score >= 0.4 and candidate.kind == "fact" and existing.kind == "decision":
            return "stage"
        if score >= 0.4 and candidate.kind == "preference" and existing.kind == "fact":
            return "supersede"
        if score >= 0.4 and candidate.kind == "fact" and existing.kind == "preference":
            return "stage"

        if score >= 0.6 and (candidate_is_preferred or existing_is_stale):
            return "supersede"
        if score >= 0.4 and candidate.kind in CONSERVATIVE_KINDS and existing.kind in CONSERVATIVE_KINDS:
            return "stage"
        if score >= 0.5 and candidate.confidence < 0.4:
            return "reject"
        return "write"

    def supersede_record(self, old_id: str, new_id: str, *, adjudication: str = "") -> MemoryRecord | None:
        """将旧记录标记为 superseded，记录被谁覆盖。"""
        with self._lock:
            existing = self.find_record(old_id)
            if existing is None:
                return None
            existing.status = "superseded"
            existing.touch()
            metadata = dict(existing.metadata)
            metadata["superseded_by"] = new_id
            if adjudication:
                metadata["adjudication"] = adjudication
            existing.metadata = metadata
            return self.upsert_record(existing, _skip_contradiction=True)

    def _upsert_entity(self, mention: EntityMention) -> tuple[str, str]:
        now = utc_now_iso()
        row = self._conn.execute("SELECT id, name, aliases FROM entities WHERE normalized = ?", (mention.normalized,)).fetchone()
        if row is None:
            entity_id = new_memory_id("ent")
            self._conn.execute(
                """
                INSERT INTO entities (id, name, normalized, entity_type, aliases, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    mention.name,
                    mention.normalized,
                    mention.entity_type,
                    json.dumps([], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            return entity_id, mention.name

        entity_id = str(row[0])
        name = str(row[1] or mention.name)
        try:
            aliases = list(json.loads(row[2]) if row[2] else [])
        except (TypeError, ValueError, json.JSONDecodeError):
            aliases = []
        if mention.name not in aliases and mention.name != name:
            aliases.append(mention.name)
        if mention.name != name:
            name = mention.name
        self._conn.execute(
            "UPDATE entities SET name = ?, entity_type = ?, aliases = ?, updated_at = ? WHERE id = ?",
            (name, mention.entity_type, json.dumps(aliases, ensure_ascii=False), now, entity_id),
        )
        return entity_id, name

    def _sync_entity_links(self, record: MemoryRecord) -> list[EntityMention]:
        # Keep entity links grounded in user-visible text. Metadata is useful for
        # filtering, but feeding raw JSON into extraction produces noisy entities
        # like trust/stage_reason that do not help retrieval or contradiction checks.
        mentions = self._entity_extractor.extract([record.content, " ".join(record.tags)])
        self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (record.id,))
        now = utc_now_iso()
        for mention in mentions:
            entity_id, entity_name = self._upsert_entity(mention)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_entities (memory_id, entity_id, entity_name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (record.id, entity_id, entity_name, now),
            )
        return mentions

    def _embedding_text(self, record: MemoryRecord) -> str:
        return " ".join(part for part in (record.domain, record.kind, record.content, " ".join(record.tags)) if part).strip()

    def _sync_embedding(self, record: MemoryRecord) -> dict[str, Any] | None:
        payload = self._embedding_backend.encode(self._embedding_text(record), kind=f"{record.domain}/{record.kind}")
        if payload is None or not payload.vector:
            self._conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (record.id,))
            return None
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO memory_embeddings (memory_id, model_name, backend, dimension, vector, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                model_name=excluded.model_name,
                backend=excluded.backend,
                dimension=excluded.dimension,
                vector=excluded.vector,
                updated_at=excluded.updated_at
            """,
            (record.id, payload.model_name, payload.backend, payload.dimension, pack_vector(payload.vector), now, now),
        )
        return {
            "memory_id": record.id,
            "model_name": payload.model_name,
            "backend": payload.backend,
            "dimension": payload.dimension,
        }

    def get_record_embedding(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT memory_id, model_name, backend, dimension, vector, created_at, updated_at FROM memory_embeddings WHERE memory_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        vector = unpack_vector(row[4])
        return {
            "memory_id": str(row[0]),
            "model_name": str(row[1]),
            "backend": str(row[2]),
            "dimension": int(row[3]),
            "vector": list(vector),
            "created_at": str(row[5]),
            "updated_at": str(row[6]),
        }

    def get_record_entities(self, record_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.id, e.name, e.normalized, e.entity_type, e.aliases, me.created_at
                FROM memory_entities me
                JOIN entities e ON e.id = me.entity_id
                WHERE me.memory_id = ?
                ORDER BY me.created_at ASC
                """,
                (record_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                aliases = list(json.loads(row[4]) if row[4] else [])
            except (TypeError, ValueError, json.JSONDecodeError):
                aliases = []
            result.append(
                {
                    "id": str(row[0]),
                    "name": str(row[1]),
                    "normalized": str(row[2]),
                    "entity_type": str(row[3]),
                    "aliases": aliases,
                    "created_at": str(row[5]),
                }
            )
        return result

    def find_records_for_entities(
        self,
        entity_names: Iterable[str],
        *,
        domain: str | None = None,
        limit: int = 5,
        status: str | None = "active",
    ) -> list[MemoryRecord]:
        normalized: list[str] = []
        seen: set[str] = set()
        for name in entity_names:
            key = re.sub(r"\s+", " ", str(name).strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        sql = [
            "SELECT DISTINCT m.* FROM memories m",
            "JOIN memory_entities me ON me.memory_id = m.id",
            "JOIN entities e ON e.id = me.entity_id",
            f"WHERE e.normalized IN ({placeholders})",
        ]
        params: list[Any] = list(normalized)
        if domain:
            sql.append("AND m.domain = ?")
            params.append(domain)
        if status is not None:
            sql.append("AND m.status = ?")
            params.append(status)
        sql.append("ORDER BY m.updated_at DESC, m.created_at DESC LIMIT ?")
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), params).fetchall()
        return [record for row in rows if (record := self._row_to_record(row)) is not None]

    def find_semantic_records(
        self,
        query: str,
        *,
        domain: str | None = None,
        limit: int = 5,
        status: str | None = "active",
    ) -> list[tuple[MemoryRecord, float]]:
        query = query.strip()
        if not query:
            return []
        query_embedding = self._embedding_backend.encode(query, kind="query")
        if query_embedding is None or not query_embedding.vector:
            return []
        sql = [
            "SELECT m.id, m.domain, m.kind, m.content, m.confidence, m.source, m.trust, m.status, m.created_at, m.updated_at, m.tags, m.metadata, m.search_text, e.vector AS vector FROM memories m",
            "JOIN memory_embeddings e ON e.memory_id = m.id",
            "WHERE 1=1",
        ]
        params: list[Any] = []
        if domain:
            sql.append("AND m.domain = ?")
            params.append(domain)
        if status is not None:
            sql.append("AND m.status = ?")
            params.append(status)
        sql.append("ORDER BY m.updated_at DESC, m.created_at DESC LIMIT ?")
        params.append(max(int(limit) * 10, 50))
        with self._lock:
            rows = self._conn.execute(" ".join(sql), params).fetchall()
        ranked: list[tuple[MemoryRecord, float]] = []
        query_vector = tuple(query_embedding.vector)
        for row in rows:
            record = self._row_to_record(row)
            if record is None:
                continue
            vector = unpack_vector(row["vector"]) if row["vector"] is not None else tuple()
            if not vector:
                continue
            similarity = cosine_similarity(query_vector, vector)
            ranked.append((record, float(similarity)))
        ranked.sort(key=lambda item: (item[1], item[0].updated_at, item[0].confidence), reverse=True)
        return ranked[:limit]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def save_record(self, record: MemoryRecord | dict[str, Any] | str) -> MemoryRecord:
        return self.upsert_record(record)

    def upsert_record(
        self,
        record: MemoryRecord | dict[str, Any] | str,
        *,
        _skip_contradiction: bool = False,
    ) -> MemoryRecord:
        item = self._coerce_record(record)
        if not item.id:
            item.id = new_memory_id()

        # ── Phase 1: 摄入前冲突检测 ──────────────────────────────────
        contradictions: list[ConflictRelation] = []
        if not _skip_contradiction and item.status not in ("superseded", "removed", "disputed"):
            contradictions = self.contradiction_check(item, domain=item.domain, limit=3)
            # 找到最严重的冲突裁决
            if contradictions:
                top = contradictions[0]
                if top.suggested_action == "reject":
                    # reject: 不写入，直接返回候选（标记 metadata 说明原因）
                    item.metadata = dict(item.metadata)
                    item.metadata["rejected"] = True
                    item.metadata["rejection_reason"] = f"conflict with {top.existing_id} (score={top.score:.2f})"
                    self._append_jsonl(self.records_path, {
                        "action": "reject", "record": item.to_dict(), "conflict": top.to_dict(),
                    })
                    return item

                if top.suggested_action == "supersede":
                    # supersede: 先标记冲突的旧记录，再写入新记录
                    for conflict in contradictions:
                        if conflict.suggested_action == "supersede":
                            self.supersede_record(
                                conflict.existing_id, item.id,
                                adjudication=f"superseded by {item.kind} '{item.content[:60]}' "
                                             f"(conflict_score={conflict.score:.2f})",
                            )
                            item.supersedes_ids = list(item.supersedes_ids) + [conflict.existing_id]

                if top.suggested_action == "stage":
                    item.status = "staged"

                # 记录所有冲突 ID
                item.contradicts_ids = list(set(
                    list(item.contradicts_ids) + [c.existing_id for c in contradictions
                                                  if c.suggested_action != "supersede"]
                ))

        # 确保冲突关系字段同步到 metadata（持久化到 SQLite）
        item.metadata = dict(item.metadata)
        if item.supersedes_ids:
            item.metadata["supersedes_ids"] = list(item.supersedes_ids)
        if item.contradicts_ids:
            item.metadata["contradicts_ids"] = list(item.contradicts_ids)
        if item.adjudication:
            item.metadata["adjudication"] = item.adjudication

        # ── 正常写入 ──────────────────────────────────────────────────
        with self._lock:
            existing = self.find_record(item.id)
            if existing is not None:
                item.created_at = existing.created_at
            if not item.created_at:
                item.created_at = utc_now_iso()
            item.touch()
            payload = self._record_to_payload(item)
            metadata = dict(item.metadata)
            metadata.setdefault("trust", float(metadata.get("trust", 0.5)))
            payload_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            payload_tags = json.dumps(list(item.tags), ensure_ascii=False)
            trust = float(metadata.get("trust", 0.5))
            self._conn.execute(
                """
                INSERT INTO memories (id, domain, kind, content, confidence, source, trust, status, created_at, updated_at, tags, metadata, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    domain=excluded.domain,
                    kind=excluded.kind,
                    content=excluded.content,
                    confidence=excluded.confidence,
                    source=excluded.source,
                    trust=excluded.trust,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    tags=excluded.tags,
                    metadata=excluded.metadata,
                    search_text=excluded.search_text
                """,
                (
                    item.id,
                    item.domain,
                    item.kind,
                    item.content,
                    float(item.confidence),
                    item.source,
                    trust,
                    item.status,
                    item.created_at,
                    item.updated_at,
                    payload_tags,
                    payload_metadata,
                    payload["search_text"],
                ),
            )
            self._sync_entity_links(item)
            self._sync_embedding(item)
            self._conn.commit()
            stored = self.find_record(item.id) or item
            action = "reject" if contradictions and contradictions[0].suggested_action == "reject" else "save"
            entry: dict[str, Any] = {"action": action, "record": stored.to_dict()}
            if contradictions:
                entry["contradictions"] = [c.to_dict() for c in contradictions]
            self._append_jsonl(self.records_path, entry)
            return stored

    def stage_record(self, record: MemoryRecord | dict[str, Any] | str, *, reason: str = "score gate", score: float | None = None) -> MemoryRecord:
        item = self._coerce_record(record)
        item.status = "staged"
        item.metadata = dict(item.metadata)
        item.metadata.update({"staged": True, "stage_reason": reason})
        if score is not None:
            item.metadata["stage_score"] = round(float(score), 3)
        stored = self.upsert_record(item)
        self._append_jsonl(self.pending_path, {"action": "stage", "record": stored.to_dict(), "reason": reason, "score": score})
        return stored

    def save_reflection(self, record: MemoryRecord | dict[str, Any] | str) -> MemoryRecord:
        item = self._coerce_record(record)
        item.metadata = dict(item.metadata)
        item.metadata.setdefault("reflection", True)
        stored = self.upsert_record(item)
        self._append_jsonl(self.reflections_path, {"action": "reflection", "record": stored.to_dict()})
        return stored

    def find_record(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (record_id,)).fetchone()
            return self._row_to_record(row)

    def list_records(
        self,
        *,
        domain: str | None = None,
        status: str | None = "active",
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        query = ["SELECT * FROM memories WHERE 1=1"]
        params: list[Any] = []
        if domain:
            query.append("AND domain = ?")
            params.append(domain)
        if status is not None:
            query.append("AND status = ?")
            params.append(status)
        query.append("ORDER BY updated_at DESC, created_at DESC")
        if limit is not None:
            query.append("LIMIT ?")
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(" ".join(query), params).fetchall()
        return [record for row in rows if (record := self._row_to_record(row)) is not None]

    def find_matching_records(
        self,
        fragment: str,
        *,
        domain: str | None = None,
        limit: int = 5,
        status: str | None = "active",
    ) -> list[MemoryRecord]:
        fragment = fragment.strip()
        if not fragment:
            return self.list_records(domain=domain, status=status, limit=limit)

        results: list[MemoryRecord] = []
        seen: set[str] = set()

        def _append_records(records: Iterable[MemoryRecord]) -> None:
            for record in records:
                if record.id in seen:
                    continue
                seen.add(record.id)
                results.append(record)
                if len(results) >= limit:
                    return

        with self._lock:
            if self._fts_enabled:
                query = self._normalize_fragment(fragment)
                sql = [
                    "SELECT m.* FROM memories_fts f",
                    "JOIN memories m ON m.rowid = f.rowid",
                    "WHERE memories_fts MATCH ?",
                ]
                params: list[Any] = [query]
                if domain:
                    sql.append("AND m.domain = ?")
                    params.append(domain)
                if status is not None:
                    sql.append("AND m.status = ?")
                    params.append(status)
                sql.append("ORDER BY bm25(memories_fts), m.updated_at DESC, m.created_at DESC LIMIT ?")
                params.append(int(limit) * 3)
                rows = self._conn.execute(" ".join(sql), params).fetchall()
                _append_records([record for row in rows if (record := self._row_to_record(row)) is not None])

            if len(results) < limit:
                entity_names = self._entity_extractor.extract_names(fragment)
                if entity_names:
                    _append_records(self.find_records_for_entities(entity_names, domain=domain, limit=limit * 2, status=status))

            if len(results) < limit:
                needle = f"%{fragment.lower()}%"
                sql = ["SELECT * FROM memories WHERE (lower(content) LIKE ? OR lower(tags) LIKE ? OR lower(search_text) LIKE ?)"]
                params = [needle, needle, needle]
                if domain:
                    sql.append("AND domain = ?")
                    params.append(domain)
                if status is not None:
                    sql.append("AND status = ?")
                    params.append(status)
                sql.append("ORDER BY updated_at DESC, created_at DESC LIMIT ?")
                params.append(int(limit) * 2)
                rows = self._conn.execute(" ".join(sql), params).fetchall()
                _append_records([record for row in rows if (record := self._row_to_record(row)) is not None])

        return results[:limit]

    def deactivate_matching_records(
        self,
        fragment: str,
        *,
        domain: str | None = None,
        status: str = "removed",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        matches = self.find_matching_records(fragment, domain=domain, limit=limit, status=None)
        updated: list[MemoryRecord] = []
        with self._lock:
            for record in matches:
                record.status = status
                record.touch()
                updated_record = self.upsert_record(record)
                updated.append(updated_record)
        return updated

    def record_feedback(
        self,
        record_id: str,
        *,
        helpful: bool,
        note: str = "",
        weight: float = 0.05,
    ) -> MemoryRecord | None:
        with self._lock:
            record = self.find_record(record_id)
            if record is None:
                return None
            delta = abs(float(weight)) if helpful else -abs(float(weight)) * 2.0
            metadata = dict(record.metadata)
            trust = float(metadata.get("trust", 0.5))
            trust = max(0.0, min(1.0, trust + delta))
            metadata.update(
                {
                    "trust": round(trust, 3),
                    "last_feedback_helpful": bool(helpful),
                    "last_feedback_note": note,
                    "last_feedback_weight": round(float(weight), 3),
                }
            )
            record.metadata = metadata
            record.touch()
            record = self.upsert_record(record)
            self._append_jsonl(
                self.feedback_path,
                {
                    "action": "feedback",
                    "record_id": record_id,
                    "helpful": helpful,
                    "note": note,
                    "weight": round(float(weight), 3),
                    "trust": metadata["trust"],
                },
            )
            return record

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def count_records(self, *, domain: str | None = None, status: str | None = None) -> int:
        query = ["SELECT COUNT(*) AS count FROM memories WHERE 1=1"]
        params: list[Any] = []
        if domain:
            query.append("AND domain = ?")
            params.append(domain)
        if status is not None:
            query.append("AND status = ?")
            params.append(status)
        with self._lock:
            row = self._conn.execute(" ".join(query), params).fetchone()
        return int(row[0] if row is not None else 0)

    def dump_all(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.list_records(status=None)]
