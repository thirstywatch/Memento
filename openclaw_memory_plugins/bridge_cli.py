from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .register import build_runtime

_COMMANDS = {
    "initialize",
    "prefetch",
    "prefetch_turn",
    "sync_turn",
    "on_pre_compress",
    "on_session_end",
    "handle_tool_call",
    "corpus_search",
    "corpus_get",
    "state",
    "system_prompt_block",
    "bundle",
}


def _read_payload(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if isinstance(data, dict):
        return data
    raise ValueError("payload must be a JSON object")


def _runtime_from_payload(payload: dict[str, Any]):
    return build_runtime(
        storage_root=payload.get("storage_root"),
        session_id=str(payload.get("session_id") or ""),
        platform=str(payload.get("platform") or "openclaw"),
        auto_extract=bool(payload.get("auto_extract", True)),
        auto_mirror=bool(payload.get("auto_mirror", True)),
        openclaw_home=payload.get("openclaw_home"),
        workspace_dir=payload.get("workspace_dir"),
        self_improving_dir=payload.get("self_improving_dir"),
        proactivity_dir=payload.get("proactivity_dir"),
    )


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "body", "message"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        parts = [_message_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _read_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        messages.append(item)
    return messages


def _last_message_text(messages: list[dict[str, Any]], *, role: str) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").strip().lower() != role:
            continue
        text = _message_text(message.get("content"))
        if text:
            return text
    return ""


def _result(value: Any) -> dict[str, Any]:
    return {"ok": True, "result": value}


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _run_command(command: str, payload: dict[str, Any]) -> Any:
    runtime = _runtime_from_payload(payload)
    session_id = str(payload.get("session_id") or payload.get("sessionKey") or "")
    if command == "initialize":
        if session_id:
            runtime.initialize(
                session_id,
                openclaw_home=payload.get("openclaw_home"),
                workspace_dir=payload.get("workspace_dir"),
                self_improving_dir=payload.get("self_improving_dir"),
                proactivity_dir=payload.get("proactivity_dir"),
                platform=str(payload.get("platform") or "openclaw"),
            )
        return runtime.build_bundle(message=str(payload.get("message") or ""), session_id=session_id, domain=payload.get("domain")).to_dict()
    if command == "system_prompt_block":
        return runtime.system_prompt_block()
    if command == "prefetch":
        return runtime.prefetch(
            str(payload.get("query") or ""),
            session_id=session_id,
            domain=payload.get("domain"),
            limit=int(payload.get("limit") or 5),
            max_chars=int(payload.get("max_chars") or 800),
        )
    if command == "prefetch_turn":
        return runtime.prefetch_turn(
            str(payload.get("message") or ""),
            session_id=session_id,
            turn_number=int(payload.get("turn_number") or 0),
            domain=payload.get("domain"),
            limit=int(payload.get("limit") or 5),
            max_chars=int(payload.get("max_chars") or 800),
        )
    if command == "sync_turn":
        messages = _read_messages(payload)
        user_content = str(payload.get("user_content") or _last_message_text(messages, role="user") or _message_text(payload.get("prompt")) or "")
        assistant_content = str(payload.get("assistant_content") or _last_message_text(messages, role="assistant") or "")
        return runtime.sync_turn(
            user_content,
            assistant_content,
            session_id=session_id,
            messages=messages,
            domain=payload.get("domain"),
            task_title=payload.get("task_title"),
            result_summary=payload.get("result_summary"),
            lessons=payload.get("lessons"),
            skill_steps=payload.get("skill_steps"),
        )
    if command == "on_pre_compress":
        return runtime.on_pre_compress(_read_messages(payload))
    if command == "on_session_end":
        return runtime.on_session_end(
            _read_messages(payload),
            task_title=payload.get("task_title"),
            result_summary=payload.get("result_summary"),
            lessons=payload.get("lessons"),
            skill_steps=payload.get("skill_steps"),
            domain=payload.get("domain"),
            ingest_skill_candidate=bool(payload.get("ingest_skill_candidate", True)),
        )
    if command == "handle_tool_call":
        tool_name = str(payload.get("tool_name") or payload.get("name") or "")
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        return json.loads(runtime.handle_tool_call(tool_name, args))
    if command == "state":
        return json.loads(runtime.handle_tool_call("memory_state", {}))
    if command == "bundle":
        return runtime.build_bundle(
            message=str(payload.get("message") or ""),
            session_id=session_id,
            turn_number=int(payload.get("turn_number") or 0),
            domain=payload.get("domain"),
        ).to_dict()
    if command == "corpus_search":
        query = str(payload.get("query") or "")
        result = json.loads(
            runtime.handle_tool_call(
                "memory_search",
                {
                    "query": query,
                    "domain": payload.get("domain"),
                    "limit": int(payload.get("limit") or 5),
                    "max_chars": int(payload.get("max_chars") or 800),
                },
            )
        )
        matches = []
        for item in result.get("matches", []):
            record = item.get("record", {}) if isinstance(item, dict) else {}
            matches.append(
                {
                    "corpus": "memento",
                    "path": f"memory/{record.get('id', '')}",
                    "title": record.get("content", "")[:80],
                    "kind": record.get("kind"),
                    "score": float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                    "snippet": record.get("content", ""),
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "updatedAt": record.get("updated_at"),
                }
            )
        return matches
    if command == "corpus_get":
        lookup = str(payload.get("lookup") or "")
        store = runtime.governor.workflow.store
        record = store.find_record(lookup)
        if record is None:
            matches = store.find_matching_records(lookup, domain=payload.get("domain"), limit=1, status=None)
            record = matches[0] if matches else None
        if record is None:
            return None
        from_line = max(1, int(payload.get("fromLine") or 1))
        line_count = max(1, int(payload.get("lineCount") or 40))
        lines = record.content.splitlines() or [record.content]
        start = min(from_line - 1, max(len(lines) - 1, 0))
        end = min(len(lines), start + line_count)
        content = "\n".join(lines[start:end])
        return {
            "corpus": "memento",
            "path": f"memory/{record.id}",
            "title": record.content[:80],
            "kind": record.kind,
            "content": content,
            "fromLine": start + 1,
            "lineCount": end - start,
            "id": record.id,
            "sourceType": "memento",
            "updatedAt": record.updated_at,
        }
    raise ValueError(f"Unknown bridge command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenClaw memory bridge CLI")
    parser.add_argument("command", choices=sorted(_COMMANDS))
    args = parser.parse_args(argv)
    try:
        payload = _read_payload(sys.stdin.read())
        result = _run_command(args.command, payload)
        sys.stdout.write(json.dumps(_result(result), ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - bridge failures are surfaced to the host
        sys.stdout.write(json.dumps(_error(str(exc)), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

