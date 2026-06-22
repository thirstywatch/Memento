from __future__ import annotations

import argparse
import json

from openclaw_memory_plugins import OpenClawHermesMemoryProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Hermes-compatible OpenClaw memory provider once and print JSON output.")
    parser.add_argument("--query", default="what do we know about this project?", help="Recall query to run first.")
    parser.add_argument("--session-id", default="", help="Optional session identifier for the provider.")
    parser.add_argument("--task-title", help="Optional task title used for a final reflection.")
    parser.add_argument("--result-summary", help="Optional result summary used for a final reflection.")
    parser.add_argument("--lesson", dest="lessons", help="Optional lesson text for reflection.")
    parser.add_argument("--platform", default="cli", help="Platform label passed through initialize().")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provider = OpenClawHermesMemoryProvider()
    provider.initialize(args.session_id or "demo-session", hermes_home="~/.hermes", platform=args.platform)

    packet: dict[str, object] = {
        "system_prompt_block": provider.system_prompt_block(),
        "memory_context": provider.prefetch(args.query, session_id=args.session_id),
    }
    packet["tool_memory_add"] = json.loads(
        provider.handle_tool_call(
            "memory_add",
            {
                "target": "user",
                "content": "The user prefers concise answers.",
                "kind": "preference",
                "confidence": 0.95,
            },
        )
    )
    if args.task_title and args.result_summary:
        packet["session_end"] = provider._ensure_governor().on_session_end(
            task_title=args.task_title,
            result_summary=args.result_summary,
            lessons=args.lessons,
        )
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
