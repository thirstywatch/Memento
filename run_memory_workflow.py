from __future__ import annotations

import argparse
import json
from pathlib import Path

from openclaw_memory_plugins import MemoryStore, MemoryWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenClaw memory workflow once and print JSON output.")
    parser.add_argument("--query", default="what do we know about this project?", help="Recall query to run first.")
    parser.add_argument(
        "--candidate",
        default="The user prefers concise answers.",
        help="Candidate memory content to ingest.",
    )
    parser.add_argument("--kind", default="note", help="Candidate kind, for example note, preference, or decision.")
    parser.add_argument("--source", default="conversation", help="Candidate source label.")
    parser.add_argument("--domain", choices=["user", "project", "agent"], help="Force the memory domain.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Candidate confidence score.")
    parser.add_argument("--tag", action="append", default=[], help="Optional tag to attach; can be repeated.")
    parser.add_argument("--task-title", help="Task title used for reflection.")
    parser.add_argument("--result-summary", help="Result summary used for reflection.")
    parser.add_argument("--lesson", dest="lessons", help="Optional lesson text for reflection.")
    parser.add_argument(
        "--skill-step",
        dest="skill_steps",
        action="append",
        default=[],
        help="Step to include in a skill candidate; can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Recall limit.")
    parser.add_argument("--max-chars", type=int, default=800, help="Maximum recall text size.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="Optional storage root; defaults to ~/.openclaw-memory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with MemoryStore(args.root_dir) if args.root_dir is not None else MemoryStore() as store:
        workflow = MemoryWorkflow(store=store)

        packet = {"recall": workflow.retrieve(args.query, domain=args.domain, limit=args.limit, max_chars=args.max_chars).to_dict()}
        packet["ingest"] = workflow.ingest(
            args.candidate,
            domain=args.domain,
            kind=args.kind,
            source=args.source,
            confidence=args.confidence,
            tags=args.tag,
        ).to_dict()
        if args.task_title and args.result_summary:
            packet["reflection"] = workflow.reflect(
                task_title=args.task_title,
                result_summary=args.result_summary,
                lessons=args.lessons,
                skill_steps=args.skill_steps or None,
                domain=args.domain or "project",
            ).to_dict()

        print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
