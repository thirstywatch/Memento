from __future__ import annotations

import argparse
import json
from pathlib import Path

from openclaw_memory_plugins import OpenClawRuntimeAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenClaw memory runtime adapter once and print JSON output.")
    parser.add_argument("--query", default="what do we know about this project?", help="Recall query to run first.")
    parser.add_argument("--assistant", default="Got it.", help="Assistant reply used for the sync turn.")
    parser.add_argument("--session-id", default="runtime-session", help="Optional session identifier.")
    parser.add_argument("--root-dir", type=Path, help="Optional storage root; defaults to ~/.openclaw-memory.")
    parser.add_argument("--workspace-dir", type=Path, help="Optional OpenClaw workspace directory.")
    parser.add_argument("--openclaw-home", type=Path, help="Optional OpenClaw home directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime = OpenClawRuntimeAdapter(
        storage_root=args.root_dir,
        session_id=args.session_id,
        platform="openclaw",
        openclaw_home=args.openclaw_home,
        workspace_dir=args.workspace_dir,
    )
    with runtime:
        runtime.initialize(
            args.session_id,
            openclaw_home=args.openclaw_home,
            workspace_dir=args.workspace_dir,
            platform="openclaw",
        )
        bundle = runtime.build_bundle(message=args.query, session_id=args.session_id)
        turn = runtime.sync_turn(
            args.query,
            args.assistant,
            session_id=args.session_id,
            messages=[
                {"role": "user", "content": args.query},
                {"role": "assistant", "content": args.assistant},
            ],
        )
        session_end = runtime.on_session_end(
            [
                {"role": "user", "content": args.query},
                {"role": "assistant", "content": args.assistant},
            ],
            task_title="runtime runner",
            result_summary="The OpenClaw memory runtime adapter executed a full cycle.",
        )
        print(
            json.dumps(
                {
                    "bundle": bundle.to_dict(),
                    "turn": turn,
                    "session_end": session_end,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
