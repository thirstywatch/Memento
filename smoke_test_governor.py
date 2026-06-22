from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from openclaw_memory_plugins import MemoryStore, OpenClawMemoryGovernor


def run_smoke_test() -> dict[str, object]:
    """Run an isolated end-to-end check against the governor wrapper."""

    with TemporaryDirectory(prefix="openclaw-governor-smoke-") as tmp_dir:
        with MemoryStore(tmp_dir) as store:
            governor = OpenClawMemoryGovernor(store=store, session_id="smoke-session")

            seeded = governor.ingest(
                "The user prefers concise answers.",
                kind="preference",
                confidence=0.95,
            )
            governor.ingest(
                "The user does not prefer concise answers.",
                kind="preference",
                confidence=0.9,
            )
            contradiction = json.loads(governor.handle_tool_call("memory_contradict", {"query": "concise answers", "limit": 5}))
            memory_context = governor.prefetch("concise answers")
            turn_packet = governor.sync_turn(
                "Remember to keep responses short and direct.",
                "Got it, I will keep it short.",
                candidate="Remember to keep responses short and direct.",
                domain="user",
            )
            session_end = governor.on_session_end(
                messages=[
                    {"role": "user", "content": "I prefer concise answers."},
                    {"role": "assistant", "content": "Noted."},
                ],
                task_title="smoke test governor",
                result_summary="The governor wrapper runs end to end.",
                lessons="Keep lifecycle glue outside storage.",
                skill_steps=["prefetch first", "ingest notable candidates", "reflect after the task"],
            )

            assert seeded.stored is not None, "expected the seeded preference to be written"
            assert contradiction["results"], "expected contradiction results"
            assert memory_context.startswith("<memory-context>")
            assert turn_packet.recall.memory_pack, "expected the turn packet to include recalled context"
            assert turn_packet.ingest is not None, "expected the turn packet to ingest a candidate"
            assert session_end["reflection"] is not None, "expected a durable reflection record"
            assert session_end["contradictions"], "expected session end contradiction review"
            assert session_end["extracted"], "expected session end auto-extraction to run"
            assert governor.last_packet is turn_packet, "expected the last packet to track the latest turn"
            assert store.records_path.exists()
            assert store.pending_path.exists()
            assert store.reflections_path.exists()
            assert store.feedback_path.exists()

            return {
                "root_dir": tmp_dir,
                "memory_context": memory_context,
                "turn_packet": turn_packet.to_dict(),
                "session_end": session_end,
                "contradiction": contradiction,
                "counts": {
                    "records": len(store.list_records()),
                    "pending": len(list(store._iter_jsonl(store.pending_path))),
                    "reflections": len(list(store._iter_jsonl(store.reflections_path))),
                    "feedback": len(list(store._iter_jsonl(store.feedback_path))),
                },
            }


def main() -> int:
    result = run_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
