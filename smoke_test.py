from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from openclaw_memory_plugins import MemoryReflector, MemoryRetriever, MemoryScorer, MemoryStore, MemoryWorkflow


def run_smoke_test() -> dict[str, object]:
    """Run a small end-to-end check against an isolated temp store."""

    with TemporaryDirectory(prefix="memento-smoke-") as tmp_dir:
        with MemoryStore(tmp_dir) as store:
            workflow = MemoryWorkflow(
                store=store,
                scorer=MemoryScorer(),
                retriever=MemoryRetriever(store),
                reflector=MemoryReflector(),
            )

            seeded = workflow.ingest(
                "The user prefers concise answers.",
                kind="preference",
                confidence=0.95,
            )
            seeded_conflict = workflow.ingest(
                "The user does not prefer concise answers.",
                kind="preference",
                confidence=0.9,
            )
            entity_record = store.save_record("OpenClawMemoryGovernor integrates with OpenClawMemoryProvider.")

            with MemoryStore(tmp_dir) as persisted_store:
                persisted_matches = MemoryRetriever(persisted_store).search("OpenClawMemoryGovernor")

            semantic_record = store.save_record("Semantic retrieval should find openclaw memory even when wording shifts.")
            recall = workflow.retrieve("semantic memory retrieval")
            packet = workflow.run_cycle(
                query="OpenClawMemoryGovernor",
                candidate="Remember OpenClawMemoryGovernor should prefer concise answers.",
                task_title="smoke test memory workflow",
                result_summary="The plugin pack runs end to end.",
                lessons="Keep orchestration separate from storage.",
                skill_steps=["retrieve first", "ingest notable candidates", "reflect after the task"],
            )

            ingest = packet["ingest"]
            reflection = packet["reflection"]

            assert seeded.action in {"auto_write", "stage", "drop"}
            assert seeded.stored is not None, "expected the seeded preference to be written"
            assert seeded_conflict.stored is not None or seeded_conflict.staged is not None, "expected conflict candidate to be handled"
            assert persisted_matches, "expected at least one persisted recalled memory"
            assert entity_record.id, "expected the entity record to be stored"
            assert store.get_record_entities(entity_record.id), "expected entity links to be created on save"
            semantic_hits = store.find_semantic_records("semantic memory retrieval", limit=3)
            assert semantic_record.id, "expected the semantic record to be stored"
            assert store.get_record_embedding(semantic_record.id), "expected an embedding row for the record"
            assert semantic_hits, "expected semantic retrieval to return results"
            assert recall.memory_pack, "expected the recall pack to include content"
            assert ingest["action"] in {"auto_write", "stage", "drop"}
            assert ingest["stored"] is not None or ingest["staged"] is not None, "expected the candidate to be handled"
            assert reflection["reflection_saved"] is not None, "expected a durable reflection record"
            assert store.records_path.exists()
            assert store.pending_path.exists()
            assert store.reflections_path.exists()

            return {
                "root_dir": tmp_dir,
                "packet": packet,
                "seeded": seeded.to_dict(),
                "seeded_conflict": seeded_conflict.to_dict(),
                "entity_record": entity_record.to_dict(),
                "semantic_record": semantic_record.to_dict(),
                "counts": {
                    "records": len(store.list_records()),
                    "pending": len(list(store._iter_jsonl(store.pending_path))),
                    "reflections": len(list(store._iter_jsonl(store.reflections_path))),
                },
            }


def main() -> int:
    result = run_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
