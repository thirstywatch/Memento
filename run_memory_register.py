from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from openclaw_memory_plugins import MemoryStore, OpenClawMemoryGovernor


def run_smoke_test() -> dict[str, object]:
    """Exercise the OpenClaw memory plugin's registration surface."""

    with TemporaryDirectory(prefix="memento-register-smoke-") as tmp_dir:
        with MemoryStore(tmp_dir) as store:
            governor = OpenClawMemoryGovernor(store=store, session_id="register-session")
            tools = governor.get_tool_schemas()
            provider = {
                "provider_name": "memento",
                "tool_names": [tool["name"] for tool in tools],
                "contradict": json.loads(governor.handle_tool_call("memory_contradict", {"query": "openclaw memory", "limit": 3})),
                "storage_root": str(store.root_dir),
            }
            assert "memory_contradict" in provider["tool_names"]
            assert isinstance(provider["contradict"], dict)
            return provider


def main() -> int:
    result = run_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
