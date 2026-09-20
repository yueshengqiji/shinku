"""Read-only gray checks for the independent Shinku memory database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinku.memory import MemoryRouter, MemoryService, MemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查独立版记忆路由与作用域")
    parser.add_argument("database", type=Path)
    parser.add_argument("--conversation-id", default="872732158")
    parser.add_argument("--conversation-type", default="group")
    args = parser.parse_args()

    store = MemoryStore(args.database)
    service = MemoryService(store=store, router=MemoryRouter())
    ordinary = service.prepare_turn(
        conversation_id=args.conversation_id,
        conversation_type=args.conversation_type,
        user_message="今天继续做项目。",
    )
    recalled = service.prepare_turn(
        conversation_id=args.conversation_id,
        conversation_type=args.conversation_type,
        user_message="你还记得 9 月 15 日发生了什么吗？",
    )
    result = {
        "database": str(args.database),
        "stats": store.stats(),
        "ordinary_chat": {
            "need_retrieval": ordinary.decision.need_retrieval,
            "record_count": len(ordinary.records),
        },
        "explicit_recall": {
            "need_retrieval": recalled.decision.need_retrieval,
            "record_count": len(recalled.records),
            "layers": [record.layer for record in recalled.records],
            "sources": [record.source_id for record in recalled.records],
        },
        "checks": {
            "ordinary_chat_does_not_recall": not ordinary.decision.need_retrieval and not ordinary.records,
            "explicit_recall_routes": recalled.decision.need_retrieval,
            "retrieval_has_provenance": all(bool(record.source_id) for record in recalled.records),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
