"""Import the old Shinku memory database into the independent memory store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinku.memory import migrate_legacy_sqlite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移旧真红记忆库；源库只读，不复制向量索引")
    parser.add_argument("source", type=Path, help="旧 shinku_memory_v01.db")
    parser.add_argument("target", type=Path, help="独立版 memory.sqlite3")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入目标库")
    args = parser.parse_args()
    report = migrate_legacy_sqlite(args.source, args.target, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
