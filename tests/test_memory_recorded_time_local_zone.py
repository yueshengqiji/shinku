"""Check saved-date boundaries using the actual process-local timezone."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires process-local POSIX timezone selection")
@pytest.mark.parametrize("zone,start,end,hours", [
    ("Asia/Shanghai", "2026-03-08", "2026-03-09", 24),
    ("UTC", "2026-03-08", "2026-03-09", 24),
    ("America/New_York", "2026-03-08", "2026-03-09", 23),
    ("America/New_York", "2026-11-01", "2026-11-02", 25),
])
def test_real_local_midnights_and_open_bounds_in_isolated_process(zone, start, end, hours):
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
from shinku.memory.query_time import recorded_time_window
time.tzset()
zone, start, end = sys.argv[1:]
actual = recorded_time_window({'start': start, 'end': end})
expected = tuple(int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(zone)).timestamp())
                 for day in (start, end))
assert actual == expected, (zone, actual, expected)
assert recorded_time_window({'start': start, 'end': ''}) == (expected[0], None)
assert recorded_time_window({'start': '', 'end': end}) == (None, expected[1])
print(json.dumps(actual))
"""
    env = dict(os.environ, TZ=zone, PYTHONPATH=str(root / "src"), SHINKU_ENV_FILE=os.devnull)
    result = subprocess.run([sys.executable, "-B", "-c", code, zone, start, end],
        cwd=root, env=env, capture_output=True, text=True, timeout=15, check=True)
    actual_start, actual_end = json.loads(result.stdout)
    assert actual_end - actual_start == hours * 3600
