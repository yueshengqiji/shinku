"""One outstanding operation, coalesced waiters, no unbounded work queue."""
from __future__ import annotations

from collections import OrderedDict
from contextvars import copy_context
import threading
import time


class SingleFlight:
    def __init__(self, *, entries=128, ttl=3600):
        self.lock = threading.Lock()
        self.pending = None
        self.cache = OrderedDict()
        self.entries, self.ttl = entries, ttl
        self.generation = 0
        self.closed = False

    def clear(self):
        with self.lock:
            self.generation += 1
            self.cache.clear()

    def stop(self):
        with self.lock:
            self.closed = True
            self.generation += 1
            self.cache.clear()

    def run(self, key, operation, timeout, *, admit=lambda: True):
        started = time.monotonic()
        with self.lock:
            if self.closed:
                return None, {"status": "stopped", "calls": 0}
            cached = self.cache.get(key)
            if cached and cached[0] > started:
                self.cache.move_to_end(key)
                return cached[1], {"status": "cache_hit", "calls": 0}
            joined = self.pending is not None
            if joined:
                job = self.pending
                if job["key"] != key:
                    return None, {"status": "busy", "calls": 0}
            else:
                if not admit():
                    return None, {"status": "call_budget", "calls": 0}
                job = {"key": key, "generation": self.generation, "event": threading.Event(),
                       "result": None, "error": "", "abandoned": False}
                self.pending = job

                def work():
                    try:
                        result = operation()
                        with self.lock:
                            if not job["abandoned"] and job["generation"] == self.generation:
                                job["result"] = result
                                # Empty/failure responses are never negative cached.
                                if result:
                                    self.cache[key] = (time.monotonic() + self.ttl, result)
                                    while len(self.cache) > self.entries:
                                        self.cache.popitem(last=False)
                    except Exception as exc:
                        job["error"] = type(exc).__name__
                    finally:
                        with self.lock:
                            if self.pending is job:
                                self.pending = None
                        job["event"].set()

                context = copy_context()
                threading.Thread(target=lambda:context.run(work), name="projection-bounded-call", daemon=True).start()
        ready = job["event"].wait(max(0, timeout))
        with self.lock:
            if not ready:
                job["abandoned"] = True
                self.cache.pop(key, None)
            stale = job["generation"] != self.generation or job["abandoned"]
            result = job["result"] if ready and not stale else None
        return result, {"status": ("timeout" if not ready else "invalidated" if stale else
                                   "failed" if job["error"] else "coalesced" if joined else "completed"),
                        "error_type": job["error"], "calls": 0 if joined else 1,
                        "elapsed_ms": round((time.monotonic()-started)*1000, 3)}
