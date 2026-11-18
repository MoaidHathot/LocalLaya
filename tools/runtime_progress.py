"""Human-readable startup stages on stderr, with the same timings included in ready metadata."""
from contextlib import contextmanager
import sys
import time


class StartupProgress:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.phases = []

    def log(self, message):
        if self.enabled:
            print(f"[startup +{time.perf_counter() - self.started:.1f}s] {message}", file=sys.stderr, flush=True)

    @contextmanager
    def stage(self, name, label):
        self.log(f"{label} ...")
        start = time.perf_counter()
        status = "done"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.phases.append({"stage": name, "ms": round(elapsed, 1), "status": status})
            self.log(f"{label}: {status} in {elapsed / 1000:.2f}s")

    def report(self):
        return {"totalMs": round((time.perf_counter() - self.started) * 1000, 1), "phases": list(self.phases)}
