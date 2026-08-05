from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceEvent:
    name: str
    elapsed_ms: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceProfiler:
    def __init__(self):
        self.events: List[PerformanceEvent] = []

    @contextmanager
    def track(self, name: str, **metadata: Any):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.events.append(
                PerformanceEvent(
                    name=name,
                    elapsed_ms=round(elapsed_ms, 2),
                    metadata=dict(metadata),
                )
            )

    def record(self, name: str, elapsed_ms: float, **metadata: Any) -> None:
        self.events.append(
            PerformanceEvent(
                name=name,
                elapsed_ms=round(float(elapsed_ms), 2),
                metadata=dict(metadata),
            )
        )

    def summary(self) -> Dict[str, Any]:
        total_ms = sum(event.elapsed_ms for event in self.events)
        return {
            "total_tracked_ms": round(total_ms, 2),
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events],
            "slowest_events": [
                event.to_dict()
                for event in sorted(
                    self.events,
                    key=lambda item: item.elapsed_ms,
                    reverse=True,
                )[:20]
            ],
        }


def create_profiler(enabled: bool = False) -> Optional[PerformanceProfiler]:
    return PerformanceProfiler() if enabled else None


@contextmanager
def track_optional(profiler: Optional[PerformanceProfiler], name: str, **metadata: Any):
    if profiler is None:
        yield
        return

    with profiler.track(name, **metadata):
        yield
