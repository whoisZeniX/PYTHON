import threading
import time

STAGES = ("stt_ms", "retrieval_ms", "generation_ms", "grounding_ms", "total_rag_ms", "total_e2e_ms")


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])

    position = (percent / 100) * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


class LatencyStore:
    def __init__(self):
        self.records: list[dict[str, float]] = []
        self._lock = threading.Lock()

    def add(self, timings: dict[str, float]) -> None:
        with self._lock:
            self.records.append({stage: timings[stage] for stage in STAGES})

    def summary(self, sample_size: int = 10) -> dict:
        with self._lock:
            records = self.records[-sample_size:] if sample_size > 0 else []

        result: dict = {
            "sample_count": len(records),
            "updated_at": time.time(),
            "source": "live_requests",
        }
        for stage in STAGES:
            if not records:
                result[stage] = {"p50": 0, "p70": 0, "p100": 0}
                continue
            values = [record[stage] for record in records]
            result[stage] = {
                "p50": round(percentile(values, 50), 2),
                "p70": round(percentile(values, 70), 2),
                "p100": round(percentile(values, 100), 2),
            }
        return result
