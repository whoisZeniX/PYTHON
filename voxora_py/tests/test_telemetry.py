import importlib
import os
import sys
from pathlib import Path

import pytest

from voxora.telemetry import STAGES, LatencyStore, percentile


def test_percentile_empty_and_single():
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 70) == 7.0


def test_percentile_linear_interpolation():
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 70) == pytest.approx(31.0)
    assert percentile(values, 100) == 40.0
    assert percentile(values, 0) == 10.0


def record(i: float) -> dict:
    return {stage: i * (n + 1) + 0.123 for n, stage in enumerate(STAGES)}


def test_summary_shape_and_window():
    store = LatencyStore()
    empty = store.summary(10)
    assert empty["sample_count"] == 0 and empty["source"] == "live_requests"
    assert empty["total_e2e_ms"] == {"p50": 0, "p70": 0, "p100": 0}

    for i in range(1, 16):
        store.add(record(i))
    summary = store.summary(10)
    assert summary["sample_count"] == 10
    assert summary["stt_ms"]["p100"] == round(15 + 0.123, 2)
    assert summary["stt_ms"]["p50"] == round(percentile([i + 0.123 for i in range(6, 16)], 50), 2)


def _reference_root() -> Path | None:
    candidates = [os.environ.get("VOXORA_REF"), Path(__file__).resolve().parents[2] / "voxora"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "app" / "api" / "routes.py").exists():
            return Path(candidate)
    return None


@pytest.mark.skipif(_reference_root() is None, reason="old voxora/ backend not found (set VOXORA_REF)")
def test_summary_matches_old_backend():
    sys.path.insert(0, str(_reference_root()))
    try:
        routes = importlib.import_module("app.api.routes")
    finally:
        sys.path.pop(0)

    import asyncio

    for count in (1, 2, 3, 7, 10):
        records = [record(i * 1.37) for i in range(1, count + 1)]
        routes.latency_records[:] = records
        old = asyncio.run(routes.latency_summary(sample_size=10))

        store = LatencyStore()
        for item in records:
            store.add(item)
        new = store.summary(10)

        old.pop("updated_at")
        new.pop("updated_at")
        assert new == old, f"mismatch with {count} records"
