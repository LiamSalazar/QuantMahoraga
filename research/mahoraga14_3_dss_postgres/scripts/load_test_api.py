from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from etl.paths import ensure_output_dirs

ENDPOINTS = [
    "/health",
    "/metadata/options",
    "/research/command-center",
    "/whatif/grid",
    "/decision/replay",
    "/ticker/contribution",
    "/regime/behavior",
    "/data/execution-evidence",
]


async def _one(client: httpx.AsyncClient, endpoint: str) -> dict[str, Any]:
    started = time.perf_counter()
    ok = False
    status = 0
    error = ""
    try:
        response = await client.get(endpoint, timeout=30)
        status = response.status_code
        ok = response.status_code < 500
    except Exception as exc:
        error = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"endpoint": endpoint, "status": status, "ok": ok, "elapsed_ms": elapsed_ms, "error": error}


async def run_load_test(base_url: str, concurrency: int, requests: int) -> list[dict[str, Any]]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(idx: int) -> dict[str, Any]:
            async with sem:
                return await _one(client, ENDPOINTS[idx % len(ENDPOINTS)])

        return await asyncio.gather(*(guarded(i) for i in range(requests)))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return values[idx]


def write_results(rows: list[dict[str, Any]], started: float, finished: float) -> None:
    paths = ensure_output_dirs()
    csv_path = paths.benchmarks_root / "api_load_test_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "status", "ok", "elapsed_ms", "error"])
        writer.writeheader()
        writer.writerows(rows)
    latencies = [float(row["elapsed_ms"]) for row in rows]
    errors = [row for row in rows if not row["ok"]]
    by_endpoint = {}
    for row in rows:
        by_endpoint.setdefault(row["endpoint"], []).append(float(row["elapsed_ms"]))
    slowest_endpoint = max(by_endpoint.items(), key=lambda item: statistics.mean(item[1]))[0] if by_endpoint else None
    report = paths.benchmarks_root / "api_load_test_summary.md"
    lines = [
        "# API Load Test Summary",
        "",
        f"- requests: `{len(rows)}`",
        f"- requests_per_second: `{len(rows) / max(0.001, finished - started):.2f}`",
        f"- error_rate: `{len(errors) / max(1, len(rows)):.2%}`",
        f"- p50_ms: `{_percentile(latencies, 0.50):.2f}`",
        f"- p95_ms: `{_percentile(latencies, 0.95):.2f}`",
        f"- p99_ms: `{_percentile(latencies, 0.99):.2f}`",
        f"- slowest_endpoint: `{slowest_endpoint}`",
        "",
        "This is a local reproducible load test, not a user-capacity claim.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local load test for DSS API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = asyncio.run(run_load_test(args.base_url, args.concurrency, args.requests))
    finished = time.perf_counter()
    write_results(rows, started, finished)
    print(f"wrote {ensure_output_dirs().benchmarks_root / 'api_load_test_summary.md'}")


if __name__ == "__main__":
    main()
