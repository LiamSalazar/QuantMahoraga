from __future__ import annotations

import re

import polars as pl

from .config import OFFICIAL_CANDIDATE_ID, OFFICIAL_UNIVERSE_ID


def _bps_from_scenario(scenario: str, kind: str) -> float:
    if scenario == "BASELINE":
        return 0.0
    match = re.search(r"PLUS_([0-9]+)", scenario)
    if not match:
        return 0.0
    value = float(match.group(1))
    return value if kind == "cost" else value


def _convert(df: pl.DataFrame, kind: str, run_id: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()
    rows = []
    for row in df.iter_rows(named=True):
        scenario = str(row.get("Scenario", "BASELINE"))
        cost_bps = _bps_from_scenario(scenario, "cost") if kind == "cost" else 0.0
        slippage_bps = _bps_from_scenario(scenario, "slippage") if kind == "slippage" else 0.0
        rows.append(
            {
                "candidate_id": OFFICIAL_CANDIDATE_ID,
                "universe_id": OFFICIAL_UNIVERSE_ID,
                "cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
                "cagr": row.get("CAGR"),
                "sharpe": row.get("Sharpe"),
                "maxdd": row.get("MaxDD"),
                "alpha": row.get("AlphaNW_QQQ"),
                "run_id": run_id,
                "demo_mode": False,
            }
        )
    return pl.DataFrame(rows)


def build_fact_cost_sensitivity(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    frames = [
        _convert(sources.get("cost_sensitivity", pl.DataFrame()), "cost", run_id),
        _convert(sources.get("slippage_sensitivity", pl.DataFrame()), "slippage", run_id),
    ]
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

