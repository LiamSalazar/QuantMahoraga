from __future__ import annotations

from itertools import product

import polars as pl

from .config import OFFICIAL_CANDIDATE_ID, OFFICIAL_UNIVERSE_ID, RuntimeConfig, official_knobs


def _observed_whatif(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    summary = sources.get("extended_summary", pl.DataFrame())
    if summary.is_empty():
        return pl.DataFrame()
    candidate_col = "candidate_id" if "candidate_id" in summary.columns else "CandidateId"
    out = summary.with_columns(
        scenario_id=pl.concat_str(
            [
                pl.lit("observed|"),
                pl.col(candidate_col).cast(pl.Utf8),
                pl.lit("|"),
                pl.col("budget_multiplier").round(3).cast(pl.Utf8),
                pl.lit("|"),
                pl.col("conviction_multiplier").round(3).cast(pl.Utf8),
                pl.lit("|"),
                pl.col("leader_multiplier").round(3).cast(pl.Utf8),
                pl.lit("|"),
                pl.col("backoff_strength").round(3).cast(pl.Utf8),
            ]
        ),
        candidate_id=pl.col(candidate_col).cast(pl.Utf8),
        cost_bps=pl.lit(5.0),
        slippage_bps=pl.lit(2.0),
        fold=pl.lit(None, dtype=pl.Int32),
        horizon=pl.lit(20),
        alpha=pl.col("AlphaNW_QQQ").cast(pl.Float64, strict=False),
        beta=pl.col("BetaQQQ").cast(pl.Float64, strict=False),
        turnover=pl.col("AvgTurnover").cast(pl.Float64, strict=False),
        avg_exposure=pl.col("AvgExposure").cast(pl.Float64, strict=False),
        robust_score=((pl.col("Sharpe") / 2.0) + (pl.col("CAGR") / 60.0) - (pl.col("MaxDD").abs() / 50.0)).clip(-2, 2),
        source_artifact=pl.lit("extended_multiplier_summary.csv"),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select(
        "scenario_id",
        "candidate_id",
        "budget_multiplier",
        "conviction_multiplier",
        "leader_multiplier",
        "backoff_strength",
        "cost_bps",
        "slippage_bps",
        "fold",
        "universe_id",
        "horizon",
        pl.col("CAGR").alias("cagr"),
        pl.col("Sharpe").alias("sharpe"),
        pl.col("Sortino").alias("sortino"),
        pl.col("MaxDD").alias("maxdd"),
        "alpha",
        "beta",
        "turnover",
        "avg_exposure",
        "robust_score",
        "source_artifact",
        "run_id",
        "demo_mode",
    )
    return out.with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int32))


def _demo_grid(config: RuntimeConfig) -> pl.DataFrame:
    if not config.include_demo_grid:
        return pl.DataFrame()
    target = config.row_target["demo_grid_points"]
    budgets = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    convictions = [0.90, 1.00, 1.10, 1.20, 1.30]
    leaders = [0.90, 1.00, 1.10, 1.20, 1.30]
    backoffs = [0.90, 1.00, 1.05, 1.10, 1.20]
    costs = [0.0, 5.0, 10.0, 20.0]
    slips = [0.0, 2.0, 5.0]
    folds = [1, 2, 3, 4, 5]
    horizons = [1, 5, 20, 60]
    official = official_knobs()
    rows = []
    for idx, (budget, conviction, leader, backoff, cost, slippage, fold, horizon) in enumerate(product(budgets, convictions, leaders, backoffs, costs, slips, folds, horizons)):
        if idx >= target:
            break
        distance = (
            abs(budget - official["budget_multiplier"]) * 1.4
            + abs(conviction - official["conviction_multiplier"]) * 0.8
            + abs(leader - official["leader_multiplier"]) * 0.7
            + abs(backoff - official["backoff_strength"]) * 0.9
        )
        cost_drag = (cost + slippage) / 1000.0
        horizon_boost = {1: -0.03, 5: 0.00, 20: 0.05, 60: 0.07}[horizon]
        fold_drag = (fold - 3) * 0.015
        sharpe = 1.48 - distance - cost_drag + horizon_boost - abs(fold_drag)
        cagr = 32.55 - distance * 18 - (cost + slippage) * 0.22 + horizon_boost * 40 - abs(fold_drag) * 18
        maxdd = -16.20 - distance * 4 - cost_drag * 20 - max(backoff - 1.05, 0) * 2
        rows.append(
            {
                "scenario_id": f"demo|{idx:05d}|B{budget:.2f}|C{conviction:.2f}|L{leader:.2f}|R{backoff:.2f}|c{cost:.0f}|s{slippage:.0f}|f{fold}|h{horizon}",
                "candidate_id": OFFICIAL_CANDIDATE_ID,
                "budget_multiplier": budget,
                "conviction_multiplier": conviction,
                "leader_multiplier": leader,
                "backoff_strength": backoff,
                "cost_bps": cost,
                "slippage_bps": slippage,
                "fold": fold,
                "universe_id": OFFICIAL_UNIVERSE_ID,
                "horizon": horizon,
                "cagr": cagr,
                "sharpe": sharpe,
                "sortino": sharpe * 1.68,
                "maxdd": maxdd,
                "alpha": cagr / 100.0 - 0.09,
                "beta": 0.52 + (budget - 1.05) * 0.2,
                "turnover": 0.05 + abs(conviction - 1.1) * 0.02 + cost / 4000,
                "avg_exposure": 0.65 + (budget - 1.05) * 0.45 + (leader - 1.1) * 0.08,
                "robust_score": sharpe / 2.0 + cagr / 60.0 - abs(maxdd) / 50.0,
                "source_artifact": "demo_synthetic_whatif_grid",
                "run_id": config.run_id,
                "demo_mode": True,
            }
        )
    return pl.DataFrame(rows).sort(["robust_score", "sharpe"], descending=True).with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int32))


def build_fact_whatif(sources: dict[str, pl.DataFrame], config: RuntimeConfig) -> pl.DataFrame:
    frames = [_observed_whatif(sources, config.run_id), _demo_grid(config)]
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
