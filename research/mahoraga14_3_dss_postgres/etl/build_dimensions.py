from __future__ import annotations

import re
from datetime import date

import polars as pl

from .config import METRICS, MODULE_ORDER, OFFICIAL_CANDIDATE_ID, official_knobs
from .date_utils import parse_date


def _parse_candidate(candidate_id: str) -> dict[str, float | None]:
    match = re.match(r"B(?P<budget>[0-9.]+)_C(?P<conviction>[0-9.]+)_L(?P<leader>[0-9.]+)_R(?P<backoff>[0-9.]+)", candidate_id or "")
    if not match:
        return {"budget_multiplier": None, "conviction_multiplier": None, "leader_multiplier": None, "backoff_strength": None}
    return {
        "budget_multiplier": float(match.group("budget")),
        "conviction_multiplier": float(match.group("conviction")),
        "leader_multiplier": float(match.group("leader")),
        "backoff_strength": float(match.group("backoff")),
    }


def build_dim_date(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    frames = []
    for name, column in [("decision", "date"), ("position", "date"), ("module_trace", "date"), ("outcome", "decision_date"), ("market_context", "date"), ("active_return", "Date")]:
        df = sources.get(name, pl.DataFrame())
        if not df.is_empty() and column in df.columns:
            frames.append(df.select(parse_date(column).alias("date_value")).drop_nulls())
    if not frames:
        return pl.DataFrame()
    dates = pl.concat(frames).unique().sort("date_value")
    return dates.with_columns(
        date_key=pl.col("date_value").dt.strftime("%Y%m%d").cast(pl.Int32),
        year=pl.col("date_value").dt.year(),
        quarter=pl.col("date_value").dt.quarter(),
        month=pl.col("date_value").dt.month(),
        month_name=pl.col("date_value").dt.strftime("%B"),
        week=pl.col("date_value").dt.week(),
        day_of_week=pl.col("date_value").dt.weekday(),
        is_month_end=(pl.col("date_value").dt.month() != pl.col("date_value").shift(-1).dt.month()).fill_null(True),
    ).select(["date_key", "date_value", "year", "quarter", "month", "month_name", "week", "day_of_week", "is_month_end"])


def build_dim_asset(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    tickers: set[str] = set()
    position = sources.get("position", pl.DataFrame())
    if not position.is_empty() and "ticker" in position.columns:
        tickers.update(position.get_column("ticker").drop_nulls().cast(pl.Utf8).to_list())
    universe = sources.get("universe_summary", pl.DataFrame())
    if not universe.is_empty() and "usable_tickers" in universe.columns:
        for value in universe.get_column("usable_tickers").drop_nulls().cast(pl.Utf8).to_list():
            tickers.update(item.strip() for item in value.split(",") if item.strip())
    rows = [{"ticker": ticker, "asset_name": None, "asset_class": "equity", "sector": None, "source_universe": None, "demo_mode": False} for ticker in sorted(tickers)]
    return pl.DataFrame(rows)


def build_dim_candidate(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    candidates: dict[str, dict[str, object]] = {}
    for source_name in ["extended_summary", "universe_summary", "decision", "position", "outcome"]:
        df = sources.get(source_name, pl.DataFrame())
        for column in ["candidate_id", "CandidateId"]:
            if not df.is_empty() and column in df.columns:
                for candidate in df.get_column(column).drop_nulls().cast(pl.Utf8).unique().to_list():
                    knobs = _parse_candidate(candidate)
                    if candidate == OFFICIAL_CANDIDATE_ID:
                        knobs = official_knobs()
                    candidates[candidate] = {
                        "candidate_id": candidate,
                        "candidate_label": "Official baseline" if candidate == OFFICIAL_CANDIDATE_ID else candidate,
                        "family": "official" if candidate == OFFICIAL_CANDIDATE_ID else ("controlled_extreme" if candidate.startswith("EXTREME_") else "extended_multiplier"),
                        "is_official": candidate == OFFICIAL_CANDIDATE_ID,
                        "demo_mode": False,
                        **knobs,
                    }
    return pl.DataFrame(list(candidates.values())).sort("candidate_id") if candidates else pl.DataFrame()


def build_dim_universe(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    universe = sources.get("universe_summary", pl.DataFrame())
    if universe.is_empty():
        return pl.DataFrame()
    keep = ["universe_id", "proposed_count", "usable_count", "usable_tickers", "missing_tickers", "effective_start", "effective_end", "mean_coverage_ratio"]
    existing = [column for column in keep if column in universe.columns]
    out = universe.select(existing).unique(subset=["universe_id"], keep="first")
    for column in ["effective_start", "effective_end"]:
        if column in out.columns:
            out = out.with_columns(parse_date(column))
    return out.with_columns(demo_mode=pl.lit(False))


def build_dim_fold(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    fold = sources.get("extended_fold_summary", pl.DataFrame())
    if fold.is_empty():
        fold = sources.get("baseline_fold_summary", pl.DataFrame())
    if fold.is_empty() or "Fold" not in fold.columns:
        return pl.DataFrame()
    out = fold.select(
        pl.col("Fold").cast(pl.Int32).alias("fold"),
        parse_date("TestStart").alias("test_start"),
        parse_date("TestEnd").alias("test_end"),
    ).drop_nulls("fold").unique(subset=["fold"], keep="first").sort("fold")
    return out.with_columns(label=pl.concat_str([pl.lit("Fold "), pl.col("fold").cast(pl.Utf8)]))


def build_dim_module(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    modules = set(MODULE_ORDER)
    trace = sources.get("module_trace", pl.DataFrame())
    if not trace.is_empty() and "module_name" in trace.columns:
        modules.update(trace.get_column("module_name").drop_nulls().cast(pl.Utf8).to_list())
    rows = []
    for module in sorted(modules, key=lambda item: MODULE_ORDER.get(item, ("other", 999))[1]):
        family, order = MODULE_ORDER.get(module, ("other", 999))
        rows.append({"module_name": module, "module_family": family, "display_order": order})
    return pl.DataFrame(rows)


def build_dim_regime(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    regimes = set()
    for source_name, column in [("decision", "participation_state"), ("market_context", "market_regime_proxy")]:
        df = sources.get(source_name, pl.DataFrame())
        if not df.is_empty() and column in df.columns:
            regimes.update(df.get_column(column).drop_nulls().cast(pl.Utf8).to_list())
    rows = [{"regime_name": item, "regime_family": "participation_state", "description": None} for item in sorted(regimes)]
    return pl.DataFrame(rows)


def build_dim_horizon(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    outcome = sources.get("outcome", pl.DataFrame())
    horizons = [1, 5, 20, 60]
    if not outcome.is_empty() and "horizon" in outcome.columns:
        horizons = sorted(set(horizons) | set(outcome.get_column("horizon").drop_nulls().cast(pl.Int32).to_list()))
    return pl.DataFrame({"horizon": horizons}).with_columns(horizon_label=pl.concat_str([pl.col("horizon").cast(pl.Utf8), pl.lit("d")]))


def build_dim_scenario(whatif: pl.DataFrame) -> pl.DataFrame:
    if whatif.is_empty() or "scenario_id" not in whatif.columns:
        return pl.DataFrame()
    keep = ["scenario_id", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cost_bps", "slippage_bps", "demo_mode"]
    return whatif.select([column for column in keep if column in whatif.columns]).unique(subset=["scenario_id"], keep="first").with_columns(
        scenario_family=pl.when(pl.col("demo_mode")).then(pl.lit("demo_grid")).otherwise(pl.lit("observed_summary"))
    )


def build_dim_metric() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "metric_name": name,
                "metric_family": family,
                "display_name": display,
                "higher_is_better": higher,
                "unit": unit,
            }
            for name, family, display, higher, unit in METRICS
        ]
    )


def build_dimensions(sources: dict[str, pl.DataFrame], whatif: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    return {
        "dim_date": build_dim_date(sources),
        "dim_asset": build_dim_asset(sources),
        "dim_candidate": build_dim_candidate(sources),
        "dim_universe": build_dim_universe(sources),
        "dim_fold": build_dim_fold(sources),
        "dim_module": build_dim_module(sources),
        "dim_regime": build_dim_regime(sources),
        "dim_horizon": build_dim_horizon(sources),
        "dim_scenario": build_dim_scenario(whatif if whatif is not None else pl.DataFrame()),
        "dim_metric": build_dim_metric(),
    }
