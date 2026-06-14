from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl

from etl.config import OFFICIAL_CANDIDATE_ID, OFFICIAL_UNIVERSE_ID
from etl.paths import get_paths


DIM_TABLES = {"dim_date", "dim_asset", "dim_candidate", "dim_universe", "dim_fold", "dim_module", "dim_regime", "dim_horizon", "dim_scenario", "dim_metric"}


def _clean(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def records(df: pl.DataFrame, limit: int = 500) -> dict[str, Any]:
    if df.is_empty():
        return {"count": 0, "rows": []}
    out = df.head(max(1, min(limit, 5000)))
    return {"count": df.height, "rows": [_clean(row) for row in out.to_dicts()]}


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


class ParquetBackend:
    backend_name = "parquet"

    def __init__(self) -> None:
        self.paths = get_paths()
        self.cache: dict[str, pl.DataFrame] = {}
        self.query_log: list[dict[str, Any]] = []

    def table_path(self, table: str) -> Path:
        family = "dimensions" if table in DIM_TABLES else "facts" if table.startswith("fact_") else "oltp"
        return self.paths.parquet_root / family / f"{table}.parquet"

    def read(self, table: str) -> pl.DataFrame:
        if table not in self.cache:
            path = self.table_path(table)
            self.cache[table] = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        return self.cache[table]

    def timed(self, endpoint: str, source_relation: str, fn) -> Any:
        started = time.perf_counter()
        result = fn()
        rows = result.get("count") if isinstance(result, dict) and isinstance(result.get("count"), int) else None
        if rows is None and isinstance(result, dict) and isinstance(result.get("rows"), list):
            rows = len(result["rows"])
        elapsed = (time.perf_counter() - started) * 1000
        self.query_log.append(
            {
                "query_id": f"q{len(self.query_log) + 1:05d}",
                "endpoint": endpoint,
                "backend": self.backend_name,
                "source_relation": source_relation,
                "rows_returned": rows or 0,
                "elapsed_ms": elapsed,
                "used_materialized_view": source_relation.startswith("mart."),
                "scanned_rows": None,
                "demo_mode": self.demo_mode(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return result

    def demo_mode(self) -> bool:
        whatif = self.read("fact_whatif")
        return not whatif.is_empty() and bool(whatif.get_column("demo_mode").any())

    def row_counts(self) -> dict[str, int]:
        manifest = self.paths.reports_root / "parquet_manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                return {str(table): int(count) for table, count in payload.get("row_counts", {}).items()}
            except Exception:
                pass
        counts: dict[str, int] = {}
        for family in ["dimensions", "facts", "oltp"]:
            root = self.paths.parquet_root / family
            if root.exists():
                for path in root.glob("*.parquet"):
                    counts[path.stem] = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        return counts

    def options(self) -> dict[str, Any]:
        candidates = self.read("dim_candidate")
        folds = self.read("dim_fold")
        universes = self.read("dim_universe")
        assets = self.read("dim_asset")
        modules = self.read("dim_module")
        horizons = self.read("dim_horizon")
        regimes = self.read("dim_regime")
        dates = self.read("dim_date")
        metrics = self.read("dim_metric")
        whatif = self.read("fact_whatif")
        slider_ranges = {}
        for column in ["budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cost_bps", "slippage_bps"]:
            if not whatif.is_empty() and column in whatif.columns:
                values = whatif.get_column(column).drop_nulls()
                slider_ranges[column] = {"min": float(values.min()), "max": float(values.max()), "values": sorted(set(round(float(v), 4) for v in values.to_list()))}
        return {
            "candidates": candidates.get_column("candidate_id").to_list() if not candidates.is_empty() else [],
            "universes": universes.get_column("universe_id").to_list() if not universes.is_empty() else [],
            "folds": folds.get_column("fold").to_list() if not folds.is_empty() else [],
            "tickers": assets.get_column("ticker").to_list() if not assets.is_empty() else [],
            "modules": modules.get_column("module_name").to_list() if not modules.is_empty() else [],
            "horizons": horizons.get_column("horizon").to_list() if not horizons.is_empty() else [],
            "regimes": regimes.get_column("regime_name").to_list() if not regimes.is_empty() else [],
            "metrics": metrics.get_column("metric_name").to_list() if not metrics.is_empty() else [],
            "benchmarks": ["QQQ", "SPY", "CONTROL"],
            "date_range": {
                "start": dates.get_column("date_value").min().isoformat() if not dates.is_empty() else None,
                "end": dates.get_column("date_value").max().isoformat() if not dates.is_empty() else None,
            },
            "slider_ranges": slider_ranges,
            "default_candidate": OFFICIAL_CANDIDATE_ID,
            "default_universe": OFFICIAL_UNIVERSE_ID,
        }

    def _common_filter(
        self,
        df: pl.DataFrame,
        date_col: str | None = None,
        candidate_id: str | None = None,
        fold: int | None = None,
        universe_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        out = df
        if out.is_empty():
            return out
        if candidate_id and "candidate_id" in out.columns:
            out = out.filter(pl.col("candidate_id") == candidate_id)
        if fold is not None and "fold" in out.columns:
            out = out.filter(pl.col("fold") == fold)
        if universe_id and "universe_id" in out.columns:
            out = out.filter(pl.col("universe_id") == universe_id)
        if date_col and date_col in out.columns:
            if start_date:
                out = out.filter(pl.col(date_col) >= _date(start_date))
            if end_date:
                out = out.filter(pl.col(date_col) <= _date(end_date))
        return out

    def scorecard(self, candidate_id: str | None = None, universe_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        df = self.read("fact_candidate_metric")
        df = self._common_filter(df, candidate_id=candidate_id, universe_id=universe_id)
        if not df.is_empty() and "sharpe" in df.columns:
            df = df.sort(["sharpe", "cagr"], descending=True)
        return records(df, limit)

    def overview(self, candidate_id: str, fold: int | None, universe_id: str, benchmark: str, start_date: str | None, end_date: str | None) -> dict[str, Any]:
        path = self._common_filter(self.read("fact_path_recursive"), "date_value", candidate_id, fold, None, start_date, end_date).sort("date_value")
        decision = self._common_filter(self.read("fact_decision_state"), "date_value", candidate_id, fold, universe_id, start_date, end_date).sort("date_value")
        score = self.scorecard(candidate_id, universe_id, 20)["rows"]
        outcome = self._common_filter(self.read("fact_outcome"), "decision_date", candidate_id, fold, universe_id, start_date, end_date)
        fold_perf = self.fold_performance(candidate_id, universe_id)
        helped = float(outcome.get_column("helped_flag").cast(pl.Int8).mean()) if not outcome.is_empty() else None
        return {
            "backend": self.backend_name,
            "demo_mode": self.demo_mode(),
            "scorecard": score,
            "equity_curve": records(path.select("date_value", "equity", "drawdown", "rolling_peak") if not path.is_empty() else path, 5000)["rows"],
            "exposure_turnover": records(decision.select("date_value", "expected_exposure", "expected_turnover", "long_budget", "participation_state") if not decision.is_empty() else decision, 5000)["rows"],
            "decision_summary": {
                "observations": decision.height,
                "avg_exposure": float(decision.get_column("expected_exposure").mean()) if not decision.is_empty() else None,
                "avg_turnover": float(decision.get_column("expected_turnover").mean()) if not decision.is_empty() else None,
                "helped_rate": helped,
                "benchmark": benchmark,
            },
            "fold_performance": fold_perf["rows"],
        }

    def robustness_surface(self, metric: str, fold: int | None, universe_id: str | None, regime: str | None, limit: int = 5000) -> dict[str, Any]:
        df = self.read("fact_robustness_surface")
        if not df.is_empty():
            df = df.filter(pl.col("metric_name") == metric)
            if universe_id:
                df = df.filter(pl.col("universe_id") == universe_id)
            if fold is not None and "fold" in df.columns:
                df = df.filter(pl.col("fold").is_null() | (pl.col("fold") == fold))
            if regime and "regime" in df.columns:
                df = df.filter(pl.col("regime").is_null() | (pl.col("regime") == regime))
            df = df.sort(["budget_multiplier", "conviction_multiplier"])
        return records(df, limit)

    def whatif_grid(
        self,
        candidate_id: str,
        fold: int | None,
        universe_id: str,
        horizon: int,
        cost_bps: float | None,
        slippage_bps: float | None,
        limit: int = 5000,
    ) -> dict[str, Any]:
        df = self.read("fact_whatif")
        df = self._common_filter(df, candidate_id=candidate_id, fold=fold, universe_id=universe_id)
        if not df.is_empty():
            df = df.filter(pl.col("horizon") == horizon)
            if cost_bps is not None:
                df = df.filter(pl.col("cost_bps") == cost_bps)
            if slippage_bps is not None:
                df = df.filter(pl.col("slippage_bps") == slippage_bps)
            df = df.sort(["robust_score", "sharpe"], descending=True)
        pareto = df.filter((pl.col("cagr") > 0) & (pl.col("maxdd") > -30)).sort(["cagr", "maxdd"], descending=[True, True]).head(80) if not df.is_empty() else pl.DataFrame()
        return {
            "count": df.height,
            "rows": records(df, limit)["rows"],
            "pareto": records(pareto, 100)["rows"],
            "demo_rows": int(df.filter(pl.col("demo_mode")).height) if not df.is_empty() else 0,
        }

    def decision_replay(self, candidate_id: str, fold: int | None, universe_id: str, date_value: str | None, ticker: str | None) -> dict[str, Any]:
        decisions = self._common_filter(self.read("fact_decision_state"), "date_value", candidate_id, fold, universe_id).sort("date_value")
        if decisions.is_empty():
            return {"decision": None, "positions": [], "modules": [], "outcomes": [], "market_context": None, "timeline": []}
        if date_value:
            selected = decisions.filter(pl.col("date_value") == _date(date_value))
            if selected.is_empty():
                selected = decisions.head(1)
        else:
            selected = decisions.sort("drawdown").head(1) if "drawdown" in decisions.columns else decisions.head(1)
        decision = selected.row(0, named=True)
        d = decision["date_value"].isoformat()
        dfold = int(decision["fold"])
        positions = self._common_filter(self.read("fact_position_daily"), "date_value", candidate_id, dfold, universe_id, d, d)
        if ticker:
            positions = positions.filter(pl.col("ticker") == ticker)
        positions = positions.sort("final_weight", descending=True) if not positions.is_empty() and "final_weight" in positions.columns else positions
        modules = self._common_filter(self.read("fact_module_trace"), "date_value", candidate_id, dfold, universe_id, d, d)
        outcomes = self._common_filter(self.read("fact_outcome"), "decision_date", candidate_id, dfold, universe_id, d, d).sort("horizon")
        market = self.read("fact_market_bar").filter((pl.col("date_value") == _date(d)) & (pl.col("ticker").is_in(["QQQ", "SPY"])))
        timeline = decisions.select("date_value", "expected_exposure", "drawdown", "participation_state").tail(120)
        return {
            "decision": _clean(decision),
            "positions": records(positions, 50)["rows"],
            "modules": records(modules, 50)["rows"],
            "outcomes": records(outcomes, 20)["rows"],
            "market_context": records(market, 10)["rows"],
            "timeline": records(timeline, 140)["rows"],
        }

    def slice_query(
        self,
        dimensions: list[str],
        measure: str,
        operation: str,
        candidate_id: str | None,
        fold: int | None,
        universe_id: str | None,
        module: str | None,
        ticker: str | None,
        regime: str | None,
        horizon: int | None,
        start_date: str | None,
        end_date: str | None,
        limit: int = 500,
    ) -> dict[str, Any]:
        table = "fact_outcome"
        date_col = "decision_date"
        measure_map = {
            "return": ("realized_return", "mean"),
            "alpha": ("alpha_vs_qqq", "mean"),
            "helped_rate": ("helped_flag", "mean"),
            "drawdown": ("drawdown", "mean"),
            "exposure": ("expected_exposure", "mean"),
            "turnover": ("expected_turnover", "mean"),
        }
        if ticker or "ticker" in dimensions:
            table, date_col = "fact_position_daily", "date_value"
            measure_map.update({"return": ("pnl_contribution", "sum"), "alpha": ("pnl_contribution", "sum"), "exposure": ("final_weight", "mean")})
        elif module or "module_name" in dimensions:
            table, date_col = "fact_module_trace", "date_value"
            measure_map.update({"helped_rate": ("module_active", "mean"), "exposure": ("intensity_score", "mean"), "turnover": ("raw_value", "mean")})
        elif measure in {"drawdown", "exposure", "turnover"} or regime or "regime" in dimensions:
            table, date_col = "fact_decision_state", "date_value"
        df = self._common_filter(self.read(table), date_col, candidate_id, fold, universe_id, start_date, end_date)
        if horizon is not None and "horizon" in df.columns:
            df = df.filter(pl.col("horizon") == horizon)
        if ticker and "ticker" in df.columns:
            df = df.filter(pl.col("ticker") == ticker)
        if module and "module_name" in df.columns:
            df = df.filter(pl.col("module_name") == module)
        if regime and "regime" in df.columns:
            df = df.filter(pl.col("regime") == regime)
        value_col, agg = measure_map[measure]
        dims = [dim for dim in dimensions if dim in df.columns]
        if df.is_empty() or not dims or value_col not in df.columns:
            return {"count": 0, "rows": [], "operation": operation, "table": table}
        expr = pl.col(value_col).cast(pl.Float64, strict=False).sum().alias(measure) if agg == "sum" else pl.col(value_col).cast(pl.Float64, strict=False).mean().alias(measure)
        out = df.group_by(dims).agg(expr, pl.len().alias("observations")).sort(measure, descending=True)
        return {"count": out.height, "rows": records(out, limit)["rows"], "operation": operation, "table": table}

    def module_effectiveness(self, candidate_id: str, universe_id: str, fold: int | None) -> dict[str, Any]:
        modules = self._common_filter(self.read("fact_module_trace"), "date_value", candidate_id, fold, universe_id)
        outcomes = self._common_filter(self.read("fact_outcome"), "decision_date", candidate_id, fold, universe_id)
        if modules.is_empty():
            return {"activation": [], "by_horizon": [], "timeline": []}
        joined = modules.join(
            outcomes,
            left_on=["date_value", "candidate_id", "fold", "universe_id"],
            right_on=["decision_date", "candidate_id", "fold", "universe_id"],
            how="left",
        )
        activation = modules.group_by("module_name").agg(
            pl.col("module_active").cast(pl.Int8).mean().alias("activation_rate"),
            pl.col("intensity_score").mean().alias("avg_intensity"),
            pl.len().alias("observations"),
        ).sort("activation_rate", descending=True)
        by_horizon = joined.group_by(["module_name", "horizon"]).agg(
            pl.col("module_active").cast(pl.Int8).mean().alias("activation_rate"),
            pl.col("helped_flag").cast(pl.Int8).mean().alias("helped_rate"),
            pl.col("alpha_vs_qqq").mean().alias("avg_alpha_vs_qqq"),
            pl.len().alias("observations"),
        ).sort(["module_name", "horizon"])
        timeline = modules.group_by(["date_value", "module_name"]).agg(pl.col("module_active").cast(pl.Int8).mean().alias("activation_rate")).sort("date_value")
        return {"activation": records(activation, 50)["rows"], "by_horizon": records(by_horizon, 200)["rows"], "timeline": records(timeline, 1000)["rows"]}

    def ticker_contribution(self, candidate_id: str, universe_id: str, fold: int | None, limit: int = 200) -> dict[str, Any]:
        df = self._common_filter(self.read("fact_position_daily"), "date_value", candidate_id, fold, universe_id)
        if df.is_empty():
            return {"count": 0, "rows": []}
        out = df.group_by("ticker").agg(
            pl.col("pnl_contribution").sum().alias("total_pnl_contribution"),
            pl.col("selected_flag").cast(pl.Int8).mean().alias("selection_rate"),
            pl.col("leader_flag").cast(pl.Int8).mean().alias("leader_flag_rate"),
            pl.col("final_weight").mean().alias("avg_final_weight"),
            pl.col("final_score").mean().alias("avg_score"),
            pl.len().alias("observations"),
        ).sort("total_pnl_contribution", descending=True)
        return records(out, limit)

    def regime_behavior(self, candidate_id: str, universe_id: str, fold: int | None) -> dict[str, Any]:
        df = self._common_filter(self.read("fact_decision_state"), "date_value", candidate_id, fold, universe_id)
        if df.is_empty():
            return {"count": 0, "rows": []}
        out = df.group_by(["regime", "participation_state"]).agg(
            pl.col("net_return").mean().alias("avg_net_return"),
            pl.col("benchmark_return").mean().alias("avg_benchmark_return"),
            pl.col("expected_exposure").mean().alias("avg_exposure"),
            pl.col("drawdown").mean().alias("avg_drawdown"),
            pl.col("backoff_flag").cast(pl.Int8).mean().alias("backoff_activation"),
            pl.col("continuation_trigger_flag").cast(pl.Int8).mean().alias("continuation_activation"),
            pl.col("leader_blend").mean().alias("avg_leader_blend"),
            pl.len().alias("observations"),
        ).sort("observations", descending=True)
        return records(out, 200)

    def fold_performance(self, candidate_id: str | None, universe_id: str | None) -> dict[str, Any]:
        df = self._common_filter(self.read("fact_outcome"), candidate_id=candidate_id, universe_id=universe_id)
        if df.is_empty():
            return {"count": 0, "rows": []}
        out = df.group_by(["candidate_id", "universe_id", "fold", "horizon"]).agg(
            pl.col("realized_return").mean().alias("avg_realized_return"),
            pl.col("alpha_vs_qqq").mean().alias("avg_alpha_vs_qqq"),
            pl.col("helped_flag").cast(pl.Int8).mean().alias("helped_rate"),
            pl.col("realized_exposure").mean().alias("avg_exposure"),
            pl.len().alias("observations"),
        ).sort(["fold", "horizon"])
        return records(out, 500)

    def candidate_compare(self, candidates: list[str] | None, universe_id: str | None) -> dict[str, Any]:
        df = self.read("fact_candidate_metric")
        if candidates:
            df = df.filter(pl.col("candidate_id").is_in(candidates))
        if universe_id:
            df = df.filter(pl.col("universe_id") == universe_id)
        df = df.sort(["sharpe", "cagr"], descending=True) if not df.is_empty() else df
        return records(df, 500)

    def query_performance(self) -> dict[str, Any]:
        if not self.query_log:
            return {"count": 0, "rows": []}
        return records(pl.DataFrame(self.query_log).sort("created_at", descending=True), 500)
