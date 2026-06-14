from __future__ import annotations

import time
from typing import Any


class PostgresBackend:
    backend_name = "postgres"

    def __init__(self, database_url: str):
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for the Postgres backend")
        import psycopg
        from psycopg.rows import dict_row

        self.database_url = database_url
        self._psycopg = psycopg
        self._row_factory = dict_row

    def timed(self, endpoint: str, source_relation: str, fn):
        started = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - started) * 1000
        rows = result.get("count", 0) if isinstance(result, dict) else 0
        try:
            self._query(
                """
                INSERT INTO oltp.dss_query_log
                    (query_id, endpoint, backend, source_relation, rows_returned, elapsed_ms, used_materialized_view, demo_mode)
                VALUES
                    (%(query_id)s, %(endpoint)s, 'postgres', %(source_relation)s, %(rows_returned)s, %(elapsed_ms)s,
                     %(used_materialized_view)s, %(demo_mode)s)
                """,
                {
                    "query_id": f"pg_{int(time.time() * 1000)}",
                    "endpoint": endpoint,
                    "source_relation": source_relation,
                    "rows_returned": rows,
                    "elapsed_ms": elapsed,
                    "used_materialized_view": source_relation.startswith("mart."),
                    "demo_mode": self.demo_mode(),
                },
                fetch=False,
            )
        except Exception:
            pass
        return result

    def _query(self, sql: str, params: dict[str, Any] | None = None, fetch: bool = True) -> dict[str, Any]:
        with self._psycopg.connect(self.database_url, row_factory=self._row_factory) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                rows = cur.fetchall() if fetch and cur.description else []
            conn.commit()
        return {"count": len(rows), "rows": rows}

    def demo_mode(self) -> bool:
        result = self._query("SELECT bool_or(demo_mode) AS demo_mode FROM mart.mv_whatif_grid")
        return bool(result["rows"][0]["demo_mode"]) if result["rows"] else False

    def row_counts(self) -> dict[str, int]:
        result = self._query(
            """
            SELECT schemaname || '.' || relname AS table_name, n_live_tup::bigint AS row_count
            FROM pg_stat_user_tables
            WHERE schemaname IN ('dw', 'mart', 'oltp')
            """
        )
        return {row["table_name"]: int(row["row_count"]) for row in result["rows"]}

    def options(self) -> dict[str, Any]:
        return {
            "candidates": [row["candidate_id"] for row in self._query("SELECT candidate_id FROM dw.dim_candidate ORDER BY candidate_id")["rows"]],
            "universes": [row["universe_id"] for row in self._query("SELECT universe_id FROM dw.dim_universe ORDER BY universe_id")["rows"]],
            "folds": [row["fold"] for row in self._query("SELECT fold FROM dw.dim_fold ORDER BY fold")["rows"]],
            "tickers": [row["ticker"] for row in self._query("SELECT ticker FROM dw.dim_asset ORDER BY ticker")["rows"]],
            "modules": [row["module_name"] for row in self._query("SELECT module_name FROM dw.dim_module ORDER BY display_order")["rows"]],
            "horizons": [row["horizon"] for row in self._query("SELECT horizon FROM dw.dim_horizon ORDER BY horizon")["rows"]],
            "regimes": [row["regime_name"] for row in self._query("SELECT regime_name FROM dw.dim_regime ORDER BY regime_name")["rows"]],
            "metrics": [row["metric_name"] for row in self._query("SELECT metric_name FROM dw.dim_metric ORDER BY metric_name")["rows"]],
            "benchmarks": ["QQQ", "SPY", "CONTROL"],
        }

    def scorecard(self, candidate_id=None, universe_id=None, limit=200):
        where = []
        params = {"limit": limit}
        if candidate_id:
            where.append("candidate_id = %(candidate_id)s")
            params["candidate_id"] = candidate_id
        if universe_id:
            where.append("universe_id = %(universe_id)s")
            params["universe_id"] = universe_id
        sql = "SELECT * FROM mart.mv_scorecard_candidate"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY sharpe DESC NULLS LAST LIMIT %(limit)s"
        return self._query(sql, params)

    def overview(self, candidate_id, fold, universe_id, benchmark, start_date, end_date):
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold, "start_date": start_date, "end_date": end_date}
        fold_sql = "AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)"
        date_sql = "AND (%(start_date)s::date IS NULL OR date_value >= %(start_date)s::date) AND (%(end_date)s::date IS NULL OR date_value <= %(end_date)s::date)"
        equity = self._query(
            f"""
            SELECT date_value, equity, drawdown, rolling_peak
            FROM mart.mv_drawdown_replay
            WHERE candidate_id = %(candidate_id)s {fold_sql} {date_sql}
            ORDER BY date_value
            """,
            params,
        )["rows"]
        exposure = self._query(
            f"""
            SELECT DISTINCT date_value, expected_exposure, expected_turnover, long_budget, participation_state
            FROM mart.mv_decision_replay
            WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s {fold_sql} {date_sql}
            ORDER BY date_value
            """,
            params,
        )["rows"]
        return {
            "backend": self.backend_name,
            "demo_mode": self.demo_mode(),
            "scorecard": self.scorecard(candidate_id, universe_id, 20)["rows"],
            "equity_curve": equity,
            "exposure_turnover": exposure,
            "decision_summary": {"benchmark": benchmark, "observations": len(exposure)},
            "fold_performance": self.fold_performance(candidate_id, universe_id)["rows"],
        }

    def robustness_surface(self, metric, fold, universe_id, regime, limit=5000):
        return self._query(
            """
            SELECT *
            FROM mart.mv_robustness_surface
            WHERE metric_name = %(metric)s
              AND (%(fold)s::int IS NULL OR fold IS NULL OR fold = %(fold)s::int)
              AND (%(universe_id)s::text IS NULL OR universe_id = %(universe_id)s::text)
              AND (%(regime)s::text IS NULL OR regime IS NULL OR regime = %(regime)s::text)
            ORDER BY budget_multiplier, conviction_multiplier
            LIMIT %(limit)s
            """,
            {"metric": metric, "fold": fold, "universe_id": universe_id, "regime": regime, "limit": limit},
        )

    def whatif_grid(self, candidate_id, fold, universe_id, horizon, cost_bps, slippage_bps, limit=5000):
        params = {
            "candidate_id": candidate_id,
            "fold": fold,
            "universe_id": universe_id,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "limit": limit,
        }
        rows = self._query(
            """
            SELECT *
            FROM mart.mv_whatif_grid
            WHERE candidate_id = %(candidate_id)s
              AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
              AND universe_id = %(universe_id)s
              AND horizon = %(horizon)s
              AND (%(cost_bps)s::double precision IS NULL OR cost_bps = %(cost_bps)s::double precision)
              AND (%(slippage_bps)s::double precision IS NULL OR slippage_bps = %(slippage_bps)s::double precision)
            ORDER BY robust_score DESC NULLS LAST, sharpe DESC NULLS LAST
            LIMIT %(limit)s
            """,
            params,
        )["rows"]
        pareto = [row for row in rows if (row.get("cagr") or 0) > 0 and (row.get("maxdd") or -999) > -30][:80]
        return {"count": len(rows), "rows": rows, "pareto": pareto, "demo_rows": sum(1 for row in rows if row.get("demo_mode"))}

    def decision_replay(self, candidate_id, fold, universe_id, date_value, ticker):
        params = {"candidate_id": candidate_id, "fold": fold, "universe_id": universe_id, "date_value": date_value, "ticker": ticker}
        decision_rows = self._query(
            """
            SELECT DISTINCT ON (date_value, candidate_id, fold, universe_id)
                *
            FROM mart.mv_decision_replay
            WHERE candidate_id = %(candidate_id)s
              AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
              AND universe_id = %(universe_id)s
              AND (%(date_value)s::date IS NULL OR date_value = %(date_value)s::date)
            ORDER BY date_value, candidate_id, fold, universe_id
            LIMIT 1
            """,
            params,
        )["rows"]
        if not decision_rows:
            return {"decision": None, "positions": [], "modules": [], "outcomes": [], "market_context": [], "timeline": []}
        decision = decision_rows[0]
        params["date_value"] = decision["date_value"]
        params["fold"] = decision["fold"]
        positions = self._query(
            """
            SELECT *
            FROM dw.fact_position_daily
            WHERE candidate_id = %(candidate_id)s AND fold = %(fold)s AND universe_id = %(universe_id)s
              AND date_value = %(date_value)s
              AND (%(ticker)s::text IS NULL OR ticker = %(ticker)s::text)
            ORDER BY final_weight DESC NULLS LAST
            LIMIT 50
            """,
            params,
        )["rows"]
        modules = self._query(
            """
            SELECT * FROM dw.fact_module_trace
            WHERE candidate_id = %(candidate_id)s AND fold = %(fold)s AND universe_id = %(universe_id)s AND date_value = %(date_value)s
            ORDER BY module_name
            """,
            params,
        )["rows"]
        outcomes = self._query(
            """
            SELECT * FROM dw.fact_outcome
            WHERE candidate_id = %(candidate_id)s AND fold = %(fold)s AND universe_id = %(universe_id)s AND decision_date = %(date_value)s
            ORDER BY horizon
            """,
            params,
        )["rows"]
        market = self._query(
            "SELECT * FROM dw.fact_market_bar WHERE date_value = %(date_value)s AND ticker IN ('QQQ', 'SPY') ORDER BY ticker",
            params,
        )["rows"]
        return {"decision": decision, "positions": positions, "modules": modules, "outcomes": outcomes, "market_context": market, "timeline": []}

    def slice_query(self, dimensions, measure, operation, candidate_id, fold, universe_id, module, ticker, regime, horizon, start_date, end_date, limit=500):
        dim_sql = ", ".join(dimensions)
        table = "dw.fact_outcome"
        date_col = "decision_date"
        measure_sql = {
            "return": "avg(realized_return)",
            "alpha": "avg(alpha_vs_qqq)",
            "helped_rate": "avg(helped_flag::int)",
            "drawdown": "avg(drawdown)",
            "exposure": "avg(expected_exposure)",
            "turnover": "avg(expected_turnover)",
        }[measure]
        if ticker or "ticker" in dimensions:
            table, date_col = "dw.fact_position_daily", "date_value"
            measure_sql = "sum(pnl_contribution)" if measure in {"return", "alpha"} else "avg(final_weight)"
        elif module or "module_name" in dimensions:
            table, date_col = "dw.fact_module_trace", "date_value"
            measure_sql = "avg(module_active::int)"
        elif measure in {"drawdown", "exposure", "turnover"} or regime or "regime" in dimensions:
            table, date_col = "dw.fact_decision_state", "date_value"
        horizon_cond = "AND (%(horizon)s::int IS NULL OR horizon = %(horizon)s::int)" if table == "dw.fact_outcome" else ""
        return self._query(
            f"""
            SELECT {dim_sql}, {measure_sql} AS "{measure}", count(*) AS observations
            FROM {table}
            WHERE (%(candidate_id)s::text IS NULL OR candidate_id = %(candidate_id)s::text)
              AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
              AND (%(universe_id)s::text IS NULL OR universe_id = %(universe_id)s::text)
              {horizon_cond}
              AND (%(start_date)s::date IS NULL OR {date_col} >= %(start_date)s::date)
              AND (%(end_date)s::date IS NULL OR {date_col} <= %(end_date)s::date)
            GROUP BY {dim_sql}
            ORDER BY "{measure}" DESC NULLS LAST
            LIMIT %(limit)s
            """,
            {
                "candidate_id": candidate_id,
                "fold": fold,
                "universe_id": universe_id,
                "horizon": horizon,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
            },
        )

    def module_effectiveness(self, candidate_id, universe_id, fold):
        return {
            "activation": self._query(
                """
                SELECT module_name, avg(activation_rate) AS activation_rate, sum(observations) AS observations
                FROM mart.mv_module_effectiveness
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
                GROUP BY module_name
                ORDER BY activation_rate DESC NULLS LAST
                """,
                {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold},
            )["rows"],
            "by_horizon": self._query(
                """
                SELECT * FROM mart.mv_module_effectiveness
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
                ORDER BY module_name, horizon
                """,
                {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold},
            )["rows"],
            "timeline": [],
        }

    def ticker_contribution(self, candidate_id, universe_id, fold, limit=200):
        return self._query(
            """
            SELECT * FROM mart.mv_ticker_contribution
            WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
            ORDER BY total_pnl_contribution DESC NULLS LAST
            LIMIT %(limit)s
            """,
            {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold, "limit": limit},
        )

    def regime_behavior(self, candidate_id, universe_id, fold):
        return self._query(
            """
            SELECT * FROM mart.mv_regime_behavior
            WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
            ORDER BY observations DESC
            """,
            {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold},
        )

    def fold_performance(self, candidate_id, universe_id):
        return self._query(
            """
            SELECT * FROM mart.mv_performance_by_fold
            WHERE (%(candidate_id)s::text IS NULL OR candidate_id = %(candidate_id)s::text)
              AND (%(universe_id)s::text IS NULL OR universe_id = %(universe_id)s::text)
            ORDER BY fold, avg_alpha_vs_qqq DESC NULLS LAST
            """,
            {"candidate_id": candidate_id, "universe_id": universe_id},
        )

    def candidate_compare(self, candidates, universe_id):
        return self._query(
            """
            SELECT *
            FROM dw.fact_candidate_metric
            WHERE (%(universe_id)s::text IS NULL OR universe_id = %(universe_id)s::text)
              AND (%(candidates)s::text[] IS NULL OR candidate_id = ANY(%(candidates)s::text[]))
            ORDER BY sharpe DESC NULLS LAST, cagr DESC NULLS LAST
            LIMIT 500
            """,
            {"candidates": candidates, "universe_id": universe_id},
        )

    def query_performance(self):
        try:
            return self._query(
                """
                SELECT
                    endpoint,
                    backend,
                    source_relation,
                    used_materialized_view,
                    count(*) AS query_count,
                    avg(elapsed_ms) AS avg_elapsed_ms,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95_elapsed_ms,
                    avg(rows_returned) AS avg_rows_returned,
                    max(created_at) AS last_seen_at
                FROM oltp.dss_query_log
                GROUP BY endpoint, backend, source_relation, used_materialized_view
                ORDER BY last_seen_at DESC NULLS LAST
                """
            )
        except Exception:
            return {"count": 0, "rows": []}
