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
        date_bounds = self._query("SELECT MIN(date_value) AS start_date, MAX(date_value) AS end_date FROM dw.dim_date")["rows"]
        date_row = date_bounds[0] if date_bounds else {}
        start_date = date_row.get("start_date")
        end_date = date_row.get("end_date")
        ranges = self._query(
            """
            SELECT
                array_agg(DISTINCT budget_multiplier ORDER BY budget_multiplier) FILTER (WHERE budget_multiplier IS NOT NULL) AS budget_multiplier,
                array_agg(DISTINCT conviction_multiplier ORDER BY conviction_multiplier) FILTER (WHERE conviction_multiplier IS NOT NULL) AS conviction_multiplier,
                array_agg(DISTINCT leader_multiplier ORDER BY leader_multiplier) FILTER (WHERE leader_multiplier IS NOT NULL) AS leader_multiplier,
                array_agg(DISTINCT backoff_strength ORDER BY backoff_strength) FILTER (WHERE backoff_strength IS NOT NULL) AS backoff_strength,
                array_agg(DISTINCT cost_bps ORDER BY cost_bps) FILTER (WHERE cost_bps IS NOT NULL) AS cost_bps,
                array_agg(DISTINCT slippage_bps ORDER BY slippage_bps) FILTER (WHERE slippage_bps IS NOT NULL) AS slippage_bps
            FROM mart.mv_whatif_grid
            """
        )["rows"]
        slider_ranges = {}
        for key, values in (ranges[0] if ranges else {}).items():
            if values:
                slider_ranges[key] = {"min": min(values), "max": max(values), "values": values}
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
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "slider_ranges": slider_ranges,
            "default_candidate": "B1.05_C1.10_L1.10_R1.05",
            "default_universe": "base_universe_12",
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
        options = self._query(
            """
            SELECT
                array_agg(DISTINCT fold ORDER BY fold) FILTER (WHERE fold IS NOT NULL) AS folds,
                array_agg(DISTINCT horizon ORDER BY horizon) FILTER (WHERE horizon IS NOT NULL) AS horizons,
                array_agg(DISTINCT cost_bps ORDER BY cost_bps) FILTER (WHERE cost_bps IS NOT NULL) AS costs,
                array_agg(DISTINCT slippage_bps ORDER BY slippage_bps) FILTER (WHERE slippage_bps IS NOT NULL) AS slippages,
                array_agg(DISTINCT budget_multiplier ORDER BY budget_multiplier) FILTER (WHERE budget_multiplier IS NOT NULL) AS budgets,
                array_agg(DISTINCT conviction_multiplier ORDER BY conviction_multiplier) FILTER (WHERE conviction_multiplier IS NOT NULL) AS convictions,
                array_agg(DISTINCT leader_multiplier ORDER BY leader_multiplier) FILTER (WHERE leader_multiplier IS NOT NULL) AS leaders,
                array_agg(DISTINCT backoff_strength ORDER BY backoff_strength) FILTER (WHERE backoff_strength IS NOT NULL) AS backoffs
            FROM mart.mv_whatif_grid
            WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s
            """,
            params,
        )["rows"]
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
        clean_rows = [row for row in rows if row.get("cagr") is not None and row.get("sharpe") is not None and row.get("maxdd") is not None]
        pareto = [row for row in clean_rows if (row.get("cagr") or 0) > 0 and (row.get("maxdd") or -999) > -30][:80]
        return {"count": len(clean_rows), "rows": clean_rows, "pareto": pareto, "demo_rows": sum(1 for row in clean_rows if row.get("demo_mode")), "available": options[0] if options else {}}

    def whatif_reference(self, candidate_id, universe_id, fold, horizon, cost_bps, slippage_bps):
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold, "horizon": horizon, "cost_bps": cost_bps, "slippage_bps": slippage_bps}
        official = self._query(
            """
            SELECT candidate_id, cagr, sharpe, sortino, maxdd, alpha_qqq, avg_exposure, avg_turnover, false AS demo_mode
            FROM mart.mv_scorecard_candidate
            WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s
            LIMIT 1
            """,
            params,
        )["rows"]
        best_observed = self._query(
            """
            SELECT candidate_id, cagr, sharpe, sortino, maxdd, alpha_qqq, avg_exposure, avg_turnover, false AS demo_mode
            FROM mart.mv_scorecard_candidate
            WHERE universe_id = %(universe_id)s
            ORDER BY sharpe DESC NULLS LAST, cagr DESC NULLS LAST
            LIMIT 1
            """,
            params,
        )["rows"]
        best_simulated = self._query(
            """
            SELECT *
            FROM mart.mv_whatif_grid
            WHERE candidate_id = %(candidate_id)s
              AND universe_id = %(universe_id)s
              AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
              AND horizon = %(horizon)s
              AND (%(cost_bps)s::double precision IS NULL OR cost_bps = %(cost_bps)s::double precision)
              AND (%(slippage_bps)s::double precision IS NULL OR slippage_bps = %(slippage_bps)s::double precision)
              AND cagr IS NOT NULL AND sharpe IS NOT NULL AND maxdd IS NOT NULL
            ORDER BY robust_score DESC NULLS LAST, sharpe DESC NULLS LAST
            LIMIT 1
            """,
            params,
        )["rows"]
        return {"official": official[0] if official else None, "best_observed": best_observed[0] if best_observed else None, "best_simulated": best_simulated[0] if best_simulated else None}

    def research_distributions(self, candidate_id, universe_id, fold):
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold}
        outcome_percentiles = self._query(
            """
            SELECT
                horizon,
                count(*) AS observations,
                avg(realized_return) AS avg_outcome,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY realized_return) AS p5_outcome,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY realized_return) AS p25_outcome,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY realized_return) AS median_outcome,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY realized_return) AS p75_outcome,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY realized_return) AS p95_outcome,
                avg(helped_flag::int) AS helped_rate,
                avg(alpha_vs_qqq) AS avg_alpha_vs_qqq,
                avg(alpha_vs_spy) AS avg_alpha_vs_spy
            FROM dw.fact_outcome
            WHERE candidate_id = %(candidate_id)s
              AND universe_id = %(universe_id)s
              AND (%(fold)s::int IS NULL OR fold = %(fold)s::int)
              AND realized_return IS NOT NULL
            GROUP BY horizon
            ORDER BY horizon
            """,
            params,
        )["rows"]
        decision_percentiles = self._query(
            """
            WITH metrics AS (
                SELECT 'Exposure' AS metric, expected_exposure AS value FROM dw.fact_decision_state
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int) AND expected_exposure IS NOT NULL
                UNION ALL
                SELECT 'Turnover', expected_turnover FROM dw.fact_decision_state
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int) AND expected_turnover IS NOT NULL
                UNION ALL
                SELECT 'Drawdown', drawdown FROM dw.fact_decision_state
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s AND (%(fold)s::int IS NULL OR fold = %(fold)s::int) AND drawdown IS NOT NULL
            )
            SELECT
                metric,
                count(*) AS observations,
                avg(value) AS average,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY value) AS p5,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY value) AS p25,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY value) AS median,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY value) AS p75,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
            FROM metrics
            GROUP BY metric
            ORDER BY metric
            """,
            params,
        )["rows"]
        exposure_buckets = self._query(
            """
            WITH joined AS (
                SELECT
                    CASE
                        WHEN d.expected_exposure < 0.50 THEN 'Low exposure'
                        WHEN d.expected_exposure < 0.80 THEN 'Mid exposure'
                        ELSE 'High exposure'
                    END AS bucket,
                    o.realized_return,
                    o.alpha_vs_qqq,
                    o.helped_flag,
                    d.expected_exposure,
                    d.drawdown
                FROM dw.fact_decision_state d
                JOIN dw.fact_outcome o
                  ON o.decision_date = d.date_value
                 AND o.candidate_id = d.candidate_id
                 AND o.fold = d.fold
                 AND o.universe_id = d.universe_id
                WHERE d.candidate_id = %(candidate_id)s
                  AND d.universe_id = %(universe_id)s
                  AND (%(fold)s::int IS NULL OR d.fold = %(fold)s::int)
                  AND o.horizon = 20
                  AND d.expected_exposure IS NOT NULL
                  AND o.realized_return IS NOT NULL
            )
            SELECT
                bucket,
                count(*) AS observations,
                avg(realized_return) AS avg_outcome,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY realized_return) AS median_outcome,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY realized_return) AS p5_outcome,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY realized_return) AS p95_outcome,
                avg(helped_flag::int) AS helped_rate,
                avg(alpha_vs_qqq) AS avg_alpha_vs_qqq,
                avg(expected_exposure) AS avg_exposure,
                avg(drawdown) AS avg_drawdown
            FROM joined
            GROUP BY bucket
            ORDER BY avg_exposure
            """,
            params,
        )["rows"]
        return {"outcome_percentiles": outcome_percentiles, "decision_percentiles": decision_percentiles, "exposure_buckets": exposure_buckets}

    def research_cohorts(self, candidate_id, universe_id, fold):
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold}
        rows = self._query(
            """
            WITH joined AS (
                SELECT
                    d.fold,
                    d.regime,
                    d.participation_state,
                    d.backoff_flag,
                    d.leader_blend,
                    d.expected_exposure,
                    d.expected_turnover,
                    d.drawdown,
                    o.horizon,
                    o.realized_return,
                    o.alpha_vs_qqq,
                    o.helped_flag
                FROM dw.fact_decision_state d
                JOIN dw.fact_outcome o
                  ON o.decision_date = d.date_value
                 AND o.candidate_id = d.candidate_id
                 AND o.fold = d.fold
                 AND o.universe_id = d.universe_id
                WHERE d.candidate_id = %(candidate_id)s
                  AND d.universe_id = %(universe_id)s
                  AND (%(fold)s::int IS NULL OR d.fold = %(fold)s::int)
                  AND o.realized_return IS NOT NULL
            ),
            cohort_rows AS (
                SELECT 'Backoff' AS cohort_type, CASE WHEN backoff_flag THEN 'Backoff active' ELSE 'Backoff inactive' END AS cohort, * FROM joined
                UNION ALL SELECT 'Leader participation', CASE WHEN coalesce(leader_blend, 0) >= 0.20 THEN 'High leader blend' ELSE 'Low leader blend' END, * FROM joined
                UNION ALL SELECT 'Exposure', CASE WHEN coalesce(expected_exposure, 0) >= 0.80 THEN 'High exposure' ELSE 'Low / mid exposure' END, * FROM joined
                UNION ALL SELECT 'Turnover', CASE WHEN coalesce(expected_turnover, 0) >= 0.03 THEN 'High turnover' ELSE 'Low turnover' END, * FROM joined
                UNION ALL SELECT 'Fold', 'Fold ' || fold::text, * FROM joined
                UNION ALL SELECT 'Regime', coalesce(regime, 'Unlabeled regime'), * FROM joined
                UNION ALL SELECT 'Participation state', coalesce(participation_state, 'Unlabeled state'), * FROM joined
                UNION ALL SELECT 'Horizon', horizon::text || 'd horizon', * FROM joined
            )
            SELECT
                cohort_type,
                cohort,
                count(*) AS observations,
                avg(realized_return) AS avg_outcome,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY realized_return) AS median_outcome,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY realized_return) AS p5_outcome,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY realized_return) AS p95_outcome,
                avg(helped_flag::int) AS helped_rate,
                avg(alpha_vs_qqq) AS avg_alpha_vs_qqq,
                avg(expected_exposure) AS avg_exposure,
                avg(drawdown) AS avg_drawdown
            FROM cohort_rows
            GROUP BY cohort_type, cohort
            HAVING count(*) >= 5
            ORDER BY cohort_type, observations DESC
            """,
            params,
        )["rows"]
        return {"count": len(rows), "rows": rows}

    def decision_replay(self, candidate_id, fold, universe_id, date_value, ticker):
        params = {"candidate_id": candidate_id, "fold": fold, "universe_id": universe_id, "date_value": date_value, "ticker": ticker}
        decision_rows = self._query(
            """
            WITH candidates AS (
                SELECT
                    d.*,
                    o.horizon,
                    o.realized_return,
                    o.alpha_vs_qqq,
                    o.alpha_vs_spy,
                    o.helped_flag,
                    (SELECT count(*) FROM dw.fact_position_daily p
                     WHERE p.candidate_id = d.candidate_id AND p.fold = d.fold AND p.universe_id = d.universe_id AND p.date_value = d.date_value) AS position_count,
                    (SELECT count(*) FROM dw.fact_module_trace m
                     WHERE m.candidate_id = d.candidate_id AND m.fold = d.fold AND m.universe_id = d.universe_id AND m.date_value = d.date_value) AS module_count,
                    (SELECT count(*) FROM dw.fact_outcome o
                     WHERE o.candidate_id = d.candidate_id AND o.fold = d.fold AND o.universe_id = d.universe_id AND o.decision_date = d.date_value) AS outcome_count
                FROM dw.fact_decision_state d
                LEFT JOIN dw.fact_outcome o
                  ON o.candidate_id = d.candidate_id
                 AND o.fold = d.fold
                 AND o.universe_id = d.universe_id
                 AND o.decision_date = d.date_value
                 AND o.horizon = 20
                WHERE d.candidate_id = %(candidate_id)s
                  AND (%(fold)s::int IS NULL OR d.fold = %(fold)s::int)
                  AND d.universe_id = %(universe_id)s
                  AND (%(date_value)s::date IS NULL OR d.date_value = %(date_value)s::date)
            ),
            collapsed AS (
                SELECT DISTINCT ON (date_value, candidate_id, fold, universe_id) *
                FROM candidates
                ORDER BY date_value, candidate_id, fold, universe_id
            )
            SELECT *
            FROM collapsed
            ORDER BY position_count DESC, module_count DESC, outcome_count DESC, abs(coalesce(alpha_vs_qqq, 0)) DESC
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
        available_tickers = self._query(
            """
            SELECT ticker, final_weight, final_score, selected_flag, leader_flag
            FROM dw.fact_position_daily
            WHERE candidate_id = %(candidate_id)s AND fold = %(fold)s AND universe_id = %(universe_id)s
              AND date_value = %(date_value)s
            ORDER BY final_weight DESC NULLS LAST, final_score DESC NULLS LAST
            LIMIT 80
            """,
            params,
        )["rows"]
        if ticker and not positions:
            params["ticker"] = None
            positions = available_tickers[:50]
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
        outcomes = [row for row in outcomes if row.get("horizon") is not None and (row.get("realized_return") is not None or row.get("alpha_vs_qqq") is not None)]
        return {"decision": decision, "positions": positions, "available_tickers": available_tickers, "modules": modules, "outcomes": outcomes, "market_context": market, "timeline": []}

    def decision_casebook(self, candidate_id, universe_id, fold):
        return self._query(
            """
            WITH base AS (
                SELECT
                    d.date_value,
                    d.candidate_id,
                    d.fold,
                    d.universe_id,
                    d.regime,
                    d.participation_state,
                    d.expected_exposure,
                    d.expected_turnover,
                    d.leader_blend,
                    d.hard_backoff_flag,
                    o.realized_return AS return_20d,
                    o.alpha_vs_qqq AS alpha_20d,
                    (SELECT count(*) FROM dw.fact_position_daily p WHERE p.candidate_id=d.candidate_id AND p.fold=d.fold AND p.universe_id=d.universe_id AND p.date_value=d.date_value) AS positions,
                    (SELECT max(final_weight) FROM dw.fact_position_daily p WHERE p.candidate_id=d.candidate_id AND p.fold=d.fold AND p.universe_id=d.universe_id AND p.date_value=d.date_value) AS max_weight,
                    (SELECT bool_or(leader_flag) FROM dw.fact_position_daily p WHERE p.candidate_id=d.candidate_id AND p.fold=d.fold AND p.universe_id=d.universe_id AND p.date_value=d.date_value) AS has_leader,
                    (SELECT min(pnl_contribution) FROM dw.fact_position_daily p WHERE p.candidate_id=d.candidate_id AND p.fold=d.fold AND p.universe_id=d.universe_id AND p.date_value=d.date_value) AS largest_ticker_drag,
                    (SELECT count(*) FROM dw.fact_module_trace m WHERE m.candidate_id=d.candidate_id AND m.fold=d.fold AND m.universe_id=d.universe_id AND m.date_value=d.date_value) AS modules
                FROM dw.fact_decision_state d
                JOIN dw.fact_outcome o
                  ON o.candidate_id=d.candidate_id AND o.fold=d.fold AND o.universe_id=d.universe_id
                 AND o.decision_date=d.date_value AND o.horizon=20
                WHERE d.candidate_id = %(candidate_id)s
                  AND d.universe_id = %(universe_id)s
                  AND (%(fold)s::int IS NULL OR d.fold = %(fold)s::int)
            )
            SELECT DISTINCT ON (case_label) *
            FROM (
                (SELECT 'Best 20d outcome' AS case_label, 'Highest realized 20d return with complete replay rows.' AS rationale, * FROM base WHERE return_20d IS NOT NULL ORDER BY return_20d DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Worst 20d outcome', 'Lowest realized 20d return with complete replay rows.', * FROM base WHERE return_20d IS NOT NULL ORDER BY return_20d ASC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Highest alpha vs QQQ', 'Best 20d benchmark-adjusted outcome.', * FROM base WHERE alpha_20d IS NOT NULL ORDER BY alpha_20d DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Worst alpha vs QQQ', 'Weakest 20d benchmark-adjusted outcome.', * FROM base WHERE alpha_20d IS NOT NULL ORDER BY alpha_20d ASC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Highest exposure decision', 'Maximum expected exposure with positions and outcomes.', * FROM base WHERE expected_exposure IS NOT NULL ORDER BY expected_exposure DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Highest turnover decision', 'Maximum expected turnover with positions and outcomes.', * FROM base WHERE expected_turnover IS NOT NULL ORDER BY expected_turnover DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Most concentrated allocation', 'Largest single final weight in the selected portfolio.', * FROM base WHERE max_weight IS NOT NULL ORDER BY max_weight DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Strongest leader participation', 'Largest leader blend / leader flag case.', * FROM base WHERE leader_blend IS NOT NULL OR has_leader ORDER BY leader_blend DESC NULLS LAST, has_leader DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Backoff with positive outcome', 'Backoff active while 20d outcome stayed positive.', * FROM base WHERE hard_backoff_flag AND return_20d > 0 ORDER BY return_20d DESC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Backoff with missed upside', 'Backoff active with positive QQQ-relative miss.', * FROM base WHERE hard_backoff_flag AND alpha_20d < 0 ORDER BY alpha_20d ASC NULLS LAST LIMIT 1)
            UNION ALL (SELECT 'Largest ticker drag decision', 'Decision containing the largest single-name drag.', * FROM base WHERE largest_ticker_drag IS NOT NULL ORDER BY largest_ticker_drag ASC NULLS LAST LIMIT 1)
            ) cases
            """,
            {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold},
        )

    def top_wins_drags(self, candidate_id, universe_id):
        return {
            "folds": self.fold_performance(candidate_id, universe_id)["rows"],
            "tickers": self.ticker_contribution(candidate_id, universe_id, None, 200)["rows"],
            "modules": self.module_effectiveness(candidate_id, universe_id, None)["by_horizon"],
            "regimes": self.regime_behavior(candidate_id, universe_id, None)["rows"],
            "candidates": self.scorecard(None, universe_id, 500)["rows"],
        }

    def robustness_compare(self, universe_id):
        return self._query(
            """
            WITH base AS (
                SELECT
                    candidate_id, universe_id, cagr, sharpe, sortino, maxdd, alpha_qqq, NULL::double precision AS robust_score,
                    NULL::int AS severe_fold_damage_count,
                    CASE WHEN candidate_id = 'B1.05_C1.10_L1.10_R1.05' THEN 'Official baseline' ELSE 'Observed candidate' END AS compare_role
                FROM mart.mv_scorecard_candidate
                WHERE universe_id = %(universe_id)s
            ),
            official AS (
                SELECT * FROM base WHERE candidate_id = 'B1.05_C1.10_L1.10_R1.05' LIMIT 1
            ),
            picked AS (
                SELECT * FROM official
                UNION ALL (SELECT * FROM base ORDER BY sharpe DESC NULLS LAST LIMIT 1)
                UNION ALL (SELECT * FROM base ORDER BY cagr DESC NULLS LAST LIMIT 1)
                UNION ALL (SELECT * FROM base ORDER BY maxdd DESC NULLS LAST LIMIT 1)
                UNION ALL (SELECT * FROM base ORDER BY sharpe ASC NULLS LAST LIMIT 1)
            )
            SELECT DISTINCT ON (candidate_id, compare_role)
                p.*,
                p.cagr - o.cagr AS delta_cagr,
                p.sharpe - o.sharpe AS delta_sharpe,
                p.sortino - o.sortino AS delta_sortino,
                p.maxdd - o.maxdd AS delta_maxdd,
                p.alpha_qqq - o.alpha_qqq AS delta_alpha_qqq
            FROM picked p CROSS JOIN official o
            ORDER BY candidate_id, compare_role
            """,
            {"universe_id": universe_id},
        )

    def olap_preset(self, preset_id, candidate_id, universe_id, fold, limit=500):
        presets = {
            "fold-best-performance": ("Which fold contributes most to official performance?", "roll-up", "mart.mv_performance_by_fold", "avg_alpha_vs_qqq", "fold"),
            "fold-worst-drawdown": ("Which fold carries the worst drawdown?", "slice", "mart.mv_performance_by_fold", "avg_realized_return", "fold"),
            "sharpe-stable-folds": ("Is Sharpe stable across folds?", "roll-up", "mart.mv_performance_by_fold", "helped_rate", "fold"),
            "performance-extreme-outcomes": ("Does performance depend on a small number of extreme outcomes?", "roll-up", "research.distributions", "p95_outcome", "horizon"),
            "candidate-cagr-maxdd": ("Which candidate has the best CAGR/MaxDD tradeoff?", "pivot", "mart.mv_scorecard_candidate", "return_per_exposure", "candidate_id"),
            "candidate-best-sharpe": ("Which candidate has the best Sharpe?", "roll-up", "mart.mv_scorecard_candidate", "sharpe", "candidate_id"),
            "candidate-severe-fold-damage": ("Which candidate has severe fold damage?", "dice", "mart.mv_scorecard_candidate", "maxdd", "candidate_id"),
            "axis-degrades-most": ("Which multiplier axis degrades the model most?", "roll-up", "mart.mv_robustness_surface", "metric_value", "sweep_role"),
            "module-helps-horizon": ("Which module helps most by horizon?", "pivot", "mart.mv_module_effectiveness", "helped_rate", "module_name"),
            "module-active-low-value": ("Which module activates often but adds little?", "dice", "mart.mv_module_effectiveness", "activation_rate", "module_name"),
            "module-better-outcomes": ("Which module coincides with better outcomes?", "roll-up", "mart.mv_module_effectiveness", "avg_alpha_vs_qqq", "module_name"),
            "ticker-top-contribution": ("Which tickers contribute most?", "drill-down", "mart.mv_ticker_contribution", "total_pnl_contribution", "ticker"),
            "ticker-largest-drags": ("Which tickers drag most?", "drill-down", "mart.mv_ticker_contribution", "total_pnl_contribution", "ticker"),
            "ticker-selection-low-contribution": ("Which tickers are frequently selected but low contribution?", "dice", "mart.mv_ticker_contribution", "selection_rate", "ticker"),
            "ticker-frequent-leaders": ("Which tickers are frequent leaders?", "roll-up", "mart.mv_ticker_contribution", "leader_flag_rate", "ticker"),
            "regime-best-alpha": ("Which regime has the best alpha proxy?", "slice", "mart.mv_regime_behavior", "avg_net_return", "regime"),
            "regime-exposure-concentration": ("Where is exposure concentrated?", "slice", "mart.mv_regime_behavior", "avg_exposure", "regime"),
            "regime-backoff-most": ("Where does backoff activate most?", "slice", "mart.mv_regime_behavior", "backoff_activation_rate", "regime"),
            "regime-weakest-outcome": ("Which regime has weakest average outcome?", "slice", "mart.mv_regime_behavior", "avg_net_return", "regime"),
            "decision-best-20d": ("Best decisions by 20d outcome.", "drill-through", "mart.mv_decision_outcome", "realized_return", "date_value"),
            "decision-worst-20d": ("Worst decisions by 20d outcome.", "drill-through", "mart.mv_decision_outcome", "realized_return", "date_value"),
            "decision-high-exposure-bad": ("High exposure with bad outcome.", "dice", "mart.mv_decision_outcome", "expected_exposure", "date_value"),
            "decision-backoff-positive": ("Backoff decisions with positive outcome.", "slice", "mart.mv_decision_outcome", "realized_return", "date_value"),
            "decision-backoff-missed-upside": ("Backoff decisions with missed upside.", "slice", "mart.mv_decision_outcome", "alpha_vs_qqq", "date_value"),
            "outcome-percentiles-horizon": ("Outcome percentiles by horizon.", "roll-up", "research.distributions", "median_outcome", "horizon"),
            "exposure-buckets-outcome": ("Exposure buckets vs outcome.", "dice", "research.distributions", "median_outcome", "bucket"),
            "turnover-buckets-outcome": ("Turnover buckets vs outcome.", "dice", "research.cohorts", "median_outcome", "cohort"),
            "drawdown-distribution-regime": ("Drawdown distribution by regime/fold.", "roll-up", "research.cohorts", "avg_drawdown", "cohort"),
            "engineering-slowest-endpoint": ("Which endpoint is slowest?", "roll-up", "oltp.dss_query_log", "avg_elapsed_ms", "endpoint"),
            "engineering-highest-p95": ("Which endpoint has highest p95?", "roll-up", "oltp.dss_query_log", "p95_elapsed_ms", "endpoint"),
            "engineering-source-most-used": ("Which source relation is used most?", "roll-up", "oltp.dss_query_log", "query_count", "source_relation"),
            "engineering-useful-mart": ("Which mart supports most DSS views?", "roll-up", "oltp.dss_query_log", "query_count", "source_relation"),
        }
        question, operation, source, measure, dimension = presets.get(preset_id, presets["fold-best-performance"])
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold, "limit": limit}
        if source == "research.distributions":
            payload = self.research_distributions(candidate_id, universe_id, fold)
            rows = payload["exposure_buckets"] if preset_id == "exposure-buckets-outcome" else payload["outcome_percentiles"]
            rows = [row for row in rows if row.get(measure) is not None]
            return {"preset_id": preset_id, "question": question, "operation": operation, "source": source, "measure": measure, "dimension": dimension, "count": len(rows), "rows": rows}
        if source == "research.cohorts":
            rows = [
                row for row in self.research_cohorts(candidate_id, universe_id, fold)["rows"]
                if (preset_id == "turnover-buckets-outcome" and row.get("cohort_type") == "Turnover")
                or (preset_id == "drawdown-distribution-regime" and row.get("cohort_type") in {"Regime", "Fold"})
            ]
            rows = [row for row in rows if row.get(measure) is not None]
            return {"preset_id": preset_id, "question": question, "operation": operation, "source": source, "measure": measure, "dimension": dimension, "count": len(rows), "rows": rows}
        sql_by_source = {
            "mart.mv_performance_by_fold": """
                SELECT fold, avg(avg_realized_return) AS avg_realized_return, avg(avg_alpha_vs_qqq) AS avg_alpha_vs_qqq,
                       avg(helped_rate) AS helped_rate, sum(observations) AS observations
                FROM mart.mv_performance_by_fold
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
                GROUP BY fold
            """,
            "mart.mv_scorecard_candidate": """
                SELECT candidate_id, cagr, sharpe, sortino, maxdd, alpha_qqq, return_per_exposure, avg_exposure, avg_turnover
                FROM mart.mv_scorecard_candidate
                WHERE universe_id=%(universe_id)s
            """,
            "mart.mv_module_effectiveness": """
                SELECT module_name, horizon, avg(activation_rate) AS activation_rate, avg(helped_rate) AS helped_rate,
                       avg(avg_alpha_vs_qqq) AS avg_alpha_vs_qqq, sum(observations) AS observations
                FROM mart.mv_module_effectiveness
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
                GROUP BY module_name, horizon
            """,
            "mart.mv_ticker_contribution": """
                SELECT ticker, sum(total_pnl_contribution) AS total_pnl_contribution, avg(avg_final_weight) AS avg_final_weight,
                       avg(selection_rate) AS selection_rate, avg(leader_flag_rate) AS leader_flag_rate, sum(observations) AS observations
                FROM mart.mv_ticker_contribution
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
                GROUP BY ticker
            """,
            "mart.mv_regime_behavior": """
                SELECT regime, avg(avg_net_return) AS avg_net_return, avg(avg_benchmark_return) AS avg_benchmark_return,
                       avg(avg_exposure) AS avg_exposure, avg(avg_drawdown) AS avg_drawdown,
                       avg(backoff_activation_rate) AS backoff_activation_rate,
                       avg(continuation_activation_rate) AS continuation_activation_rate, avg(avg_leader_blend) AS avg_leader_blend,
                       sum(observations) AS observations
                FROM mart.mv_regime_behavior
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
                GROUP BY regime
            """,
            "mart.mv_robustness_surface": """
                SELECT sweep_role, avg(metric_value) AS metric_value, avg(robust_score) AS robust_score, count(*) AS observations
                FROM mart.mv_robustness_surface
                WHERE universe_id=%(universe_id)s AND metric_name='Sharpe'
                GROUP BY sweep_role
            """,
            "mart.mv_decision_outcome": """
                SELECT date_value, fold, regime, participation_state, expected_exposure, expected_turnover, horizon,
                       realized_return, alpha_vs_qqq, drawdown_change
                FROM mart.mv_decision_outcome
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND horizon=20
                  AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
                  AND (
                    %(preset_id)s NOT IN ('decision-high-exposure-bad', 'decision-backoff-positive', 'decision-backoff-missed-upside')
                    OR (%(preset_id)s = 'decision-high-exposure-bad' AND expected_exposure >= 0.80 AND realized_return < 0)
                    OR (%(preset_id)s = 'decision-backoff-positive' AND participation_state ILIKE '%%BACKOFF%%' AND realized_return > 0)
                    OR (%(preset_id)s = 'decision-backoff-missed-upside' AND participation_state ILIKE '%%BACKOFF%%' AND alpha_vs_qqq < 0)
                  )
            """,
            "oltp.dss_query_log": """
                SELECT endpoint, source_relation, count(*) AS query_count, avg(elapsed_ms) AS avg_elapsed_ms,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95_elapsed_ms,
                       avg(rows_returned) AS avg_rows_returned
                FROM oltp.dss_query_log
                GROUP BY endpoint, source_relation
            """,
        }
        params["preset_id"] = preset_id
        rows = self._query(f"SELECT * FROM ({sql_by_source[source]}) q ORDER BY {measure} DESC NULLS LAST LIMIT %(limit)s", params)["rows"]
        if preset_id in {"ticker-largest-drags", "decision-worst-20d", "fold-worst-drawdown", "candidate-severe-fold-damage", "axis-degrades-most", "regime-weakest-outcome"}:
            rows = self._query(f"SELECT * FROM ({sql_by_source[source]}) q ORDER BY {measure} ASC NULLS LAST LIMIT %(limit)s", params)["rows"]
        if preset_id == "ticker-selection-low-contribution":
            rows = [row for row in rows if (row.get("selection_rate") or 0) > 0 and (row.get("total_pnl_contribution") or 0) <= 0]
        rows = [row for row in rows if row.get(measure) is not None]
        return {"preset_id": preset_id, "question": question, "operation": operation, "source": source, "measure": measure, "dimension": dimension, "count": len(rows), "rows": rows}

    def execution_evidence(self):
        counts = self.row_counts()
        perf = self.query_performance()["rows"]
        source_usage = self._query(
            """
            SELECT source_relation, count(*) AS query_count, avg(elapsed_ms) AS avg_elapsed_ms, sum(rows_returned) AS rows_returned
            FROM oltp.dss_query_log
            GROUP BY source_relation
            ORDER BY query_count DESC NULLS LAST
            """
        )["rows"]
        try:
            active = self._query(
                """
                SELECT active_run_id, activated_at, status
                FROM oltp.active_dss_run
                WHERE singleton_key
                """
            )["rows"]
        except Exception:
            active = []
        try:
            invalidations = self._query(
                """
                SELECT run_id, endpoint_pattern, reason, invalidated_at
                FROM oltp.cache_invalidation_log
                ORDER BY invalidated_at DESC
                LIMIT 50
                """
            )["rows"]
        except Exception:
            invalidations = []
        return {
            "row_counts": counts,
            "query_performance": perf,
            "source_usage": source_usage,
            "active_run": active[0] if active else None,
            "cache_invalidation_plan": invalidations,
        }

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
        if fold is None:
            return self._query(
                """
                SELECT
                    candidate_id,
                    universe_id,
                    ticker,
                    sum(total_pnl_contribution) AS total_pnl_contribution,
                    avg(avg_final_weight) AS avg_final_weight,
                    avg(selection_rate) AS selection_rate,
                    avg(leader_flag_rate) AS leader_flag_rate,
                    min(worst_daily_contribution) AS worst_daily_contribution,
                    sum(observations) AS observations,
                    bool_or(demo_mode) AS demo_mode
                FROM mart.mv_ticker_contribution
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s
                GROUP BY candidate_id, universe_id, ticker
                ORDER BY total_pnl_contribution DESC NULLS LAST
                LIMIT %(limit)s
                """,
                {"candidate_id": candidate_id, "universe_id": universe_id, "limit": limit},
            )
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
        if fold is None:
            return self._query(
                """
                SELECT
                    regime,
                    candidate_id,
                    universe_id,
                    avg(avg_net_return) AS avg_net_return,
                    avg(avg_benchmark_return) AS avg_benchmark_return,
                    avg(avg_exposure) AS avg_exposure,
                    avg(avg_turnover) AS avg_turnover,
                    avg(avg_drawdown) AS avg_drawdown,
                    avg(backoff_activation_rate) AS backoff_activation_rate,
                    avg(continuation_activation_rate) AS continuation_activation_rate,
                    avg(avg_leader_blend) AS avg_leader_blend,
                    sum(observations) AS observations,
                    bool_or(demo_mode) AS demo_mode
                FROM mart.mv_regime_behavior
                WHERE candidate_id = %(candidate_id)s AND universe_id = %(universe_id)s
                GROUP BY regime, candidate_id, universe_id
                ORDER BY observations DESC
                """,
                {"candidate_id": candidate_id, "universe_id": universe_id},
            )
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
