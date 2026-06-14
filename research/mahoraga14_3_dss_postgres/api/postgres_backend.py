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

    def decision_replay(self, candidate_id, fold, universe_id, date_value, ticker):
        params = {"candidate_id": candidate_id, "fold": fold, "universe_id": universe_id, "date_value": date_value, "ticker": ticker}
        decision_rows = self._query(
            """
            WITH candidates AS (
                SELECT
                    d.*,
                    (SELECT count(*) FROM dw.fact_position_daily p
                     WHERE p.candidate_id = d.candidate_id AND p.fold = d.fold AND p.universe_id = d.universe_id AND p.date_value = d.date_value) AS position_count,
                    (SELECT count(*) FROM dw.fact_module_trace m
                     WHERE m.candidate_id = d.candidate_id AND m.fold = d.fold AND m.universe_id = d.universe_id AND m.date_value = d.date_value) AS module_count,
                    (SELECT count(*) FROM dw.fact_outcome o
                     WHERE o.candidate_id = d.candidate_id AND o.fold = d.fold AND o.universe_id = d.universe_id AND o.decision_date = d.date_value) AS outcome_count
                FROM mart.mv_decision_replay d
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
            "candidate-cagr-maxdd": ("Which candidate has the best CAGR/MaxDD tradeoff?", "pivot", "mart.mv_scorecard_candidate", "return_per_exposure", "candidate_id"),
            "candidate-best-sharpe": ("Which candidate has the best Sharpe?", "roll-up", "mart.mv_scorecard_candidate", "sharpe", "candidate_id"),
            "candidate-severe-fold-damage": ("Which candidate has severe fold damage?", "dice", "mart.mv_scorecard_candidate", "maxdd", "candidate_id"),
            "axis-degrades-most": ("Which multiplier axis degrades the model most?", "roll-up", "mart.mv_robustness_surface", "metric_value", "sweep_role"),
            "module-helps-horizon": ("Which module helps most by horizon?", "pivot", "mart.mv_module_effectiveness", "helped_rate", "module_name"),
            "module-active-low-value": ("Which module activates often but adds little?", "dice", "mart.mv_module_effectiveness", "activation_rate", "module_name"),
            "ticker-top-contribution": ("Which tickers contribute most?", "drill-down", "mart.mv_ticker_contribution", "total_pnl_contribution", "ticker"),
            "ticker-largest-drags": ("Which tickers drag most?", "drill-down", "mart.mv_ticker_contribution", "total_pnl_contribution", "ticker"),
            "ticker-frequent-leaders": ("Which tickers are frequent leaders?", "roll-up", "mart.mv_ticker_contribution", "leader_flag_rate", "ticker"),
            "regime-best-alpha": ("Which regime has the best alpha proxy?", "slice", "mart.mv_regime_behavior", "avg_net_return", "regime"),
            "regime-exposure-concentration": ("Where is exposure concentrated?", "slice", "mart.mv_regime_behavior", "avg_exposure", "regime"),
            "regime-backoff-most": ("Where does backoff activate most?", "slice", "mart.mv_regime_behavior", "backoff_activation_rate", "regime"),
            "decision-best-20d": ("Best decisions by 20d outcome.", "drill-through", "mart.mv_decision_outcome", "realized_return", "date_value"),
            "decision-worst-20d": ("Worst decisions by 20d outcome.", "drill-through", "mart.mv_decision_outcome", "realized_return", "date_value"),
            "engineering-slowest-endpoint": ("Which endpoint is slowest?", "roll-up", "oltp.dss_query_log", "avg_elapsed_ms", "endpoint"),
            "engineering-source-most-used": ("Which source relation is used most?", "roll-up", "oltp.dss_query_log", "query_count", "source_relation"),
        }
        question, operation, source, measure, dimension = presets.get(preset_id, presets["fold-best-performance"])
        params = {"candidate_id": candidate_id, "universe_id": universe_id, "fold": fold, "limit": limit}
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
                SELECT ticker, total_pnl_contribution, avg_final_weight, selection_rate, leader_flag_rate, observations
                FROM mart.mv_ticker_contribution
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
            """,
            "mart.mv_regime_behavior": """
                SELECT regime, avg_net_return, avg_benchmark_return, avg_exposure, avg_drawdown, backoff_activation_rate,
                       continuation_activation_rate, avg_leader_blend, observations
                FROM mart.mv_regime_behavior
                WHERE candidate_id=%(candidate_id)s AND universe_id=%(universe_id)s AND (%(fold)s::int IS NULL OR fold=%(fold)s::int)
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
            """,
            "oltp.dss_query_log": """
                SELECT endpoint, source_relation, count(*) AS query_count, avg(elapsed_ms) AS avg_elapsed_ms,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95_elapsed_ms,
                       avg(rows_returned) AS avg_rows_returned
                FROM oltp.dss_query_log
                GROUP BY endpoint, source_relation
            """,
        }
        rows = self._query(f"SELECT * FROM ({sql_by_source[source]}) q ORDER BY {measure} DESC NULLS LAST LIMIT %(limit)s", params)["rows"]
        if preset_id in {"ticker-largest-drags", "decision-worst-20d", "fold-worst-drawdown", "candidate-severe-fold-damage", "axis-degrades-most"}:
            rows = self._query(f"SELECT * FROM ({sql_by_source[source]}) q ORDER BY {measure} ASC NULLS LAST LIMIT %(limit)s", params)["rows"]
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
        return {"row_counts": counts, "query_performance": perf, "source_usage": source_usage}

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
