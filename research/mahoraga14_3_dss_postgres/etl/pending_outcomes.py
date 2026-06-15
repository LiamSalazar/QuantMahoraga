from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl


def maturity_date_from_available_dates(decision_date: date, horizon: int, available_dates: list[date]) -> date | None:
    ordered = [item for item in available_dates if item >= decision_date]
    if not ordered:
        return None
    idx = min(len(ordered) - 1, horizon)
    return ordered[idx]


def detect_pending_outcomes(decisions: pl.DataFrame, outcomes: pl.DataFrame, run_id: str) -> pl.DataFrame:
    if decisions.is_empty() or "date_value" not in decisions.columns:
        return pl.DataFrame()
    available_dates = sorted(decisions.get_column("date_value").drop_nulls().unique().to_list())
    horizons = [1, 5, 20, 60]
    base = decisions.select("candidate_id", "universe_id", "fold", pl.col("date_value").alias("decision_date")).unique()
    expected = base.join(pl.DataFrame({"horizon": horizons}), how="cross")
    existing = pl.DataFrame()
    if not outcomes.is_empty():
        existing = outcomes.select("candidate_id", "universe_id", "fold", "decision_date", "horizon").unique()
    rows: list[dict[str, Any]] = []
    existing_keys = {
        (row["candidate_id"], row["universe_id"], row["fold"], row["decision_date"], row["horizon"])
        for row in existing.to_dicts()
    } if not existing.is_empty() else set()
    max_date = available_dates[-1] if available_dates else None
    for row in expected.to_dicts():
        key = (row["candidate_id"], row["universe_id"], row["fold"], row["decision_date"], row["horizon"])
        maturity = maturity_date_from_available_dates(row["decision_date"], int(row["horizon"]), available_dates)
        if key in existing_keys:
            status = "computed"
        elif maturity and max_date and maturity <= max_date:
            status = "ready"
        else:
            status = "pending"
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "universe_id": row["universe_id"],
                "fold": row["fold"],
                "decision_date": row["decision_date"],
                "horizon": row["horizon"],
                "maturity_date": maturity,
                "status": status,
                "computed_at": None,
                "run_id": run_id,
            }
        )
    return pl.DataFrame(rows)


def persist_pending_outcomes(database_url: str | None, pending: pl.DataFrame) -> None:
    if not database_url or pending.is_empty():
        return
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for row in pending.to_dicts():
                    cur.execute(
                        """
                        INSERT INTO oltp.pending_outcome
                            (candidate_id, universe_id, fold, decision_date, horizon, maturity_date,
                             status, computed_at, run_id)
                        VALUES
                            (%(candidate_id)s, %(universe_id)s, %(fold)s, %(decision_date)s, %(horizon)s,
                             %(maturity_date)s, %(status)s, %(computed_at)s, %(run_id)s)
                        ON CONFLICT (candidate_id, universe_id, fold, decision_date, horizon) DO UPDATE SET
                            maturity_date = EXCLUDED.maturity_date,
                            status = EXCLUDED.status,
                            computed_at = EXCLUDED.computed_at,
                            run_id = EXCLUDED.run_id
                        """,
                        row,
                    )
            conn.commit()
    except Exception:
        return
