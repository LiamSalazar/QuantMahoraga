from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import SCHEMA_VERSION
from .load_postgres import execute_sql_file
from .paths import DssPaths, get_paths


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def write_control_json(paths: DssPaths, name: str, payload: Any) -> Path:
    paths.control_root.mkdir(parents=True, exist_ok=True)
    target = paths.control_root / name
    target.write_text(json.dumps(_jsonable(payload), indent=2, default=str), encoding="utf-8")
    return target


def ensure_control_plane(database_url: str | None, paths: DssPaths | None = None) -> None:
    if not database_url:
        return
    paths = paths or get_paths()
    execute_sql_file(database_url, paths.sql_root / "011_create_control_plane.sql")


def _safe_execute(database_url: str | None, sql: str, params: dict[str, Any] | None = None) -> None:
    if not database_url:
        return
    try:
        import psycopg
        from psycopg.types.json import Jsonb

        converted = {}
        for key, value in (params or {}).items():
            converted[key] = Jsonb(value) if isinstance(value, dict | list) else value
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, converted)
            conn.commit()
    except Exception:
        return


def start_pipeline_run(
    database_url: str | None,
    *,
    run_id: str,
    strategy: str,
    profile: str,
    mode: str,
    changed_sources_count: int = 0,
    changed_partitions_count: int = 0,
) -> None:
    _safe_execute(
        database_url,
        """
        INSERT INTO oltp.pipeline_run
            (run_id, strategy, profile, mode, status, changed_sources_count, changed_partitions_count)
        VALUES
            (%(run_id)s, %(strategy)s, %(profile)s, %(mode)s, 'STARTED',
             %(changed_sources_count)s, %(changed_partitions_count)s)
        ON CONFLICT (run_id) DO UPDATE SET
            strategy = EXCLUDED.strategy,
            profile = EXCLUDED.profile,
            mode = EXCLUDED.mode,
            status = 'STARTED',
            changed_sources_count = EXCLUDED.changed_sources_count,
            changed_partitions_count = EXCLUDED.changed_partitions_count,
            error_message = NULL
        """,
        {
            "run_id": run_id,
            "strategy": strategy,
            "profile": profile,
            "mode": mode,
            "changed_sources_count": changed_sources_count,
            "changed_partitions_count": changed_partitions_count,
        },
    )


def finish_pipeline_run(
    database_url: str | None,
    *,
    run_id: str,
    status: str,
    total_rows_processed: int = 0,
    total_rows_loaded: int = 0,
    validation_status: str | None = None,
    published: bool = False,
    error_message: str | None = None,
) -> None:
    _safe_execute(
        database_url,
        """
        UPDATE oltp.pipeline_run
        SET status = %(status)s,
            finished_at = now(),
            total_rows_processed = %(total_rows_processed)s,
            total_rows_loaded = %(total_rows_loaded)s,
            validation_status = %(validation_status)s,
            published = %(published)s,
            error_message = %(error_message)s
        WHERE run_id = %(run_id)s
        """,
        {
            "run_id": run_id,
            "status": status,
            "total_rows_processed": total_rows_processed,
            "total_rows_loaded": total_rows_loaded,
            "validation_status": validation_status,
            "published": published,
            "error_message": error_message,
        },
    )


def log_stage(
    database_url: str | None,
    *,
    run_id: str,
    stage_name: str,
    status: str,
    started_at: datetime,
    rows_read: int | None = None,
    rows_written: int | None = None,
    output_bytes: int | None = None,
    error_message: str | None = None,
) -> None:
    finished_at = _utc_now()
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)
    _safe_execute(
        database_url,
        """
        INSERT INTO oltp.pipeline_stage_log
            (run_id, stage_name, status, started_at, finished_at, duration_ms,
             rows_read, rows_written, output_bytes, error_message)
        VALUES
            (%(run_id)s, %(stage_name)s, %(status)s, %(started_at)s, %(finished_at)s, %(duration_ms)s,
             %(rows_read)s, %(rows_written)s, %(output_bytes)s, %(error_message)s)
        """,
        {
            "run_id": run_id,
            "stage_name": stage_name,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "rows_read": rows_read,
            "rows_written": rows_written,
            "output_bytes": output_bytes,
            "error_message": error_message,
        },
    )


@contextmanager
def stage_timer(database_url: str | None, run_id: str, stage_name: str) -> Iterator[dict[str, int | None]]:
    metrics: dict[str, int | None] = {"rows_read": None, "rows_written": None, "output_bytes": None}
    started_at = _utc_now()
    try:
        yield metrics
    except Exception as exc:
        log_stage(database_url, run_id=run_id, stage_name=stage_name, status="FAILED", started_at=started_at, error_message=str(exc))
        raise
    else:
        log_stage(
            database_url,
            run_id=run_id,
            stage_name=stage_name,
            status="COMPLETED",
            started_at=started_at,
            rows_read=metrics.get("rows_read"),
            rows_written=metrics.get("rows_written"),
            output_bytes=metrics.get("output_bytes"),
        )


def upsert_source_manifest(database_url: str | None, entries: list[Any], run_id: str) -> None:
    if not database_url or not entries:
        return
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for entry in entries:
                    row = asdict(entry) if is_dataclass(entry) else dict(entry)
                    cur.execute(
                        """
                        INSERT INTO oltp.source_manifest
                            (source_name, source_path, source_type, source_hash, row_count, min_date, max_date,
                             candidate_id, universe_id, fold, horizon, module_name, first_seen_at, last_seen_at,
                             last_loaded_run_id)
                        VALUES
                            (%(source_name)s, %(source_path)s, %(source_type)s, %(source_hash)s, %(row_count)s,
                             %(min_date)s, %(max_date)s, %(candidate_id)s, %(universe_id)s, %(fold)s, %(horizon)s,
                             %(module_name)s, now(), now(), %(run_id)s)
                        ON CONFLICT (source_name, source_path) DO UPDATE SET
                            source_type = EXCLUDED.source_type,
                            source_hash = EXCLUDED.source_hash,
                            row_count = EXCLUDED.row_count,
                            min_date = EXCLUDED.min_date,
                            max_date = EXCLUDED.max_date,
                            candidate_id = EXCLUDED.candidate_id,
                            universe_id = EXCLUDED.universe_id,
                            fold = EXCLUDED.fold,
                            horizon = EXCLUDED.horizon,
                            module_name = EXCLUDED.module_name,
                            last_seen_at = now(),
                            last_loaded_run_id = EXCLUDED.last_loaded_run_id
                        """,
                        {**row, "run_id": run_id},
                    )
            conn.commit()
    except Exception:
        return


def log_data_quality(database_url: str | None, run_id: str, results: list[Any]) -> None:
    if not database_url or not results:
        return
    try:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for result in results:
                    payload = asdict(result) if is_dataclass(result) else dict(result)
                    for check in payload.get("checks", []):
                        cur.execute(
                            """
                            INSERT INTO oltp.data_quality_check
                                (run_id, table_name, check_name, status, severity,
                                 observed_value, expected_value, details)
                            VALUES
                                (%(run_id)s, %(table_name)s, %(check_name)s, %(status)s, %(severity)s,
                                 %(observed_value)s, %(expected_value)s, %(details)s)
                            """,
                            {
                                "run_id": run_id,
                                "table_name": payload.get("table_name"),
                                "check_name": check.get("check_name"),
                                "status": check.get("status"),
                                "severity": check.get("severity"),
                                "observed_value": str(check.get("observed_value")),
                                "expected_value": str(check.get("expected_value")),
                                "details": Jsonb(check.get("details") or {}),
                            },
                        )
            conn.commit()
    except Exception:
        return


def publish_run(database_url: str | None, run_id: str, status: str = "published") -> None:
    _safe_execute(
        database_url,
        """
        WITH previous AS (
            SELECT new_active_run_id
            FROM oltp.publish_log
            WHERE status = 'published'
            ORDER BY published_at DESC
            LIMIT 1
        )
        INSERT INTO oltp.publish_log
            (run_id, previous_active_run_id, new_active_run_id, status, rollback_available)
        SELECT
            %(run_id)s,
            (SELECT new_active_run_id FROM previous),
            %(run_id)s,
            %(status)s,
            (SELECT new_active_run_id FROM previous) IS NOT NULL
        """,
        {"run_id": run_id, "status": status},
    )


def log_cache_invalidations(database_url: str | None, run_id: str, endpoints: list[str], reason: str) -> None:
    if not database_url or not endpoints:
        return
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for endpoint in sorted(set(endpoints)):
                    cur.execute(
                        """
                        INSERT INTO oltp.cache_invalidation_log (run_id, endpoint_pattern, reason)
                        VALUES (%(run_id)s, %(endpoint)s, %(reason)s)
                        """,
                        {"run_id": run_id, "endpoint": endpoint, "reason": reason},
                    )
            conn.commit()
    except Exception:
        return


def log_partition_manifest(database_url: str | None, run_id: str, rows: list[dict[str, Any]]) -> None:
    if not database_url or not rows:
        return
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO oltp.partition_manifest
                            (run_id, table_name, partition_key, partition_type, row_count, min_date, max_date, source_hash, status)
                        VALUES
                            (%(run_id)s, %(table_name)s, %(partition_key)s, %(partition_type)s, %(row_count)s,
                             %(min_date)s, %(max_date)s, %(source_hash)s, %(status)s)
                        """,
                        {**row, "run_id": run_id},
                    )
            conn.commit()
    except Exception:
        return


def schema_metadata() -> dict[str, str]:
    return {"schema_version": SCHEMA_VERSION}
