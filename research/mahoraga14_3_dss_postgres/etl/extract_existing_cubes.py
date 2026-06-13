from __future__ import annotations

from pathlib import Path

import polars as pl

from .paths import DssPaths, get_paths
from .date_utils import parse_date


PARQUET_CUBES = {
    "decision": "outputs/audit_cube/decision_date_cube.parquet",
    "position": "outputs/audit_cube/position_cube.parquet",
    "module_trace": "outputs/audit_cube/module_trace_cube.parquet",
    "outcome": "outputs/audit_cube/outcome_cube.parquet",
    "market_context": "outputs/audit_cube/market_context_cube.parquet",
}

CSV_TABLES = {
    "extended_summary": "outputs/extended_multiplier_robustness/extended_multiplier_summary.csv",
    "extended_fold_summary": "outputs/extended_multiplier_robustness/extended_multiplier_fold_summary.csv",
    "universe_summary": "outputs/universe_robustness/universe_robustness_summary.csv",
    "baseline_fold_summary": "baseline/mahoraga14_3_baseline/outputs/fold_summary_official.csv",
    "active_return": "baseline/mahoraga14_3_baseline/outputs/active_return_vs_qqq_official.csv",
    "allocator_cash_drag": "baseline/mahoraga14_3_baseline/audit/allocator_cash_drag_official.csv",
    "cost_sensitivity": "baseline/mahoraga14_3_baseline/outputs/cost_sensitivity_official.csv",
    "slippage_sensitivity": "baseline/mahoraga14_3_baseline/outputs/slippage_sensitivity_official.csv",
}


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=1000, ignore_errors=True) if path.exists() else pl.DataFrame()


def load_sources(paths: DssPaths | None = None) -> dict[str, pl.DataFrame]:
    paths = paths or get_paths()
    sources: dict[str, pl.DataFrame] = {}
    for name, rel in PARQUET_CUBES.items():
        sources[name] = _read_parquet(paths.extended_root / rel)
    for name, rel in CSV_TABLES.items():
        root = paths.extended_root if rel.startswith("outputs/") else paths.repo_root
        sources[name] = _read_csv((root / rel).resolve())
    return sources


def normalize_date_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    exprs = []
    for column in columns:
        if column in df.columns:
            exprs.append(parse_date(column).alias(column))
    return df.with_columns(exprs) if exprs else df
