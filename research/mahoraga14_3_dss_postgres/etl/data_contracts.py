from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .control_plane import log_data_quality
from .paths import DssPaths, ensure_output_dirs, get_paths


@dataclass(frozen=True)
class ContractResult:
    table_name: str
    passed: bool
    error_count: int
    warning_count: int
    checks: list[dict[str, Any]]


CONTRACTS: dict[str, dict[str, Any]] = {
    "fact_position_daily": {
        "grain": ["date_value", "candidate_id", "fold", "universe_id", "ticker", "run_id"],
        "required": ["date_value", "candidate_id", "fold", "universe_id", "ticker", "run_id", "demo_mode"],
        "allowed": {"fold": [1, 2, 3, 4, 5]},
        "no_null": ["date_value", "candidate_id", "fold", "universe_id", "ticker", "run_id"],
        "simulation": "demo_mode_required",
    },
    "fact_outcome": {
        "grain": ["decision_date", "candidate_id", "universe_id", "fold", "horizon", "run_id"],
        "required": ["decision_date", "candidate_id", "universe_id", "fold", "horizon", "realized_return", "run_id", "demo_mode"],
        "allowed": {"horizon": [1, 5, 20, 60], "fold": [1, 2, 3, 4, 5]},
        "no_null": ["decision_date", "candidate_id", "universe_id", "fold", "horizon", "run_id"],
        "simulation": "demo_mode_required",
    },
    "fact_module_trace": {
        "grain": ["date_value", "candidate_id", "universe_id", "fold", "module_name", "run_id"],
        "required": ["date_value", "candidate_id", "universe_id", "fold", "module_name", "run_id", "demo_mode"],
        "allowed": {"fold": [1, 2, 3, 4, 5]},
        "no_null": ["date_value", "candidate_id", "universe_id", "fold", "module_name", "run_id"],
        "simulation": "demo_mode_required",
    },
    "fact_decision_state": {
        "grain": ["date_value", "candidate_id", "universe_id", "fold", "run_id"],
        "required": ["date_value", "candidate_id", "universe_id", "fold", "run_id", "demo_mode"],
        "allowed": {"fold": [1, 2, 3, 4, 5]},
        "no_null": ["date_value", "candidate_id", "universe_id", "fold", "run_id"],
        "simulation": "demo_mode_required",
    },
    "fact_whatif": {
        "grain": ["scenario_id", "candidate_id", "universe_id", "fold", "horizon", "cost_bps", "slippage_bps", "run_id"],
        "required": ["scenario_id", "candidate_id", "universe_id", "horizon", "source_artifact", "run_id", "demo_mode"],
        "allowed": {"horizon": [1, 5, 20, 60]},
        "no_null": ["scenario_id", "candidate_id", "universe_id", "horizon", "source_artifact", "run_id"],
        "simulation": "synthetic_source_must_be_demo",
    },
    "dim_candidate": {
        "grain": ["candidate_id"],
        "required": ["candidate_id", "candidate_label", "family", "demo_mode"],
        "no_null": ["candidate_id", "candidate_label"],
    },
    "dim_asset": {
        "grain": ["ticker"],
        "required": ["ticker", "asset_class", "demo_mode"],
        "no_null": ["ticker"],
    },
    "dim_date": {
        "grain": ["date_value"],
        "required": ["date_key", "date_value", "year", "quarter", "month"],
        "no_null": ["date_key", "date_value"],
    },
}


def _check(status: bool, name: str, severity: str, observed: Any, expected: Any, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "PASS" if status else "FAIL",
        "severity": severity.upper(),
        "observed_value": observed,
        "expected_value": expected,
        "details": details or {},
    }


def _duplicate_count(df: pl.DataFrame, key: list[str]) -> int:
    if not set(key).issubset(df.columns):
        return -1
    return int(df.group_by(key).len().filter(pl.col("len") > 1).height)


def validate_table_contract(name: str, df: pl.DataFrame) -> ContractResult:
    contract = CONTRACTS.get(name)
    if contract is None:
        return ContractResult(name, True, 0, 0, [_check(True, "contract_defined", "INFO", "not_configured", "optional")])

    checks: list[dict[str, Any]] = []
    required = contract.get("required", [])
    missing = [column for column in required if column not in df.columns]
    checks.append(_check(not missing, "required_columns_present", "ERROR", missing, "[]", {"required": required}))
    if missing:
        return _result(name, checks)

    for column in contract.get("no_null", []):
        nulls = int(df.select(pl.col(column).is_null().sum()).item()) if column in df.columns else -1
        checks.append(_check(nulls == 0, f"{column}_not_null", "ERROR", nulls, 0))

    for column, allowed in contract.get("allowed", {}).items():
        if column in df.columns:
            invalid = int(df.filter(pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed)).height)
            checks.append(_check(invalid == 0, f"{column}_allowed_values", "ERROR", invalid, 0, {"allowed": allowed}))

    grain = contract.get("grain", [])
    if grain:
        duplicates = _duplicate_count(df, grain)
        checks.append(_check(duplicates == 0, "duplicate_grain_check", "ERROR", duplicates, 0, {"grain": grain}))

    if contract.get("simulation") == "demo_mode_required" and "demo_mode" in df.columns:
        null_demo = int(df.select(pl.col("demo_mode").is_null().sum()).item())
        checks.append(_check(null_demo == 0, "demo_mode_not_null", "ERROR", null_demo, 0))
    if contract.get("simulation") == "synthetic_source_must_be_demo" and {"source_artifact", "demo_mode"}.issubset(df.columns):
        synthetic_unflagged = int(df.filter((pl.col("source_artifact") == "demo_synthetic_whatif_grid") & (~pl.col("demo_mode"))).height)
        checks.append(_check(synthetic_unflagged == 0, "synthetic_rows_are_flagged", "ERROR", synthetic_unflagged, 0))

    return _result(name, checks)


def _result(name: str, checks: list[dict[str, Any]]) -> ContractResult:
    error_count = sum(1 for check in checks if check["severity"] == "ERROR" and check["status"] != "PASS")
    warning_count = sum(1 for check in checks if check["severity"] == "WARNING" and check["status"] != "PASS")
    return ContractResult(name, error_count == 0, error_count, warning_count, checks)


def validate_all_contracts(tables: dict[str, pl.DataFrame]) -> list[ContractResult]:
    return [validate_table_contract(name, tables[name]) for name in CONTRACTS if name in tables]


def write_contract_report(results: list[ContractResult], run_id: str, paths: DssPaths | None = None) -> Path:
    paths = ensure_output_dirs(paths or get_paths())
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(result.passed for result in results),
        "results": [asdict(result) for result in results],
    }
    target = paths.control_root / f"data_quality_report_{run_id}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = paths.control_root / f"data_quality_report_{run_id}.md"
    lines = [f"# Data Quality Report {run_id}", "", f"Passed: `{payload['passed']}`", ""]
    for result in results:
        lines.append(f"## {result.table_name}")
        lines.append(f"- passed: `{result.passed}`")
        lines.append(f"- errors: `{result.error_count}`")
        lines.append(f"- warnings: `{result.warning_count}`")
        for check in result.checks:
            lines.append(f"- {check['severity']} {check['check_name']}: {check['status']} ({check['observed_value']} vs {check['expected_value']})")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return target


def persist_contract_results(database_url: str | None, run_id: str, results: list[ContractResult], *, strict: bool = False) -> None:
    log_data_quality(database_url, run_id, results, strict=strict)
