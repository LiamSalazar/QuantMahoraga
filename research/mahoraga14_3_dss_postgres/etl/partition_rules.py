from __future__ import annotations

PARTITION_SPECS: dict[str, list[str]] = {
    "fact_position_daily": ["year", "fold", "candidate_id", "universe_id"],
    "fact_signal_daily": ["year", "fold", "candidate_id", "universe_id"],
    "fact_market_bar": ["year"],
    "fact_outcome": ["horizon", "fold", "candidate_id", "universe_id"],
    "fact_module_trace": ["module_name", "fold", "candidate_id", "universe_id"],
    "fact_whatif": ["scenario_id", "fold", "horizon", "demo_mode"],
    "fact_path_recursive": ["year", "candidate_id", "fold"],
}

TABLE_DATE_COLUMN = {
    "fact_position_daily": "date_value",
    "fact_signal_daily": "date_value",
    "fact_market_bar": "date_value",
    "fact_path_recursive": "date_value",
    "fact_outcome": "decision_date",
    "fact_module_trace": "date_value",
}

SOURCE_PARTITION_COLUMNS: dict[str, dict[str, str]] = {
    "fact_position_daily": {
        "year": "date",
        "fold": "fold",
        "candidate_id": "candidate_id",
        "universe_id": "universe_id",
    },
    "fact_signal_daily": {
        "year": "date",
        "fold": "fold",
        "candidate_id": "candidate_id",
        "universe_id": "universe_id",
    },
    "fact_market_bar": {"year": "date"},
    "fact_outcome": {
        "horizon": "horizon",
        "fold": "fold",
        "candidate_id": "candidate_id",
        "universe_id": "universe_id",
    },
    "fact_module_trace": {
        "module_name": "module_name",
        "fold": "fold",
        "candidate_id": "candidate_id",
        "universe_id": "universe_id",
    },
    "fact_path_recursive": {
        "year": "date",
        "candidate_id": "candidate_id",
        "fold": "fold",
    },
}


def partition_label(values: dict[str, object]) -> str:
    if values.get("ALL"):
        return "ALL"
    return "/".join(f"{key}={values[key]}" for key in values)


def parse_partition_label(table_name: str, label: str) -> dict[str, object]:
    if label == "ALL":
        return {"ALL": True}
    parsed: dict[str, object] = {}
    for part in label.split("/"):
        key, raw = part.split("=", 1)
        if raw == "__null__":
            parsed[key] = None
        elif key in {"year", "fold", "horizon"}:
            parsed[key] = int(raw)
        elif key == "demo_mode":
            parsed[key] = raw.lower() in {"true", "1", "t"}
        else:
            parsed[key] = raw
    return {key: parsed[key] for key in PARTITION_SPECS.get(table_name, parsed.keys()) if key in parsed}
