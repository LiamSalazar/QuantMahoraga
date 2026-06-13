from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05"
OFFICIAL_UNIVERSE_ID = "base_universe_12"
BASELINE_REFERENCE = "Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05"
PHASE = "mahoraga14_3_dss_postgres"

MODULE_ORDER = {
    "BASE_ALPHA_V2": ("base alpha", 10),
    "continuation_v2_model": ("continuation", 20),
    "structural_defense_model": ("structural defense", 30),
    "participation_allocator_v2": ("participation allocator", 40),
    "conviction_amplifier_layer": ("conviction amplifier", 50),
    "risk_backoff_layer_v2": ("risk backoff", 60),
    "leader_participation_layer": ("leader participation", 70),
}

METRICS = [
    ("CAGR", "performance", "CAGR", True, "pct"),
    ("Sharpe", "risk_adjusted", "Sharpe", True, "ratio"),
    ("Sortino", "risk_adjusted", "Sortino", True, "ratio"),
    ("MaxDD", "risk", "Max drawdown", False, "pct"),
    ("AlphaNW_QQQ", "alpha", "Alpha vs QQQ", True, "pct"),
    ("AlphaNW_SPY", "alpha", "Alpha vs SPY", True, "pct"),
    ("AvgExposure", "exposure", "Average exposure", True, "ratio"),
    ("AvgTurnover", "trading", "Average turnover", False, "ratio"),
    ("robust_score", "robustness", "Robust score", True, "score"),
]

PROFILE_ROW_TARGETS = {
    "small": {"demo_grid_points": 360, "expected_real_min_rows": 1},
    "standard": {"demo_grid_points": 2_500, "expected_real_min_rows": 4_000_000},
    "competition": {"demo_grid_points": 12_000, "expected_real_min_rows": 40_000_000},
}


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str = "small"
    mode: str = "parquet"
    database_url: str | None = None
    include_demo_grid: bool = True
    run_id: str = ""

    @property
    def demo_mode(self) -> bool:
        return self.mode in {"demo", "parquet"} and self.include_demo_grid

    @property
    def row_target(self) -> dict[str, int]:
        return PROFILE_ROW_TARGETS[self.profile]


def utc_run_id(prefix: str = "dss") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_config(
    profile: str = "small",
    mode: str = "parquet",
    database_url: str | None = None,
    include_demo_grid: bool = True,
    run_id: str | None = None,
) -> RuntimeConfig:
    if profile not in PROFILE_ROW_TARGETS:
        raise ValueError(f"Unknown profile {profile!r}. Expected one of: {', '.join(PROFILE_ROW_TARGETS)}")
    if mode not in {"parquet", "postgres", "demo"}:
        raise ValueError("mode must be parquet, postgres, or demo")
    return RuntimeConfig(
        profile=profile,
        mode=mode,
        database_url=database_url or os.getenv("DATABASE_URL"),
        include_demo_grid=include_demo_grid,
        run_id=run_id or utc_run_id(),
    )


def official_knobs() -> dict[str, float]:
    return {
        "budget_multiplier": 1.05,
        "conviction_multiplier": 1.10,
        "leader_multiplier": 1.10,
        "backoff_strength": 1.05,
    }


def profile_names() -> Iterable[str]:
    return PROFILE_ROW_TARGETS.keys()
