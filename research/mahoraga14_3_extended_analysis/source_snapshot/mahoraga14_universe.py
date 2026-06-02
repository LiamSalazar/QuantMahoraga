from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

import mahoraga6_1 as m6


def union_universe(ohlcv: dict, universe_schedule: Optional[pd.DataFrame], static_universe: List[str]) -> List[str]:
    if universe_schedule is not None and len(universe_schedule) > 0 and "members" in universe_schedule.columns:
        members = set()
        for x in universe_schedule["members"]:
            try:
                members |= set(json.loads(x))
            except Exception:
                continue
        return sorted([t for t in members if t in ohlcv["close"].columns])
    return sorted([t for t in static_universe if t in ohlcv["close"].columns])


def members_at_date(universe_schedule: Optional[pd.DataFrame], dt: pd.Timestamp, fallback: List[str]) -> List[str]:
    if universe_schedule is None or len(universe_schedule) == 0:
        return fallback
    try:
        members = m6.get_universe_at_date(universe_schedule, dt)
        return [t for t in members if t in fallback]
    except Exception:
        return fallback

