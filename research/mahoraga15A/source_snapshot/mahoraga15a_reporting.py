from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from fast_report import save_fast_outputs
from mahoraga15a_config import Mahoraga15AConfig


def save_outputs(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> Dict[str, pd.DataFrame]:
    return save_fast_outputs(wf, cfg, costs)
