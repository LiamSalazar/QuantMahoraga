from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from fast_report import build_fast_report_text, save_fast_outputs
from full_report import build_full_report_text, save_full_outputs
from mahoraga11_config import Mahoraga11Config


def build_fold_summary(wf: Dict[str, Any]) -> pd.DataFrame:
    return wf["fold_df"].copy().sort_values("fold")


def build_final_report_text(wf: Dict[str, Any], cfg: Mahoraga11Config) -> str:
    if cfg.run_mode.upper() == "FULL":
        return build_full_report_text(wf, cfg)
    return build_fast_report_text(wf, cfg)


def save_outputs(wf: Dict[str, Any], cfg: Mahoraga11Config, ff=None) -> Dict[str, pd.DataFrame]:
    if cfg.run_mode.upper() == "FULL":
        return save_full_outputs(wf, cfg, ff=ff)
    return save_fast_outputs(wf, cfg)
