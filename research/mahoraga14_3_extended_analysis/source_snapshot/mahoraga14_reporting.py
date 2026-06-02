from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from mahoraga14_config import Mahoraga14Config


def build_fold_summary(wf: Dict[str, Any]) -> pd.DataFrame:
    return wf["fold_df"].copy().sort_values("fold")


def build_final_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    if cfg.run_mode.upper() == "FULL":
        from full_report import build_full_report_text

        return build_full_report_text(wf, cfg)
    from fast_fail_diagnostics_14_3 import build_fast_report_text

    return build_fast_report_text(wf, cfg)


def save_outputs(wf: Dict[str, Any], cfg: Mahoraga14Config, ff=None, costs=None) -> Dict[str, pd.DataFrame]:
    if cfg.run_mode.upper() == "FULL":
        from full_report import save_full_outputs

        return save_full_outputs(wf, cfg, ff=ff, costs=costs)
    from fast_fail_diagnostics_14_3 import save_fast_outputs

    return save_fast_outputs(wf, cfg, costs=costs)

