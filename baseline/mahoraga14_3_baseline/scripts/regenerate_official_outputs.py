from __future__ import annotations

from _bootstrap import bootstrap_paths

bootstrap_paths()

from mahoraga14_3_baseline.official_baseline_runner import run_official_baseline


if __name__ == "__main__":
    run_official_baseline(make_plots_flag=True, run_mode="FAST")
