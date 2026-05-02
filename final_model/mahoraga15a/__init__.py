from __future__ import annotations

import importlib.util
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BACKTEST_SPEC = importlib.util.spec_from_file_location("mahoraga15a_local_backtest_executor", _THIS_DIR / "backtest_executor.py")
if _BACKTEST_SPEC is None or _BACKTEST_SPEC.loader is None:
    raise ImportError("Unable to load Mahoraga15A backtest executor.")
_BACKTEST_MODULE = importlib.util.module_from_spec(_BACKTEST_SPEC)
_BACKTEST_SPEC.loader.exec_module(_BACKTEST_MODULE)

rebuild_ls_fold = _BACKTEST_MODULE.rebuild_ls_fold
run_walk_forward_mahoraga15a = _BACKTEST_MODULE.run_walk_forward_mahoraga15a

run_walk_forward_mahoraga15a1 = run_walk_forward_mahoraga15a
run_walk_forward_mahoraga15a2 = run_walk_forward_mahoraga15a
run_walk_forward_mahoraga15a3 = run_walk_forward_mahoraga15a

__all__ = ["run_walk_forward_mahoraga15a", "run_walk_forward_mahoraga15a1", "run_walk_forward_mahoraga15a2", "run_walk_forward_mahoraga15a3", "rebuild_ls_fold"]
