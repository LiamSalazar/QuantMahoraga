from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE = "mahoraga14_3_extended_analysis"
BASELINE_REFERENCE = "Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05"
OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05"
OFFICIAL_KNOBS = {
    "budget_multiplier": 1.05,
    "conviction_multiplier": 1.10,
    "leader_multiplier": 1.10,
    "backoff_strength": 1.05,
}
OFFICIAL_LABEL = "MAHORAGA14_3_BASELINE_OFFICIAL"
CONTROL_LABEL = "MAHORAGA14_1_LONG_ONLY_CONTROL"

ROBUST_SHARPE_DROP = 0.10
ROBUST_CAGR_DROP = 0.10
ROBUST_MAXDD_WORSENING_PP = 5.0

MULTIPLIER_RANGES = {
    "budget_multiplier": [0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
    "conviction_multiplier": [0.90, 1.00, 1.10, 1.20, 1.30],
    "leader_multiplier": [0.90, 1.00, 1.10, 1.20, 1.30],
    "backoff_strength": [0.90, 1.00, 1.05, 1.10, 1.20],
}

EXTREME_CASES = {
    "pro-risk": {
        "budget_multiplier": 1.15,
        "conviction_multiplier": 1.30,
        "leader_multiplier": 1.30,
        "backoff_strength": 0.90,
    },
    "pro-defense": {
        "budget_multiplier": 0.90,
        "conviction_multiplier": 0.90,
        "leader_multiplier": 0.90,
        "backoff_strength": 1.20,
    },
    "all-high": {
        "budget_multiplier": 1.15,
        "conviction_multiplier": 1.30,
        "leader_multiplier": 1.30,
        "backoff_strength": 1.20,
    },
    "all-low": {
        "budget_multiplier": 0.90,
        "conviction_multiplier": 0.90,
        "leader_multiplier": 0.90,
        "backoff_strength": 0.90,
    },
}

UNIVERSES = {
    "base_universe_12": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ASML", "TSM", "ADBE", "NFLX", "AMD"],
    "tech_20": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "AVGO",
        "ASML",
        "TSM",
        "ADBE",
        "NFLX",
        "AMD",
        "CRM",
        "ORCL",
        "NOW",
        "INTU",
        "CSCO",
        "IBM",
        "QCOM",
        "TXN",
    ],
    "tech_plus_semis": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "AVGO",
        "ASML",
        "TSM",
        "ADBE",
        "NFLX",
        "AMD",
        "INTC",
        "QCOM",
        "MU",
        "AMAT",
        "LRCX",
        "KLAC",
        "TXN",
        "MCHP",
        "MRVL",
        "ON",
        "ADI",
    ],
    "wider_largecap_growth": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "AVGO",
        "ASML",
        "TSM",
        "ADBE",
        "NFLX",
        "AMD",
        "CRM",
        "ORCL",
        "NOW",
        "INTU",
        "SHOP",
        "COST",
        "UNH",
        "MA",
        "V",
        "LLY",
        "BKNG",
        "ISRG",
    ],
    "negative_control_nontech": [
        "XOM",
        "CVX",
        "COP",
        "JPM",
        "BAC",
        "WMT",
        "PG",
        "KO",
        "PEP",
        "MCD",
        "HD",
        "NKE",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
    ],
}

UNIVERSE_STRESS_CANDIDATES = {
    OFFICIAL_CANDIDATE_ID: OFFICIAL_KNOBS,
    "EXTREME_pro-risk": EXTREME_CASES["pro-risk"],
    "EXTREME_pro-defense": EXTREME_CASES["pro-defense"],
}

MODULE_NAMES = [
    "BASE_ALPHA_V2",
    "continuation_v2_model",
    "structural_defense_model",
    "participation_allocator_v2",
    "conviction_amplifier_layer",
    "risk_backoff_layer_v2",
    "leader_participation_layer",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def phase_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    outputs: Path
    multiplier: Path
    universe: Path
    cube: Path
    figures: Path
    reports: Path
    manifests: Path
    cache: Path
    source_snapshot: Path
    configs: Path


def get_paths() -> Paths:
    root = phase_root()
    outputs = root / "outputs"
    return Paths(
        root=root,
        outputs=outputs,
        multiplier=outputs / "extended_multiplier_robustness",
        universe=outputs / "universe_robustness",
        cube=outputs / "audit_cube",
        figures=outputs / "figures",
        reports=outputs / "reports",
        manifests=outputs / "manifests",
        cache=outputs / "cache",
        source_snapshot=root / "source_snapshot",
        configs=root / "configs",
    )


def ensure_dirs(paths: Paths) -> None:
    for path in [
        paths.outputs,
        paths.multiplier,
        paths.universe,
        paths.cube,
        paths.figures,
        paths.reports,
        paths.manifests,
        paths.cache,
        paths.cache / "data",
        paths.configs,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_source_imports(paths: Paths) -> None:
    for item in [str(project_root()), str(paths.source_snapshot), str(paths.root / "src")]:
        if item not in sys.path:
            sys.path.insert(0, item)


def import_snapshot_modules(paths: Paths) -> Dict[str, Any]:
    setup_source_imports(paths)
    module_names = [
        "mahoraga6_1",
        "mahoraga14_config",
        "mahoraga14_data",
        "mahoraga14_backtest",
        "acceptance_suite_14_3R",
        "promotion_gate_suite",
        "fast_fail_diagnostics_14_3",
    ]
    return {name: importlib.import_module(name) for name in module_names}


def generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_run_id() -> str:
    now = generated_at()
    return f"ext14_3_{now.replace(':', '').replace('-', '').replace('+', 'Z')}"


def candidate_id_from_knobs(knobs: Dict[str, float]) -> str:
    return (
        f"B{float(knobs['budget_multiplier']):.2f}_"
        f"C{float(knobs['conviction_multiplier']):.2f}_"
        f"L{float(knobs['leader_multiplier']):.2f}_"
        f"R{float(knobs['backoff_strength']):.2f}"
    )


def knobs_with(**updates: float) -> Dict[str, float]:
    out = dict(OFFICIAL_KNOBS)
    out.update({k: float(v) for k, v in updates.items()})
    return out


def candidate_spec(candidate_id: str, knobs: Dict[str, float], role: str) -> Dict[str, Any]:
    return {
        "CandidateId": candidate_id,
        "GateRole": role,
        "budget_multiplier": float(knobs["budget_multiplier"]),
        "conviction_multiplier": float(knobs["conviction_multiplier"]),
        "leader_multiplier": float(knobs["leader_multiplier"]),
        "backoff_strength": float(knobs["backoff_strength"]),
    }


def unique_specs(specs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for spec in specs:
        cid = str(spec["CandidateId"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(spec)
    return out


def one_dimensional_specs() -> List[Dict[str, Any]]:
    specs = [candidate_spec(OFFICIAL_CANDIDATE_ID, OFFICIAL_KNOBS, "OFFICIAL_REFERENCE")]
    for axis, values in MULTIPLIER_RANGES.items():
        for value in values:
            knobs = knobs_with(**{axis: value})
            specs.append(candidate_spec(candidate_id_from_knobs(knobs), knobs, f"ONE_DIM_{axis}"))
    return unique_specs(specs)


def two_dimensional_specs(axes: Sequence[str]) -> List[Dict[str, Any]]:
    if len(axes) != 2:
        return []
    a, b = axes
    specs: List[Dict[str, Any]] = []
    for av in MULTIPLIER_RANGES[a]:
        for bv in MULTIPLIER_RANGES[b]:
            knobs = knobs_with(**{a: av, b: bv})
            specs.append(candidate_spec(candidate_id_from_knobs(knobs), knobs, f"TWO_DIM_{a}__{b}"))
    return unique_specs(specs)


def extreme_specs() -> List[Dict[str, Any]]:
    return [
        candidate_spec(f"EXTREME_{label}", knobs, "CONTROLLED_EXTREME")
        for label, knobs in EXTREME_CASES.items()
    ]


def metadata_columns(df: pd.DataFrame, run_id: str, candidate_id: Optional[str] = None, universe_id: str = "base_universe_12") -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    out["analysis_phase"] = PHASE
    if candidate_id is not None and "candidate_id" not in out.columns and "CandidateId" not in out.columns:
        out["candidate_id"] = candidate_id
    out["universe_id"] = universe_id
    out["baseline_reference"] = BASELINE_REFERENCE
    out["generated_at"] = generated_at()
    return out


def md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_No rows available._"
    return "```\n" + df.to_string(index=False) + "\n```"


def prepare_cfg(paths: Paths, modules: Dict[str, Any], universe_id: str, tickers: Optional[Sequence[str]] = None):
    cfg = modules["mahoraga14_config"].Mahoraga14Config()
    cfg.run_mode = "FAST"
    cfg.make_plots_flag = False
    cfg.max_outer_jobs = min(int(getattr(cfg, "max_outer_jobs", 2)), 3)
    cfg.outer_parallel = True
    cfg.cache_dir = str(paths.cache / "data")
    cfg.outputs_dir = str(paths.outputs / "_unused_baseline_outputs")
    cfg.plots_dir = str(paths.outputs / "_unused_baseline_outputs" / "figures")
    cfg.audit_dir = str(paths.outputs / "_unused_baseline_audit")
    cfg.paper_pack_dir = str(paths.outputs / "_unused_paper_pack")
    cfg.docs_dir = str(paths.outputs / "_unused_docs")
    cfg.manifests_dir = str(paths.outputs / "_unused_manifests")
    cfg.config_dir = str(paths.outputs / "_unused_config")
    cfg.random_seed = 42
    if tickers is not None:
        cfg.universe_static = tuple(tickers)
    cfg.extended_analysis_universe_id = universe_id
    return cfg


def seed_research_cache(paths: Paths) -> List[str]:
    copied: List[str] = []
    dest = paths.cache / "data"
    dest.mkdir(parents=True, exist_ok=True)
    candidates = [project_root() / "data_cache", paths.source_snapshot / "data_cache"]
    for src in candidates:
        if not src.exists():
            continue
        for file in src.glob("*.pkl"):
            target = dest / file.name
            if not target.exists():
                shutil.copy2(file, target)
                copied.append(str(target.relative_to(paths.root)).replace("\\", "/"))
    return copied


def cache_path(paths: Paths, universe_id: str, kind: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in universe_id)
    return paths.cache / f"{kind}_{safe}.joblib"


def load_or_run_wf(paths: Paths, modules: Dict[str, Any], universe_id: str, tickers: Sequence[str], force: bool = False) -> Tuple[Dict[str, Any], Any, Any, Optional[pd.DataFrame], Any, bool, float]:
    t0 = time.perf_counter()
    cfg = prepare_cfg(paths, modules, universe_id, tickers)
    path = cache_path(paths, universe_id, "wf")
    if path.exists() and not force:
        payload = joblib.load(path)
        return payload["wf"], payload["cfg"], payload.get("ohlcv"), payload.get("universe_schedule"), payload.get("universe_snaps"), True, time.perf_counter() - t0

    costs = modules["mahoraga6_1"].CostsConfig()
    ohlcv, universe_schedule, _ff, universe_snaps = modules["mahoraga14_data"].load_inputs(cfg)
    wf = modules["mahoraga14_backtest"].run_walk_forward_mahoraga14(ohlcv, cfg, costs, universe_schedule)
    payload = {
        "wf": wf,
        "cfg": cfg,
        "ohlcv": ohlcv,
        "universe_schedule": universe_schedule,
        "universe_snaps": universe_snaps,
        "generated_at": generated_at(),
        "universe_id": universe_id,
        "tickers": list(tickers),
    }
    joblib.dump(payload, path, compress=3)
    return wf, cfg, ohlcv, universe_schedule, universe_snaps, False, time.perf_counter() - t0


def load_inputs_for_coverage(paths: Paths, modules: Dict[str, Any], universe_id: str, tickers: Sequence[str]) -> Tuple[Any, Optional[pd.DataFrame], Any]:
    cfg = prepare_cfg(paths, modules, universe_id, tickers)
    ohlcv, universe_schedule, _ff, universe_snaps = modules["mahoraga14_data"].load_inputs(cfg)
    return ohlcv, universe_schedule, universe_snaps


def coverage_audit_for_universe(paths: Paths, modules: Dict[str, Any], universe_id: str, tickers: Sequence[str], force: bool = False) -> Tuple[pd.DataFrame, Any, Optional[pd.DataFrame], Any, bool]:
    path = cache_path(paths, universe_id, "coverage")
    if path.exists() and not force:
        payload = joblib.load(path)
        return payload["coverage"], payload.get("ohlcv"), payload.get("universe_schedule"), payload.get("universe_snaps"), True
    ohlcv, universe_schedule, universe_snaps = load_inputs_for_coverage(paths, modules, universe_id, tickers)
    close = ohlcv["close"]
    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in close.columns:
            rows.append(
                {
                    "universe_id": universe_id,
                    "ticker": ticker,
                    "proposed_flag": 1,
                    "usable_flag": 0,
                    "first_valid_date": "",
                    "last_valid_date": "",
                    "coverage_ratio": 0.0,
                    "missing_observations": np.nan,
                    "limitation": "ticker missing from downloaded OHLCV panel",
                }
            )
            continue
        s = close[ticker].dropna()
        idx = close.index
        ratio = float(len(s) / max(1, len(idx)))
        first = s.index.min() if len(s) else pd.NaT
        last = s.index.max() if len(s) else pd.NaT
        usable = int(len(s) >= 756 and ratio >= 0.50)
        rows.append(
            {
                "universe_id": universe_id,
                "ticker": ticker,
                "proposed_flag": 1,
                "usable_flag": usable,
                "first_valid_date": "" if pd.isna(first) else str(pd.Timestamp(first).date()),
                "last_valid_date": "" if pd.isna(last) else str(pd.Timestamp(last).date()),
                "coverage_ratio": ratio,
                "missing_observations": int(len(idx) - len(s)),
                "limitation": "" if usable else "insufficient history or sparse coverage for robust walk-forward inference",
            }
        )
    coverage = pd.DataFrame(rows)
    joblib.dump(
        {
            "coverage": coverage,
            "ohlcv": ohlcv,
            "universe_schedule": universe_schedule,
            "universe_snaps": universe_snaps,
            "generated_at": generated_at(),
        },
        path,
        compress=3,
    )
    return coverage, ohlcv, universe_schedule, universe_snaps, False


def build_context(modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    return modules["acceptance_suite_14_3R"]._stitched_base_context(wf, cfg)


def apply_candidate(modules: Dict[str, Any], context: Dict[str, Any], cfg: Any, knobs: Dict[str, float]) -> Dict[str, Any]:
    return modules["acceptance_suite_14_3R"]._apply_frozen_knobs(
        context["primary"],
        context["allocator"],
        context["leader_diag"],
        context["qqq"],
        cfg,
        float(knobs["budget_multiplier"]),
        float(knobs["conviction_multiplier"]),
        float(knobs["leader_multiplier"]),
        float(knobs["backoff_strength"]),
    )


def metrics_row(modules: Dict[str, Any], label: str, obj: Dict[str, Any], context: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    raw = modules["fast_fail_diagnostics_14_3"]._metrics_row(label, obj, context["qqq"], context["spy"], cfg)
    return {
        "CAGR": float(raw["CAGR"]) * 100.0,
        "Sharpe": float(raw["Sharpe"]),
        "Sortino": float(raw["Sortino"]),
        "MaxDD": float(raw["MaxDD"]) * 100.0,
        "BetaQQQ": float(raw["BetaQQQ"]),
        "BetaSPY": float(raw["BetaSPY"]),
        "AlphaNW_QQQ": float(raw["AlphaNW_QQQ"]),
        "AlphaNW_SPY": float(raw["AlphaNW_SPY"]),
        "UpsideCaptureQQQ": float(raw["UpsideCaptureQQQ"]),
        "DownsideCaptureQQQ": float(raw["DownsideCaptureQQQ"]),
        "AvgExposure": float(raw["AvgExposure"]),
        "AvgTurnover": float(raw["AvgTurnover"]),
        "ReturnPerExposure": float(raw["ReturnPerExposure"]),
    }


def slice_object(modules: Dict[str, Any], obj: Dict[str, Any], start: Any, end: Any, cfg: Any) -> Dict[str, Any]:
    return modules["acceptance_suite_14_3R"]._slice_object(obj, pd.Timestamp(start), pd.Timestamp(end), cfg)


def fold_metrics_df(modules: Dict[str, Any], wf: Dict[str, Any], obj_map: Dict[str, Dict[str, Any]], context: Dict[str, Any], cfg: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        qqq_slice = slice_object(modules, context["qqq"], start, end, cfg)
        spy_slice = slice_object(modules, context["spy"], start, end, cfg)
        for cid, obj in obj_map.items():
            sliced = slice_object(modules, obj, start, end, cfg)
            met = modules["fast_fail_diagnostics_14_3"]._metrics_row(cid, sliced, qqq_slice, spy_slice, cfg)
            rows.append(
                {
                    "Fold": fold,
                    "TestStart": str(start.date()),
                    "TestEnd": str(end.date()),
                    "CandidateId": cid,
                    "CAGR": float(met["CAGR"]) * 100.0,
                    "Sharpe": float(met["Sharpe"]),
                    "Sortino": float(met["Sortino"]),
                    "MaxDD": float(met["MaxDD"]) * 100.0,
                    "AlphaNW_QQQ": float(met["AlphaNW_QQQ"]),
                    "AlphaNW_SPY": float(met["AlphaNW_SPY"]),
                    "UpsideCaptureQQQ": float(met["UpsideCaptureQQQ"]),
                    "DownsideCaptureQQQ": float(met["DownsideCaptureQQQ"]),
                    "Exposure": float(met["AvgExposure"]),
                    "Turnover": float(met["AvgTurnover"]),
                }
            )
    return pd.DataFrame(rows)


def active_return_stats(obj: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, float]:
    r = pd.Series(obj["returns"], dtype=float)
    q = pd.Series(context["qqq"]["returns"], dtype=float).reindex(r.index).fillna(0.0)
    active = r - q
    cum = (1.0 + active).cumprod() - 1.0
    if len(cum) == 0:
        return 0.0, 0.0
    return float(cum.iloc[-1]), float(cum.min())


def priority_window_stats(modules: Dict[str, Any], candidate_obj: Dict[str, Any], context: Dict[str, Any], cfg: Any) -> Tuple[int, pd.DataFrame]:
    windows = modules["fast_fail_diagnostics_14_3"]._window_specs(
        pd.DatetimeIndex(context["qqq"]["returns"].index),
        context["qqq"]["returns"],
        cfg,
    )
    rows: List[Dict[str, Any]] = []
    for window_name, start, end, source in windows:
        cand_slice = slice_object(modules, candidate_obj, start, end, cfg)
        ctrl_slice = slice_object(modules, context["control"], start, end, cfg)
        qqq_slice = slice_object(modules, context["qqq"], start, end, cfg)
        cand_vs_qqq = modules["fast_fail_diagnostics_14_3"]._window_summary(cand_slice, qqq_slice, cfg, str(window_name))
        ctrl_vs_qqq = modules["fast_fail_diagnostics_14_3"]._window_summary(ctrl_slice, qqq_slice, cfg, str(window_name))
        qqq_return = float(np.prod(1.0 + pd.Series(qqq_slice["returns"], dtype=float).values) - 1.0)
        delta_control = float(cand_vs_qqq["Return"] - ctrl_vs_qqq["Return"])
        delta_qqq = float(cand_vs_qqq["Return"] - qqq_return)
        if delta_control >= 0.0 and delta_qqq >= -0.02:
            status = "PASS"
        elif delta_control >= -0.02 and delta_qqq >= -0.05:
            status = "WATCH"
        else:
            status = "FAIL"
        rows.append(
            {
                "Window": str(window_name),
                "Source": str(source),
                "Start": str(pd.Timestamp(start).date()),
                "End": str(pd.Timestamp(end).date()),
                "CandidateReturn": float(cand_vs_qqq["Return"]),
                "ControlReturn": float(ctrl_vs_qqq["Return"]),
                "QQQReturn": qqq_return,
                "DeltaReturn_vs_Control": delta_control,
                "DeltaReturn_vs_QQQ": delta_qqq,
                "GateStatus": status,
            }
        )
    priority = pd.DataFrame(rows)
    priority = priority[priority["Window"].isin(["2017_2018", "2020_2021", "2023_2024"])] if len(priority) else priority
    pass_count = int((priority["GateStatus"] == "PASS").sum()) if len(priority) else 0
    return pass_count, priority


def evaluate_specs(
    paths: Paths,
    modules: Dict[str, Any],
    wf: Dict[str, Any],
    cfg: Any,
    specs: Sequence[Dict[str, Any]],
    universe_id: str,
    run_id: str,
    role_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    context = build_context(modules, wf, cfg)
    specs = unique_specs(specs)
    objects: Dict[str, Dict[str, Any]] = {}
    priority_map: Dict[str, pd.DataFrame] = {}

    official_obj = apply_candidate(modules, context, cfg, OFFICIAL_KNOBS)
    objects[OFFICIAL_CANDIDATE_ID] = official_obj
    official_metrics = metrics_row(modules, OFFICIAL_CANDIDATE_ID, official_obj, context, cfg)
    official_folds = fold_metrics_df(modules, wf, {OFFICIAL_CANDIDATE_ID: official_obj}, context, cfg)
    official_fold_map = official_folds.set_index("Fold")

    rows: List[Dict[str, Any]] = []
    fold_rows: List[pd.DataFrame] = []
    for spec in specs:
        cid = str(spec["CandidateId"])
        knobs = {
            "budget_multiplier": float(spec["budget_multiplier"]),
            "conviction_multiplier": float(spec["conviction_multiplier"]),
            "leader_multiplier": float(spec["leader_multiplier"]),
            "backoff_strength": float(spec["backoff_strength"]),
        }
        obj = official_obj if cid == OFFICIAL_CANDIDATE_ID else apply_candidate(modules, context, cfg, knobs)
        objects[cid] = obj
        met = metrics_row(modules, cid, obj, context, cfg)
        final_active, worst_active = active_return_stats(obj, context)
        priority_pass, priority_df = priority_window_stats(modules, obj, context, cfg)
        priority_map[cid] = priority_df

        fdf = fold_metrics_df(modules, wf, {cid: obj}, context, cfg)
        fold_rows.append(fdf)
        merged = fdf.merge(
            official_folds[["Fold", "Sharpe", "CAGR", "MaxDD"]].rename(
                columns={"Sharpe": "OfficialSharpe", "CAGR": "OfficialCAGR", "MaxDD": "OfficialMaxDD"}
            ),
            on="Fold",
            how="left",
        )
        worst_fold_sharpe_delta = float((merged["Sharpe"] - merged["OfficialSharpe"]).min()) if len(merged) else 0.0
        worst_fold_cagr_delta = float((merged["CAGR"] - merged["OfficialCAGR"]).min()) if len(merged) else 0.0
        max_fold_maxdd_worsening = float((merged["OfficialMaxDD"] - merged["MaxDD"]).max()) if len(merged) else 0.0
        severe_fold_damage = int(
            (
                ((merged["Sharpe"] - merged["OfficialSharpe"]) < -0.25)
                | ((merged["CAGR"] - merged["OfficialCAGR"]) < -6.0)
                | ((merged["OfficialMaxDD"] - merged["MaxDD"]) > 5.0)
            ).sum()
        )
        sharpe_drop = (official_metrics["Sharpe"] - met["Sharpe"]) / max(abs(official_metrics["Sharpe"]), 1e-12)
        cagr_drop = (official_metrics["CAGR"] - met["CAGR"]) / max(abs(official_metrics["CAGR"]), 1e-12)
        maxdd_worsening = official_metrics["MaxDD"] - met["MaxDD"]
        robust_region_flag = int(
            (sharpe_drop <= ROBUST_SHARPE_DROP)
            and (cagr_drop <= ROBUST_CAGR_DROP)
            and (maxdd_worsening <= ROBUST_MAXDD_WORSENING_PP)
            and (severe_fold_damage == 0)
        )
        row = {
            "CandidateId": cid,
            "candidate_id": cid,
            "sweep_role": str(spec.get("GateRole", role_label)),
            "universe_id": universe_id,
            **knobs,
            **met,
            "active_return_vs_QQQ_final": final_active,
            "active_return_vs_QQQ_worst_trough": worst_active,
            "priority_window_pass_count": priority_pass,
            "severe_fold_damage_count": severe_fold_damage,
            "worst_fold_sharpe_delta_vs_official": worst_fold_sharpe_delta,
            "worst_fold_cagr_delta_vs_official": worst_fold_cagr_delta,
            "max_fold_maxdd_worsening_vs_official": max_fold_maxdd_worsening,
            "SharpeDropVsOfficial": sharpe_drop,
            "CAGRDropVsOfficial": cagr_drop,
            "MaxDDWorseningVsOfficial": maxdd_worsening,
            "robust_region_flag": robust_region_flag,
            "run_id": run_id,
            "analysis_phase": PHASE,
            "baseline_reference": BASELINE_REFERENCE,
            "generated_at": generated_at(),
        }
        rows.append(row)
    fold_all = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    if len(fold_all):
        fold_all = metadata_columns(fold_all, run_id, universe_id=universe_id)
    return pd.DataFrame(rows).round(8), fold_all.round(8), objects, priority_map


def sensitivity_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    official = summary[summary["CandidateId"] == OFFICIAL_CANDIDATE_ID].iloc[0]
    for axis in MULTIPLIER_RANGES:
        subset = summary[summary["sweep_role"] == f"ONE_DIM_{axis}"].copy()
        if len(subset) == 0:
            continue
        subset = subset[subset["CandidateId"] != OFFICIAL_CANDIDATE_ID].copy()
        if len(subset) == 0:
            continue
        metric = (
            subset["SharpeDropVsOfficial"].clip(lower=0.0)
            + subset["CAGRDropVsOfficial"].clip(lower=0.0)
            + (subset["MaxDDWorseningVsOfficial"].clip(lower=0.0) / ROBUST_MAXDD_WORSENING_PP)
            + subset["severe_fold_damage_count"].clip(lower=0.0)
        )
        max_idx = metric.idxmax()
        rows.append(
            {
                "axis": axis,
                "sensitivity_score": float(metric.max()),
                "mean_sensitivity_score": float(metric.mean()),
                "worst_candidate_id": str(summary.loc[max_idx, "CandidateId"]),
                "worst_sharpe_drop": float(summary.loc[max_idx, "SharpeDropVsOfficial"]),
                "worst_cagr_drop": float(summary.loc[max_idx, "CAGRDropVsOfficial"]),
                "worst_maxdd_worsening": float(summary.loc[max_idx, "MaxDDWorseningVsOfficial"]),
                "worst_severe_fold_damage_count": int(summary.loc[max_idx, "severe_fold_damage_count"]),
                "official_value": float(official[axis]),
                "sampled_values": ",".join(str(x) for x in MULTIPLIER_RANGES[axis]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["sensitivity_score", "mean_sensitivity_score"], ascending=False)
    return out.reset_index(drop=True)


def relative_perturbation(row: pd.Series) -> float:
    vals = []
    for axis, official in OFFICIAL_KNOBS.items():
        vals.append(abs(float(row[axis]) / float(official) - 1.0))
    return float(max(vals))


def plateau_metrics(summary: pd.DataFrame) -> Tuple[float, pd.DataFrame, float]:
    decay = summary[summary["robust_region_flag"] == 0].copy()
    distance_to_decay = float(decay.apply(relative_perturbation, axis=1).min()) if len(decay) else math.inf
    robust_share = float(summary["robust_region_flag"].mean()) if len(summary) else 0.0

    rows: List[Dict[str, Any]] = []
    for axis in MULTIPLIER_RANGES:
        subset = summary[(summary["sweep_role"] == f"ONE_DIM_{axis}") | (summary["CandidateId"] == OFFICIAL_CANDIDATE_ID)].copy()
        good = subset[subset["robust_region_flag"] == 1].copy()
        if len(good) == 0:
            lower = upper = float(OFFICIAL_KNOBS[axis])
            radius = 0.0
        else:
            lower = float(good[axis].min())
            upper = float(good[axis].max())
            radius = min(abs(lower / OFFICIAL_KNOBS[axis] - 1.0), abs(upper / OFFICIAL_KNOBS[axis] - 1.0))
        rows.append(
            {
                "axis": axis,
                "official_value": float(OFFICIAL_KNOBS[axis]),
                "robust_min_sampled_value": lower,
                "robust_max_sampled_value": upper,
                "plateau_radius_relative": radius,
                "plateau_radius_absolute_low": float(OFFICIAL_KNOBS[axis] - lower),
                "plateau_radius_absolute_high": float(upper - OFFICIAL_KNOBS[axis]),
            }
        )
    return distance_to_decay, pd.DataFrame(rows), robust_share


def write_plateau_report(path: Path, distance_to_decay: float, plateau_df: pd.DataFrame, robust_share: float, sensitivity_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    dtd = "not observed in sampled region" if math.isinf(distance_to_decay) else f"{distance_to_decay:.4f}"
    lines = [
        "# Plateau Radius Report",
        "",
        "## Formal definitions",
        "",
        "Let the official multiplier vector be m0 = (1.05, 1.10, 1.10, 1.05). For any sampled candidate m, define relative perturbation d(m,m0)=max_i |m_i/m0_i - 1|.",
        "",
        "`distance_to_decay` is min d(m,m0) over sampled candidates where at least one decay condition holds: Sharpe drop > 10%, CAGR drop > 10%, MaxDD worsening > 5 percentage points, or severe_fold_damage_count > 0.",
        "",
        "`plateau_radius` is computed per axis by holding the other three multipliers at the official values and taking the sampled interval around the official value where all robustness conditions hold.",
        "",
        "`robust_region_share_extended` is the count of sampled candidates satisfying all robustness conditions divided by the total sampled candidate count.",
        "",
        "The 10% relative Sharpe/CAGR and 5 percentage point MaxDD thresholds are not new optimization criteria; they are stress-audit tolerances requested for this phase and interpreted as degradation boundaries, not promotion rules.",
        "",
        "## Results",
        "",
        f"- distance_to_decay: {dtd}",
        f"- robust_region_share_extended: {robust_share:.2%}",
        f"- sampled candidates: {len(summary)}",
        "",
        "## Plateau by axis",
        "",
        md_table(plateau_df),
        "",
        "## Sensitivity ranking",
        "",
        md_table(sensitivity_df) if len(sensitivity_df) else "No sensitivity rows available.",
        "",
        "## Interpretation guardrails",
        "",
        "- These metrics describe stability in the sampled perturbation set only.",
        "- A large plateau does not prove global optimality.",
        "- A small distance_to_decay does not automatically invalidate the baseline; it identifies where additional audit attention is needed.",
        "- Worst-fold degradation is treated as a first-class risk because stitched performance can hide fold-local fragility.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_multiplier_heatmap(summary: pd.DataFrame, sensitivity_df: pd.DataFrame, out_png: Path) -> None:
    if len(sensitivity_df) >= 2:
        axes = list(sensitivity_df["axis"].head(2))
    else:
        axes = ["budget_multiplier", "leader_multiplier"]
    sub = summary[summary["sweep_role"] == f"TWO_DIM_{axes[0]}__{axes[1]}"].copy()
    if len(sub) == 0:
        sub = summary.copy()
    pivot = sub.pivot_table(index=axes[0], columns=axes[1], values="Sharpe", aggfunc="mean")
    plt.figure(figsize=(8.5, 6.0))
    if len(pivot) > 0:
        im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn")
        plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
        plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        plt.colorbar(im, label="Sharpe")
    plt.xlabel(axes[1])
    plt.ylabel(axes[0])
    plt.title("Extended Multiplier Robustness: Sharpe Surface")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_1d_degradation(summary: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10.5, 6.2))
    for axis in MULTIPLIER_RANGES:
        sub = summary[summary["sweep_role"] == f"ONE_DIM_{axis}"].copy().sort_values(axis)
        if len(sub) == 0:
            continue
        plt.plot(sub[axis], sub["SharpeDropVsOfficial"], marker="o", linewidth=1.4, label=axis)
    plt.axhline(ROBUST_SHARPE_DROP, color="black", linewidth=0.9, linestyle="--", label="10% Sharpe drop")
    plt.xlabel("Multiplier value")
    plt.ylabel("Sharpe drop vs official")
    plt.title("One-Dimensional Multiplier Degradation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_multiplier_analysis(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, run_id: str, force: bool = False) -> Dict[str, Any]:
    one_specs = one_dimensional_specs()
    one_df, one_folds, one_objects, one_priority = evaluate_specs(paths, modules, wf, cfg, one_specs, "base_universe_12", run_id, "ONE_DIM")
    sens = sensitivity_ranking(one_df)
    top_axes = list(sens["axis"].head(2)) if len(sens) >= 2 else ["budget_multiplier", "leader_multiplier"]
    two_specs = two_dimensional_specs(top_axes)
    two_df, two_folds, two_objects, two_priority = evaluate_specs(paths, modules, wf, cfg, two_specs, "base_universe_12", run_id, "TWO_DIM")
    ext_specs = extreme_specs()
    ext_df, ext_folds, ext_objects, ext_priority = evaluate_specs(paths, modules, wf, cfg, ext_specs, "base_universe_12", run_id, "EXTREME")

    combined = pd.concat([one_df, two_df, ext_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["CandidateId"], keep="first").reset_index(drop=True)
    fold_df = pd.concat([one_folds, two_folds, ext_folds], ignore_index=True).drop_duplicates(subset=["Fold", "CandidateId"], keep="first")
    distance, plateau_df, robust_share = plateau_metrics(combined)

    one_df.to_csv(paths.multiplier / "one_dimensional_sweeps.csv", index=False)
    two_df.to_csv(paths.multiplier / "two_dimensional_sweeps.csv", index=False)
    ext_df.to_csv(paths.multiplier / "extreme_cases.csv", index=False)
    sens.to_csv(paths.multiplier / "sensitivity_ranking.csv", index=False)
    combined.to_csv(paths.multiplier / "extended_multiplier_summary.csv", index=False)
    fold_df.to_csv(paths.multiplier / "extended_multiplier_fold_summary.csv", index=False)
    plateau_df.to_csv(paths.multiplier / "plateau_radius_by_axis.csv", index=False)
    write_plateau_report(paths.multiplier / "plateau_radius_report.md", distance, plateau_df, robust_share, sens, combined)
    plot_multiplier_heatmap(combined, sens, paths.figures / "extended_multiplier_heatmap.png")
    plot_1d_degradation(combined, paths.figures / "multiplier_1d_degradation.png")

    objects = {**one_objects, **two_objects, **ext_objects}
    priority = {**one_priority, **two_priority, **ext_priority}
    joblib.dump({"objects": objects, "priority": priority, "summary": combined}, paths.cache / "candidate_objects_base_universe_12.joblib", compress=3)
    return {
        "summary": combined,
        "folds": fold_df,
        "sensitivity": sens,
        "top_axes": top_axes,
        "objects": objects,
        "priority": priority,
        "distance_to_decay": distance,
        "plateau_df": plateau_df,
        "robust_share": robust_share,
    }


def summarize_universe_coverage(coverage: pd.DataFrame) -> Dict[str, Any]:
    usable = coverage[coverage["usable_flag"] == 1]
    if len(usable):
        start = pd.to_datetime(usable["first_valid_date"], errors="coerce").max()
        end = pd.to_datetime(usable["last_valid_date"], errors="coerce").min()
    else:
        start = pd.NaT
        end = pd.NaT
    return {
        "proposed_count": int(len(coverage)),
        "usable_count": int(coverage["usable_flag"].sum()),
        "usable_tickers": ",".join(usable["ticker"].tolist()),
        "missing_tickers": ",".join(coverage.loc[coverage["usable_flag"] == 0, "ticker"].tolist()),
        "effective_start": "" if pd.isna(start) else str(pd.Timestamp(start).date()),
        "effective_end": "" if pd.isna(end) else str(pd.Timestamp(end).date()),
        "mean_coverage_ratio": float(coverage["coverage_ratio"].mean()) if len(coverage) else 0.0,
    }


def run_universe_analysis(
    paths: Paths,
    modules: Dict[str, Any],
    run_id: str,
    force: bool = False,
    max_new_universe_runs: int = 1,
) -> Dict[str, Any]:
    coverage_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    run_meta: List[Dict[str, Any]] = []
    new_universe_runs = 0
    for universe_id, tickers in UNIVERSES.items():
        t0 = time.perf_counter()
        coverage, _ohlcv_cov, _sched_cov, _snaps_cov, coverage_cached = coverage_audit_for_universe(paths, modules, universe_id, tickers, force=force)
        coverage_rows.append(coverage)
        cov_summary = summarize_universe_coverage(coverage)
        if cov_summary["usable_count"] < 6:
            for cid, knobs in UNIVERSE_STRESS_CANDIDATES.items():
                summary_rows.append(
                    {
                        "universe_id": universe_id,
                        "CandidateId": cid,
                        "candidate_id": cid,
                        "run_status": "SKIPPED_INSUFFICIENT_COVERAGE",
                        **cov_summary,
                        **knobs,
                        "CAGR": np.nan,
                        "Sharpe": np.nan,
                        "Sortino": np.nan,
                        "MaxDD": np.nan,
                        "AlphaNW_QQQ": np.nan,
                        "AlphaNW_SPY": np.nan,
                        "UpsideCaptureQQQ": np.nan,
                        "DownsideCaptureQQQ": np.nan,
                        "AvgExposure": np.nan,
                        "AvgTurnover": np.nan,
                        "ReturnPerExposure": np.nan,
                        "analysis_phase": PHASE,
                        "baseline_reference": BASELINE_REFERENCE,
                        "run_id": run_id,
                        "generated_at": generated_at(),
                    }
                )
            run_meta.append({"universe_id": universe_id, "coverage_cached": coverage_cached, "wf_cached": None, "seconds": time.perf_counter() - t0, "status": "skipped"})
            continue
        wf_exists = cache_path(paths, universe_id, "wf").exists() and not force
        if (not wf_exists) and universe_id != "base_universe_12" and new_universe_runs >= max_new_universe_runs:
            for cid, knobs in UNIVERSE_STRESS_CANDIDATES.items():
                summary_rows.append(
                    {
                        "universe_id": universe_id,
                        "CandidateId": cid,
                        "candidate_id": cid,
                        "run_status": "ABORTED_COMPUTE_BUDGET",
                        **cov_summary,
                        **knobs,
                        "CAGR": np.nan,
                        "Sharpe": np.nan,
                        "Sortino": np.nan,
                        "MaxDD": np.nan,
                        "AlphaNW_QQQ": np.nan,
                        "AlphaNW_SPY": np.nan,
                        "UpsideCaptureQQQ": np.nan,
                        "DownsideCaptureQQQ": np.nan,
                        "AvgExposure": np.nan,
                        "AvgTurnover": np.nan,
                        "ReturnPerExposure": np.nan,
                        "analysis_phase": PHASE,
                        "baseline_reference": BASELINE_REFERENCE,
                        "run_id": run_id,
                        "generated_at": generated_at(),
                    }
                )
            run_meta.append(
                {
                    "universe_id": universe_id,
                    "coverage_cached": coverage_cached,
                    "wf_cached": False,
                    "seconds": time.perf_counter() - t0,
                    "status": "aborted_compute_budget",
                    "reason": f"max_new_universe_runs={max_new_universe_runs} reached after observed high runtime",
                }
            )
            continue
        wf, cfg, _ohlcv, _sched, _snaps, wf_cached, seconds = load_or_run_wf(paths, modules, universe_id, tickers, force=force)
        if not wf_cached and universe_id != "base_universe_12":
            new_universe_runs += 1
        specs = [candidate_spec(cid, knobs, "UNIVERSE_STRESS") for cid, knobs in UNIVERSE_STRESS_CANDIDATES.items()]
        udf, _folds, _objects, _priority = evaluate_specs(paths, modules, wf, cfg, specs, universe_id, run_id, "UNIVERSE_STRESS")
        udf["run_status"] = "OK"
        for key, value in cov_summary.items():
            udf[key] = value
        summary_rows.extend(udf.to_dict("records"))
        run_meta.append({"universe_id": universe_id, "coverage_cached": coverage_cached, "wf_cached": wf_cached, "seconds": seconds, "status": "ok"})
    coverage_all = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    coverage_all.to_csv(paths.universe / "universe_coverage_audit.csv", index=False)
    summary.to_csv(paths.universe / "universe_robustness_summary.csv", index=False)
    write_universe_report(paths.universe / "universe_robustness_report.md", coverage_all, summary, run_meta)
    plot_universe_comparison(summary, paths.figures / "universe_robustness_comparison.png")
    pd.DataFrame(run_meta).to_csv(paths.universe / "universe_run_meta.csv", index=False)
    return {"coverage": coverage_all, "summary": summary, "run_meta": run_meta}


def write_universe_report(path: Path, coverage: pd.DataFrame, summary: pd.DataFrame, run_meta: List[Dict[str, Any]]) -> None:
    ok = summary[summary["run_status"] == "OK"].copy() if "run_status" in summary.columns else summary
    official = ok[ok["CandidateId"] == OFFICIAL_CANDIDATE_ID].copy() if len(ok) else pd.DataFrame()
    lines = [
        "# Universe Robustness Report",
        "",
        "## Methodology",
        "",
        "Each alternate universe is treated as an input-universe stress, not a new baseline search. The official multiplier vector is evaluated first; at most two controlled extremes are added to test whether risk-seeking or defensive perturbations explain major degradation.",
        "",
        "The negative control is non-technology by design and is interpreted only as a technical sanity check, not as evidence for or against an economic technology edge.",
        "",
        "## Coverage summary",
        "",
        md_table(coverage.groupby("universe_id").agg(
            proposed=("ticker", "count"),
            usable=("usable_flag", "sum"),
            mean_coverage=("coverage_ratio", "mean"),
        ).reset_index())
        if len(coverage)
        else "No coverage rows available.",
        "",
        "## Official candidate by universe",
        "",
        md_table(official[["universe_id", "CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "run_status"]])
        if len(official)
        else "No successful official universe runs available.",
        "",
        "## Run metadata",
        "",
        md_table(pd.DataFrame(run_meta)) if run_meta else "No run metadata available.",
        "",
        "## Interpretation",
        "",
        "- A collapse in the negative control is expected and should not be read as a failed technology edge.",
        "- Changes across tech universes can arise from composition, seasoning, volatility, and canonical schedule membership, not only from model failure.",
        "- Alternate universe runs reuse the same policy-layer architecture and official multipliers; they are not reoptimized for each universe.",
        "- Coverage gaps are first-class limitations and are not hidden by forced backfills.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_universe_comparison(summary: pd.DataFrame, out_png: Path) -> None:
    ok = summary[(summary.get("run_status", "OK") == "OK") & (summary["CandidateId"] == OFFICIAL_CANDIDATE_ID)].copy() if len(summary) else pd.DataFrame()
    plt.figure(figsize=(10, 5.8))
    if len(ok):
        x = np.arange(len(ok))
        plt.bar(x - 0.18, ok["Sharpe"], width=0.36, label="Sharpe")
        plt.bar(x + 0.18, ok["CAGR"] / 20.0, width=0.36, label="CAGR / 20")
        plt.xticks(x, ok["universe_id"], rotation=25, ha="right")
    plt.title("Official Candidate Across Universes")
    plt.ylabel("Scaled metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def candidate_for_rep_cube(mult: Dict[str, Any]) -> List[str]:
    summary = mult["summary"]
    sens = mult["sensitivity"]
    reps = [OFFICIAL_CANDIDATE_ID, "EXTREME_pro-risk", "EXTREME_pro-defense"]
    if len(sens):
        for cid in sens["worst_candidate_id"].head(2):
            if cid not in reps:
                reps.append(str(cid))
    robust = summary[(summary["robust_region_flag"] == 1) & (~summary["CandidateId"].isin(reps))].copy()
    if len(robust):
        robust["_dist"] = robust.apply(relative_perturbation, axis=1)
        mid = robust.sort_values("_dist", ascending=False).head(1)["CandidateId"].iloc[0]
        if mid not in reps:
            reps.append(str(mid))
    return reps[:7]


def _fold_for_date(wf: Dict[str, Any], date: pd.Timestamp) -> int:
    for result in wf["results"]:
        if pd.Timestamp(result["test_start"]) <= date <= pd.Timestamp(result["test_end"]):
            return int(result["fold"])
    return -1


def _stitch_variant_frame(modules: Dict[str, Any], wf: Dict[str, Any], variant_key: str, field: str) -> pd.DataFrame:
    return modules["fast_fail_diagnostics_14_3"]._stitch_variant_frame(wf["results"], variant_key, field)


def _series_from_frame(frame: pd.DataFrame, column: str, idx: pd.Index, default: float = 0.0) -> pd.Series:
    if frame is None or len(frame) == 0 or column not in frame.columns:
        return pd.Series(default, index=idx, dtype=float)
    return pd.Series(frame[column], dtype=float).reindex(idx).ffill().fillna(default)


def build_decision_cube(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, candidate_objects: Dict[str, Dict[str, Any]], rep_candidates: Sequence[str], run_id: str) -> pd.DataFrame:
    context = build_context(modules, wf, cfg)
    idx = pd.DatetimeIndex(context["primary"]["returns"].index)
    allocator = context["allocator"].reindex(idx).ffill()
    override = _stitch_variant_frame(modules, wf, cfg.primary_variant_key, "override_daily").reindex(idx).ffill()
    rows: List[Dict[str, Any]] = []
    for cid in rep_candidates:
        obj = candidate_objects[cid]
        exposure = pd.Series(obj["exposure"], dtype=float).reindex(idx).fillna(0.0)
        turnover = pd.Series(obj["turnover"], dtype=float).reindex(idx).fillna(0.0)
        for dt in idx:
            a = allocator.loc[dt] if dt in allocator.index else pd.Series(dtype=float)
            o = override.loc[dt] if dt in override.index else pd.Series(dtype=float)
            rows.append(
                {
                    "date": dt,
                    "fold": _fold_for_date(wf, dt),
                    "candidate_id": cid,
                    "universe_id": "base_universe_12",
                    "participation_state": str(a.get("participation_state", "")),
                    "long_budget": float(a.get("long_budget", np.nan)),
                    "gate_scale": float(o.get("gate_scale", a.get("gate_scale_adjustment", np.nan))),
                    "vol_mult": float(o.get("vol_mult", a.get("vol_mult_adjustment", np.nan))),
                    "exp_cap": float(o.get("exp_cap", a.get("exp_cap_adjustment", np.nan))),
                    "leader_blend": float(a.get("leader_blend", np.nan)),
                    "conviction_multiplier": float(a.get("conviction_multiplier", np.nan)),
                    "backoff_strength_applied": float(a.get("risk_backoff_score", np.nan)),
                    "continuation_trigger_p": float(a.get("continuation_trigger_p", a.get("continuation_allocator_score", np.nan))),
                    "continuation_pressure_p": float(a.get("continuation_pressure_p", a.get("participation_pressure_score", np.nan))),
                    "continuation_break_risk_p": float(a.get("continuation_break_risk_p", a.get("break_risk_score", np.nan))),
                    "structural_p": float(o.get("p_structural", o.get("structural_p", np.nan))),
                    "crisis_gate": float(o.get("crisis_gate", np.nan)),
                    "turbulence_scale": float(o.get("turb_scale", np.nan)),
                    "correlation_shield": float(o.get("corr_shield", np.nan)),
                    "cash_budget_target": float(a.get("cash_budget_target", np.nan)),
                    "expected_exposure": float(exposure.loc[dt]),
                    "expected_turnover": float(turnover.loc[dt]),
                    "override_state": str(o.get("override_state", o.get("state", ""))),
                    "hard_backoff_flag": int(float(a.get("risk_backoff_hard_guard", 0.0)) > 0.0),
                    "run_id": run_id,
                    "analysis_phase": PHASE,
                    "baseline_reference": BASELINE_REFERENCE,
                    "generated_at": generated_at(),
                }
            )
    return pd.DataFrame(rows)


def build_position_cube(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, candidate_objects: Dict[str, Dict[str, Any]], rep_candidates: Sequence[str], run_id: str) -> pd.DataFrame:
    context = build_context(modules, wf, cfg)
    idx = pd.DatetimeIndex(context["primary"]["returns"].index)
    rows: List[Dict[str, Any]] = []
    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        test_idx = idx[(idx >= start) & (idx <= end)]
        base_scores = result["variant_bts"][cfg.combo_variant_key].get("scores", pd.DataFrame()).reindex(test_idx)
        base_weights = result.get("base_weights_exec_1x", pd.DataFrame()).reindex(test_idx)
        primary_weights = result["variant_runs"][cfg.combo_variant_key].get("weights_exec_1x", pd.DataFrame()).reindex(test_idx)
        leader_diag = result["variant_runs"][cfg.combo_variant_key].get("leader_diagnostics", pd.DataFrame()).reindex(test_idx).ffill()
        stop_hits = result["variant_bts"][cfg.combo_variant_key].get("stop_hits", pd.DataFrame()).reindex(test_idx)
        rets = result["stress_pre"]["rets"].reindex(test_idx)
        tickers = list(primary_weights.columns)
        for cid in rep_candidates:
            obj = candidate_objects[cid]
            exposure_scale = pd.Series(obj["exposure"], dtype=float).reindex(test_idx).fillna(0.0) / pd.Series(context["primary"]["exposure"], dtype=float).reindex(test_idx).replace(0.0, np.nan)
            exposure_scale = exposure_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 2.0)
            for dt in test_idx:
                weights_row = primary_weights.loc[dt].fillna(0.0) * float(exposure_scale.loc[dt])
                scores_row = base_scores.loc[dt].fillna(0.0) if len(base_scores.columns) else pd.Series(0.0, index=tickers)
                ranks = scores_row.rank(ascending=False, method="min")
                next_1 = rets.shift(-1).loc[dt] if dt in rets.index else pd.Series(np.nan, index=tickers)
                next_5 = (1.0 + rets).rolling(5).apply(np.prod, raw=True).shift(-5).loc[dt] - 1.0 if dt in rets.index else pd.Series(np.nan, index=tickers)
                next_20 = (1.0 + rets).rolling(20).apply(np.prod, raw=True).shift(-20).loc[dt] - 1.0 if dt in rets.index else pd.Series(np.nan, index=tickers)
                for ticker in tickers:
                    final_weight = float(weights_row.get(ticker, 0.0))
                    rows.append(
                        {
                            "date": dt,
                            "ticker": ticker,
                            "fold": fold,
                            "candidate_id": cid,
                            "universe_id": "base_universe_12",
                            "base_score": float(scores_row.get(ticker, np.nan)),
                            "raw_trend": np.nan,
                            "raw_momentum": np.nan,
                            "relative_momentum": np.nan,
                            "residual_score": np.nan,
                            "beta_drag": np.nan,
                            "rank": float(ranks.get(ticker, np.nan)),
                            "selected_flag": int(final_weight > 0.0),
                            "base_weight": float(base_weights.loc[dt].get(ticker, 0.0)) if ticker in base_weights.columns else 0.0,
                            "leader_flag": int(float(leader_diag.loc[dt].get("leader_active_weight", 0.0)) > 0.0 and final_weight > 0.0) if len(leader_diag) else 0,
                            "leader_adjusted_weight": final_weight,
                            "final_weight": final_weight,
                            "stop_flag": int(float(stop_hits.loc[dt].get(ticker, 0.0)) > 0.0) if len(stop_hits.columns) and ticker in stop_hits.columns else 0,
                            "next_ret_1d": float(next_1.get(ticker, np.nan)) if hasattr(next_1, "get") else np.nan,
                            "next_ret_5d": float(next_5.get(ticker, np.nan)) if hasattr(next_5, "get") else np.nan,
                            "next_ret_20d": float(next_20.get(ticker, np.nan)) if hasattr(next_20, "get") else np.nan,
                            "pnl_contribution": final_weight * float(rets.loc[dt].get(ticker, 0.0)) if ticker in rets.columns else 0.0,
                            "run_id": run_id,
                            "analysis_phase": PHASE,
                            "baseline_reference": BASELINE_REFERENCE,
                            "generated_at": generated_at(),
                        }
                    )
    return pd.DataFrame(rows)


def _json_summary(values: Dict[str, Any]) -> str:
    clean: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.floating, np.integer)):
            clean[key] = float(value)
        elif isinstance(value, (float, int, str, bool)) or value is None:
            if isinstance(value, float) and not np.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        else:
            clean[key] = str(value)
    return json.dumps(clean, sort_keys=True)


def build_module_trace_cube(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, rep_candidates: Sequence[str], run_id: str) -> pd.DataFrame:
    context = build_context(modules, wf, cfg)
    idx = pd.DatetimeIndex(context["primary"]["returns"].index)
    allocator = context["allocator"].reindex(idx).ffill()
    leader_diag = context["leader_diag"].reindex(idx).ffill()
    override = _stitch_variant_frame(modules, wf, cfg.primary_variant_key, "override_daily").reindex(idx).ffill()
    rows: List[Dict[str, Any]] = []
    for cid in rep_candidates:
        for dt in idx:
            a = allocator.loc[dt] if dt in allocator.index else pd.Series(dtype=float)
            l = leader_diag.loc[dt] if dt in leader_diag.index else pd.Series(dtype=float)
            o = override.loc[dt] if dt in override.index else pd.Series(dtype=float)
            module_payloads = {
                "BASE_ALPHA_V2": (
                    "selected_book" if True else "flat",
                    float(context["primary"]["exposure"].reindex(idx).fillna(0.0).loc[dt]),
                    {"score_source": "combo_variant_scores", "fold": _fold_for_date(wf, dt)},
                    {"primary_return": float(context["primary"]["returns"].reindex(idx).fillna(0.0).loc[dt])},
                ),
                "continuation_v2_model": (
                    "quality_filter",
                    float(a.get("continuation_allocator_score", 0.0)),
                    {"pressure": float(a.get("participation_pressure_score", np.nan)), "break_risk": float(a.get("break_risk_score", np.nan))},
                    {"continuation_score": float(a.get("continuation_allocator_score", np.nan))},
                ),
                "structural_defense_model": (
                    "defense_blend" if float(o.get("defense_blend", 0.0)) > 0.0 else "no_defense_blend",
                    float(o.get("defense_blend", 0.0)),
                    {"structural_p": float(o.get("p_structural", np.nan))},
                    {"defense_blend": float(o.get("defense_blend", np.nan))},
                ),
                "participation_allocator_v2": (
                    str(a.get("participation_state", "")),
                    float(a.get("long_budget", np.nan)),
                    {"benchmark_strength": float(a.get("benchmark_strength_score", np.nan)), "breadth": float(a.get("breadth_health_score", np.nan))},
                    {"long_budget": float(a.get("long_budget", np.nan)), "cash_budget_target": float(a.get("cash_budget_target", np.nan))},
                ),
                "conviction_amplifier_layer": (
                    "amplified" if float(a.get("conviction_multiplier", 1.0)) > 1.0 else "neutral",
                    float(a.get("conviction_amplifier_score", np.nan)),
                    {"conviction_regime": float(a.get("conviction_regime_score", np.nan))},
                    {"conviction_multiplier": float(a.get("conviction_multiplier", np.nan))},
                ),
                "risk_backoff_layer_v2": (
                    "hard_backoff" if float(a.get("risk_backoff_hard_guard", 0.0)) > 0.0 else "soft_or_none",
                    float(a.get("risk_backoff_score", np.nan)),
                    {"fragility": float(a.get("fragility_score", np.nan)), "benchmark_weakness": float(a.get("benchmark_weakness_score", np.nan))},
                    {"risk_backoff_score": float(a.get("risk_backoff_score", np.nan)), "hard_guard": float(a.get("risk_backoff_hard_guard", np.nan))},
                ),
                "leader_participation_layer": (
                    "leader_active" if float(l.get("leader_active_weight", 0.0)) > 0.0 else "leader_inactive",
                    float(l.get("leader_active_weight", np.nan)),
                    {"leader_opportunity": float(a.get("leader_opportunity_score", np.nan)), "leader_blend": float(a.get("leader_blend", np.nan))},
                    {"leader_active_weight": float(l.get("leader_active_weight", np.nan)), "cash_redeployed": float(l.get("cash_redeployed", np.nan))},
                ),
            }
            for module_name, (branch, strength, inputs, outputs) in module_payloads.items():
                rows.append(
                    {
                        "date": dt,
                        "module_name": module_name,
                        "fold": _fold_for_date(wf, dt),
                        "candidate_id": cid,
                        "universe_id": "base_universe_12",
                        "branch_taken": branch,
                        "threshold_crossed": int(np.isfinite(strength) and abs(strength) > 0.0),
                        "signal_strength": strength,
                        "main_inputs_summary_json": _json_summary(inputs),
                        "main_outputs_summary_json": _json_summary(outputs),
                        "comment_code_or_reason": "snapshot-derived audit trace; candidate multiplier layer changes returns/exposure, while module states come from frozen policy path",
                        "run_id": run_id,
                        "analysis_phase": PHASE,
                        "baseline_reference": BASELINE_REFERENCE,
                        "generated_at": generated_at(),
                    }
                )
    return pd.DataFrame(rows)


def forward_return(series: pd.Series, dt: pd.Timestamp, horizon: int) -> float:
    s = pd.Series(series, dtype=float)
    if dt not in s.index:
        return np.nan
    loc = s.index.get_loc(dt)
    if isinstance(loc, slice):
        loc = loc.start
    end = min(len(s), int(loc) + horizon)
    vals = s.iloc[int(loc) : end]
    if len(vals) == 0:
        return np.nan
    return float(np.prod(1.0 + vals.values) - 1.0)


def build_outcome_cube(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, candidate_objects: Dict[str, Dict[str, Any]], rep_candidates: Sequence[str], run_id: str) -> pd.DataFrame:
    context = build_context(modules, wf, cfg)
    idx = pd.DatetimeIndex(context["primary"]["returns"].index)
    control_r = pd.Series(context["control"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    qqq_r = pd.Series(context["qqq"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    spy_r = pd.Series(context["spy"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    allocator = context["allocator"].reindex(idx).ffill()
    leader = context["leader_diag"].reindex(idx).ffill()
    rows: List[Dict[str, Any]] = []
    for cid in rep_candidates:
        obj = candidate_objects[cid]
        r = pd.Series(obj["returns"], dtype=float).reindex(idx).fillna(0.0)
        turnover = pd.Series(obj["turnover"], dtype=float).reindex(idx).fillna(0.0)
        exposure = pd.Series(obj["exposure"], dtype=float).reindex(idx).fillna(0.0)
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        for dt in idx:
            for horizon in [1, 5, 20]:
                rr = forward_return(r, dt, horizon)
                qret = forward_return(qqq_r, dt, horizon)
                sret = forward_return(spy_r, dt, horizon)
                cret = forward_return(control_r, dt, horizon)
                loc = idx.get_loc(dt)
                if isinstance(loc, slice):
                    loc = loc.start
                end = min(len(idx) - 1, int(loc) + horizon - 1)
                rows.append(
                    {
                        "decision_date": dt,
                        "horizon": horizon,
                        "candidate_id": cid,
                        "fold": _fold_for_date(wf, dt),
                        "universe_id": "base_universe_12",
                        "realized_return": rr,
                        "realized_alpha_vs_qqq": rr - qret if np.isfinite(rr) and np.isfinite(qret) else np.nan,
                        "realized_alpha_vs_spy": rr - sret if np.isfinite(rr) and np.isfinite(sret) else np.nan,
                        "realized_turnover": float(turnover.iloc[int(loc) : end + 1].sum()),
                        "realized_exposure": float(exposure.iloc[int(loc) : end + 1].mean()),
                        "realized_drawdown_change": float(dd.iloc[end] - dd.iloc[int(loc)]),
                        "decision_helped_flag_vs_qqq": int(np.isfinite(rr) and np.isfinite(qret) and rr > qret),
                        "decision_helped_flag_vs_control": int(np.isfinite(rr) and np.isfinite(cret) and rr > cret),
                        "continuation_helped_flag": int(float(allocator.loc[dt].get("continuation_allocator_score", 0.0)) > 0.0 and np.isfinite(rr) and np.isfinite(cret) and rr > cret),
                        "backoff_helped_flag": int(float(allocator.loc[dt].get("risk_backoff_score", 0.0)) > 0.0 and np.isfinite(rr) and np.isfinite(qret) and rr > qret),
                        "leader_helped_flag": int(float(leader.loc[dt].get("leader_active_weight", 0.0)) > 0.0 and np.isfinite(rr) and np.isfinite(cret) and rr > cret),
                        "run_id": run_id,
                        "analysis_phase": PHASE,
                        "baseline_reference": BASELINE_REFERENCE,
                        "generated_at": generated_at(),
                    }
                )
    return pd.DataFrame(rows)


def build_market_context_cube(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, run_id: str) -> pd.DataFrame:
    context = build_context(modules, wf, cfg)
    idx = pd.DatetimeIndex(context["qqq"]["returns"].index)
    qqq_r = pd.Series(context["qqq"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    spy_r = pd.Series(context["spy"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    qqq_eq = (1.0 + qqq_r).cumprod()
    spy_eq = (1.0 + spy_r).cumprod()
    allocator = context["allocator"].reindex(idx).ffill()
    override = _stitch_variant_frame(modules, wf, cfg.primary_variant_key, "override_daily").reindex(idx).ffill()
    rows = []
    for dt in idx:
        a = allocator.loc[dt] if dt in allocator.index else pd.Series(dtype=float)
        o = override.loc[dt] if dt in override.index else pd.Series(dtype=float)
        qvol = float(qqq_r.loc[:dt].tail(63).std() * np.sqrt(252)) if len(qqq_r.loc[:dt].tail(63)) else np.nan
        rows.append(
            {
                "date": dt,
                "qqq_return": float(qqq_r.loc[dt]),
                "qqq_drawdown": float(qqq_eq.loc[dt] / qqq_eq.loc[:dt].max() - 1.0),
                "qqq_vol": qvol,
                "spy_return": float(spy_r.loc[dt]),
                "spy_drawdown": float(spy_eq.loc[dt] / spy_eq.loc[:dt].max() - 1.0),
                "vix": np.nan,
                "avg_corr": float(o.get("avg_corr", np.nan)),
                "breadth": float(a.get("breadth_health_score", np.nan)),
                "benchmark_strength": float(a.get("benchmark_strength_score", np.nan)),
                "benchmark_weakness": float(a.get("benchmark_weakness_score", np.nan)),
                "market_regime_proxy": str(a.get("participation_state", "")),
                "run_id": run_id,
                "analysis_phase": PHASE,
                "candidate_id": "MARKET_CONTEXT",
                "universe_id": "base_universe_12",
                "baseline_reference": BASELINE_REFERENCE,
                "generated_at": generated_at(),
            }
        )
    return pd.DataFrame(rows)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, compression="zstd")


def build_derived_audit_views(paths: Paths, decision: pd.DataFrame, position: pd.DataFrame, trace: pd.DataFrame, outcome: pd.DataFrame) -> None:
    drivers = (
        trace.groupby(["module_name", "branch_taken"], dropna=False)
        .agg(
            observations=("signal_strength", "size"),
            mean_signal_strength=("signal_strength", "mean"),
            threshold_cross_rate=("threshold_crossed", "mean"),
        )
        .reset_index()
        .sort_values(["threshold_cross_rate", "mean_signal_strength"], ascending=False)
    )
    drivers.to_csv(paths.cube / "top_decision_drivers.csv", index=False)
    cont = outcome.groupby(["candidate_id", "horizon"]).agg(continuation_help_rate=("continuation_helped_flag", "mean")).reset_index()
    cont.to_csv(paths.cube / "continuation_activation_audit.csv", index=False)
    structural = trace[trace["module_name"] == "structural_defense_model"].groupby(["candidate_id", "branch_taken"]).agg(count=("date", "size"), avg_signal=("signal_strength", "mean")).reset_index()
    structural.to_csv(paths.cube / "structural_defense_audit.csv", index=False)
    leader = outcome.groupby(["candidate_id", "horizon"]).agg(leader_help_rate=("leader_helped_flag", "mean")).reset_index()
    leader.to_csv(paths.cube / "leader_participation_audit.csv", index=False)
    backoff = outcome.groupby(["candidate_id", "horizon"]).agg(backoff_help_rate=("backoff_helped_flag", "mean")).reset_index()
    backoff.to_csv(paths.cube / "backoff_audit.csv", index=False)
    stop = position.groupby(["candidate_id", "ticker"]).agg(stop_count=("stop_flag", "sum"), selected_days=("selected_flag", "sum"), pnl_contribution=("pnl_contribution", "sum")).reset_index()
    stop.to_csv(paths.cube / "stop_loss_audit.csv", index=False)


def write_cube_docs(paths: Paths, rep_candidates: Sequence[str]) -> None:
    dictionary = [
        "# Audit Cube Dictionary",
        "",
        "## Design decision: representative granular subset",
        "",
        f"The full granular `position_cube` and `module_trace_cube` are limited to {len(rep_candidates)} representative candidates: {', '.join(rep_candidates)}.",
        "",
        "This is intentional. The extended multiplier sweep is an audit sample, not a production portfolio catalogue. Building ticker-date-module traces for every perturbation would make the frontend slower, increase storage, and add little audit value beyond the candidates that define the official point, controlled extremes, and most sensitive directions.",
        "",
        "## Common metadata",
        "",
        "All cube tables include `run_id`, `analysis_phase`, `candidate_id`, `universe_id`, `baseline_reference`, and `generated_at`.",
        "",
        "## decision_date_cube.parquet",
        "",
        "One row per date/fold/candidate/universe. It records allocator state, participation state, gate/vol/exp controls, continuation/backoff/leader signals, expected exposure and turnover.",
        "",
        "## position_cube.parquet",
        "",
        "One row per date/ticker/fold/candidate/universe for representative candidates. It records scores, ranks, weights, stop flags, forward returns, and PnL contribution. Some component-level raw fields are nullable when the frozen snapshot exposes only composite scores at that layer.",
        "",
        "## module_trace_cube.parquet",
        "",
        "One row per date/module/candidate/fold/universe. JSON summaries capture structured main inputs/outputs for audit without forcing a brittle schema for every module internals.",
        "",
        "## outcome_cube.parquet",
        "",
        "One row per decision date/horizon/candidate/fold/universe. `decision_helped_flag_vs_qqq` equals 1 when realized system return over the horizon exceeds QQQ. `decision_helped_flag_vs_control` equals 1 when realized system return exceeds the official historical control.",
        "",
        "## market_context_cube.parquet",
        "",
        "One row per date with QQQ/SPY returns, drawdowns, realized vol, breadth and benchmark strength/weakness proxies from the frozen allocator path.",
        "",
        "## Limitations",
        "",
        "- Cubes inherit the granularity exposed by the frozen 14.3R snapshot.",
        "- Candidate perturbations alter returns/exposure through the frozen multiplier layer; module-state traces remain anchored to the frozen policy path.",
        "- Nullable fields are explicit rather than imputed when the source module does not expose a stable primitive.",
    ]
    (paths.cube / "cube_dictionary.md").write_text("\n".join(dictionary), encoding="utf-8")
    lineage = [
        "# Cube Lineage",
        "",
        "- Source code snapshot: `research/mahoraga14_3_extended_analysis/source_snapshot/` copied from the frozen official baseline source.",
        "- Walk-forward cache: `outputs/cache/wf_base_universe_12.joblib`.",
        "- Candidate objects: generated by applying frozen multiplier logic to the base stitched context.",
        "- Decision fields: allocator daily state, override daily state, candidate returns/exposure/turnover.",
        "- Position fields: fold-level primary weights, base weights, score table, stop-hit table, and next-period returns.",
        "- Module trace fields: structured summaries derived from allocator, override and leader diagnostic tables.",
        "- Outcome fields: forward realized returns over 1, 5 and 20 trading-day horizons.",
        "",
        "No official baseline files are modified by this lineage. Official source is read or copied once as a research snapshot.",
    ]
    (paths.cube / "cube_lineage.md").write_text("\n".join(lineage), encoding="utf-8")


def run_cube_analysis(paths: Paths, modules: Dict[str, Any], wf: Dict[str, Any], cfg: Any, mult: Dict[str, Any], run_id: str) -> Dict[str, pd.DataFrame]:
    reps = candidate_for_rep_cube(mult)
    objects = mult["objects"]
    missing = [cid for cid in reps if cid not in objects]
    if missing:
        raise RuntimeError(f"Representative candidates missing from candidate object cache: {missing}")
    decision = build_decision_cube(paths, modules, wf, cfg, objects, reps, run_id)
    position = build_position_cube(paths, modules, wf, cfg, objects, reps, run_id)
    trace = build_module_trace_cube(paths, modules, wf, cfg, reps, run_id)
    outcome = build_outcome_cube(paths, modules, wf, cfg, objects, reps, run_id)
    market = build_market_context_cube(paths, modules, wf, cfg, run_id)

    write_parquet(decision, paths.cube / "decision_date_cube.parquet")
    write_parquet(position, paths.cube / "position_cube.parquet")
    write_parquet(trace, paths.cube / "module_trace_cube.parquet")
    write_parquet(outcome, paths.cube / "outcome_cube.parquet")
    write_parquet(market, paths.cube / "market_context_cube.parquet")
    build_derived_audit_views(paths, decision, position, trace, outcome)
    write_cube_docs(paths, reps)
    (paths.cube / "representative_candidates.json").write_text(json.dumps(reps, indent=2), encoding="utf-8")
    return {"decision": decision, "position": position, "trace": trace, "outcome": outcome, "market": market}


def write_final_reports(
    paths: Paths,
    run_id: str,
    timings: Dict[str, float],
    cache_notes: List[str],
    mult: Dict[str, Any],
    universe: Dict[str, Any],
    cubes: Dict[str, pd.DataFrame],
    skipped: List[str],
) -> None:
    summary = mult["summary"]
    official = summary[summary["CandidateId"] == OFFICIAL_CANDIDATE_ID].iloc[0]
    sens = mult["sensitivity"]
    dtd = "not observed" if math.isinf(mult["distance_to_decay"]) else f"{mult['distance_to_decay']:.4f}"
    successful_universes = universe["summary"][universe["summary"].get("run_status", "OK") == "OK"] if len(universe["summary"]) else pd.DataFrame()
    official_universes = successful_universes[successful_universes["CandidateId"] == OFFICIAL_CANDIDATE_ID] if len(successful_universes) else pd.DataFrame()
    report = [
        "# Final Extended Analysis Report",
        "",
        f"- run_id: `{run_id}`",
        f"- baseline reference: `{BASELINE_REFERENCE}`",
        "",
        "## Objective",
        "",
        "This research phase audits the frozen Mahoraga14_3 baseline for extended multiplier robustness, universe dependence, and granular decision traceability. It does not define a new baseline and does not reoptimize the official candidate.",
        "",
        "## Theoretical foundation",
        "",
        "The robustness analysis treats parameter perturbation as local stability testing rather than search. The core degradation checks combine risk-adjusted return, growth, drawdown control, benchmark-adjusted alpha and fold-local damage because financial backtests can look strong in stitched aggregate while hiding path-local instability.",
        "",
        "Newey-West alpha is retained because daily strategy returns can exhibit serial correlation and heteroskedasticity. Fold degradation is explicitly measured because walk-forward validation is the relevant out-of-sample unit for this architecture.",
        "",
        "## Computational design",
        "",
        "- Baseline walk-forward is run or loaded inside this research phase.",
        "- One-dimensional sweeps identify sensitive axes.",
        "- Only the two most sensitive axes receive a two-dimensional sweep.",
        "- Controlled extremes test boundary behavior without becoming optimization candidates.",
        "- Full granular cubes are limited to representative candidates for auditability and usability.",
        "",
        "## Multiplier robustness",
        "",
        f"- Official CAGR: {float(official['CAGR']):.4f}%",
        f"- Official Sharpe: {float(official['Sharpe']):.4f}",
        f"- Official Sortino: {float(official['Sortino']):.4f}",
        f"- Official MaxDD: {float(official['MaxDD']):.4f}%",
        f"- distance_to_decay: {dtd}",
        f"- robust_region_share_extended: {float(mult['robust_share']):.2%}",
        f"- sampled candidates: {len(summary)}",
        "",
        "## Sensitivity ranking",
        "",
        md_table(sens) if len(sens) else "No sensitivity ranking available.",
        "",
        "## Plateau radius",
        "",
        md_table(mult["plateau_df"]),
        "",
        "## Worst-fold degradation",
        "",
        md_table(summary.sort_values("worst_fold_sharpe_delta_vs_official").head(10)[
            [
                "CandidateId",
                "sweep_role",
                "worst_fold_sharpe_delta_vs_official",
                "worst_fold_cagr_delta_vs_official",
                "max_fold_maxdd_worsening_vs_official",
                "severe_fold_damage_count",
            ]
        ]),
        "",
        "## Universe robustness",
        "",
        md_table(official_universes[
            ["universe_id", "CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "usable_count", "run_status"]
        ])
        if len(official_universes)
        else "No successful official universe rows available.",
        "",
        "## Limitations",
        "",
        "- Extended samples do not prove global parameter stability outside sampled ranges.",
        "- Alternate universes mix economic composition with data coverage and canonical schedule effects.",
        "- The decision audit cube exposes the stable fields available in the frozen snapshot; nullable fields indicate unavailable primitives rather than inferred values.",
        "- Candidate perturbation is applied through the frozen multiplier layer, so granular module traces remain anchored to the official policy path.",
        "",
        "## Open risks",
        "",
        "- Universe robustness can be sensitive to ticker seasoning and corporate history.",
        "- Large controlled extremes may test unrealistic operating regimes.",
        "- Future data could move both fold-local and universe-local conclusions.",
        "",
        "## Conclusion",
        "",
        "The generated outputs should be read as an independent audit layer: they clarify where the official point remains stable, where degradation first appears, and which decisions can be reconstructed date by date. They do not alter the official freeze.",
    ]
    (paths.reports / "final_extended_analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    implementation = [
        "# Implementation Report",
        "",
        f"- run_id: `{run_id}`",
        "",
        "## Structure created",
        "",
        "- `README.md`, `requirements_extended.txt`, `run_extended_analysis.py`, `run_api.py`",
        "- `src/extended_analysis/` analysis package",
        "- `api/` FastAPI app",
        "- `frontend/` React/TypeScript/Tailwind app",
        "- `outputs/extended_multiplier_robustness/`",
        "- `outputs/universe_robustness/`",
        "- `outputs/audit_cube/`",
        "- `outputs/figures/`",
        "- `outputs/reports/`",
        "- `source_snapshot/` copied from frozen baseline source",
        "",
        "## What executed",
        "",
        f"- Estimated candidate evaluations before run: one-dimensional {len(one_dimensional_specs())}, two-dimensional selected after sensitivity, extremes {len(extreme_specs())}.",
        f"- Actual multiplier candidate rows: {len(summary)}",
        f"- Universe rows: {len(universe['summary'])}",
        f"- Decision cube rows: {len(cubes['decision'])}",
        f"- Position cube rows: {len(cubes['position'])}",
        f"- Module trace cube rows: {len(cubes['trace'])}",
        f"- Outcome cube rows: {len(cubes['outcome'])}",
        f"- Market context rows: {len(cubes['market'])}",
        "",
        "## Timings",
        "",
        md_table(pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items()])),
        "",
        "## Cache and fallback",
        "",
        "\n".join(f"- {note}" for note in cache_notes) if cache_notes else "- No cache notes recorded.",
        "",
        "## Skipped or limited",
        "",
        "\n".join(f"- {item}" for item in skipped) if skipped else "- No major steps skipped.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "cd D:\\QuantMahoraga",
        "python .\\research\\mahoraga14_3_extended_analysis\\run_extended_analysis.py",
        "python .\\research\\mahoraga14_3_extended_analysis\\run_api.py",
        "```",
        "",
        "Frontend:",
        "",
        "```powershell",
        "cd D:\\QuantMahoraga\\research\\mahoraga14_3_extended_analysis\\frontend",
        "npm install",
        "npm run dev",
        "```",
        "",
        "## Baseline safety",
        "",
        "The implementation writes only inside `research/mahoraga14_3_extended_analysis`. The official `baseline/` package is treated as read-only.",
    ]
    (paths.reports / "implementation_report.md").write_text("\n".join(implementation), encoding="utf-8")


def write_manifest(paths: Paths) -> None:
    rows = []
    for file in sorted(paths.root.rglob("*")):
        if file.is_file():
            rel = file.relative_to(paths.root)
            rel_parts = set(rel.parts)
            if (
                "node_modules" in rel_parts
                or "__pycache__" in rel_parts
                or (len(rel.parts) >= 2 and rel.parts[0] == "outputs" and rel.parts[1] == "cache")
                or (len(rel.parts) >= 2 and rel.parts[0] == "frontend" and rel.parts[1] == "dist")
            ):
                continue
            rows.append(
                {
                    "relative_path": str(rel).replace("\\", "/"),
                    "size_bytes": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(rows).to_csv(paths.manifests / "file_manifest.csv", index=False)


def run_all(force: bool = False, skip_universes: bool = False, max_new_universe_runs: int = 1) -> Dict[str, Any]:
    paths = get_paths()
    ensure_dirs(paths)
    copied_cache = seed_research_cache(paths)
    modules = import_snapshot_modules(paths)
    run_id = stable_run_id()
    timings: Dict[str, float] = {}
    cache_notes = [f"seeded research data cache: {x}" for x in copied_cache]
    skipped: List[str] = []

    t0 = time.perf_counter()
    wf, cfg, _ohlcv, _schedule, _snaps, wf_cached, wf_seconds = load_or_run_wf(paths, modules, "base_universe_12", UNIVERSES["base_universe_12"], force=force)
    timings["base_walk_forward"] = wf_seconds
    cache_notes.append("base walk-forward loaded from research cache" if wf_cached else "base walk-forward recalculated inside research and cached")

    t1 = time.perf_counter()
    mult = run_multiplier_analysis(paths, modules, wf, cfg, run_id, force=force)
    timings["extended_multiplier_robustness"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    if skip_universes:
        universe = {"coverage": pd.DataFrame(), "summary": pd.DataFrame(), "run_meta": []}
        skipped.append("universe robustness skipped by command-line flag")
    else:
        universe = run_universe_analysis(paths, modules, run_id, force=force, max_new_universe_runs=max_new_universe_runs)
        aborted = [m["universe_id"] for m in universe["run_meta"] if str(m.get("status")) == "aborted_compute_budget"]
        if aborted:
            skipped.append(
                "alternate universe walk-forward aborted by compute budget after materializing coverage: "
                + ", ".join(aborted)
            )
    timings["universe_robustness"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    cubes = run_cube_analysis(paths, modules, wf, cfg, mult, run_id)
    timings["audit_cube"] = time.perf_counter() - t3
    timings["total"] = time.perf_counter() - t0

    write_final_reports(paths, run_id, timings, cache_notes, mult, universe, cubes, skipped)
    write_manifest(paths)
    return {
        "run_id": run_id,
        "paths": paths,
        "timings": timings,
        "multiplier": mult,
        "universe": universe,
        "cubes": cubes,
        "skipped": skipped,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent Mahoraga 14.3 extended analysis.")
    parser.add_argument("--force", action="store_true", help="Ignore research caches and recompute.")
    parser.add_argument("--skip-universes", action="store_true", help="Skip alternate universe runs.")
    parser.add_argument(
        "--max-new-universe-runs",
        type=int,
        default=1,
        help="Maximum uncached alternate-universe walk-forwards to run before marking remaining universes as compute-budget aborted.",
    )
    args = parser.parse_args(argv)
    try:
        result = run_all(force=args.force, skip_universes=args.skip_universes, max_new_universe_runs=max(0, args.max_new_universe_runs))
    except Exception:
        paths = get_paths()
        ensure_dirs(paths)
        (paths.reports / "last_run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(json.dumps({"run_id": result["run_id"], "timings": result["timings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
