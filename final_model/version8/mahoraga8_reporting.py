from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    import mahoraga6_1 as m6
except Exception:
    import mahoraga6_1 as m6  # type: ignore

from mahoraga8_config import Mahoraga8Config


def build_fold_summary(wf: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([r['fold_row'] for r in wf['results']]).sort_values('fold')


def build_regime_comparison(wf: Dict[str, Any], cfg: Mahoraga8Config) -> pd.DataFrame:
    rows = []
    for reg in ['NORMAL', 'STRESS', 'PANIC', 'RECOVERY']:
        base_all, ov_all = [], []
        for r in wf['results']:
            base_bt = r['base_bt']
            ov_bt = r['ov_bt']
            pol = r['policy_table']['active_regime_state'].reindex(ov_bt['returns_net'].index).ffill().fillna('NORMAL')
            mask = pol == reg
            base_all.append(base_bt['returns_net'].reindex(mask.index).fillna(0.0).loc[mask])
            ov_all.append(ov_bt['returns_net'].reindex(mask.index).fillna(0.0).loc[mask])
        rb = pd.concat(base_all).sort_index() if base_all else pd.Series(dtype=float)
        ro = pd.concat(ov_all).sort_index() if ov_all else pd.Series(dtype=float)
        if len(rb) == 0:
            continue
        base_eq = cfg.capital_initial * (1.0 + rb).cumprod()
        ov_eq = cfg.capital_initial * (1.0 + ro).cumprod()
        sb = m6.summarize(rb, base_eq, pd.Series(1.0, index=rb.index), pd.Series(0.0, index=rb.index), cfg, f'BASE_{reg}')
        so = m6.summarize(ro, ov_eq, pd.Series(1.0, index=ro.index), pd.Series(0.0, index=ro.index), cfg, f'H8_{reg}')
        rows.append({'Regime': reg, 'Days': len(rb), 'BASE_CAGR%': round(sb['CAGR'] * 100, 2), 'BASE_Sharpe': round(sb['Sharpe'], 4), 'BASE_MaxDD%': round(sb['MaxDD'] * 100, 2), 'H8_CAGR%': round(so['CAGR'] * 100, 2), 'H8_Sharpe': round(so['Sharpe'], 4), 'H8_MaxDD%': round(so['MaxDD'] * 100, 2), 'DeltaSharpe': round(so['Sharpe'] - sb['Sharpe'], 4)})
    return pd.DataFrame(rows)


def selection_audit_h8(wf: Dict[str, Any]) -> pd.DataFrame:
    fold_df = build_fold_summary(wf)
    return pd.DataFrame([
        {'Method': 'BASELINE_6_1_FROZEN', 'MeanSharpe': round(float(fold_df['BASE_Sharpe'].mean()), 5), 'MeanCAGR%': round(float(fold_df['BASE_CAGR%'].mean()), 3), 'MeanMaxDD%': round(float(fold_df['BASE_MaxDD%'].mean()), 3)},
        {'Method': 'H8', 'MeanSharpe': round(float(fold_df['H8_Sharpe'].mean()), 5), 'MeanCAGR%': round(float(fold_df['H8_CAGR%'].mean()), 3), 'MeanMaxDD%': round(float(fold_df['H8_MaxDD%'].mean()), 3), 'MeanRiskBudget': round(float(fold_df['MeanRiskBudget'].mean()), 4), 'MeanInterventionRate': round(float(fold_df['InterventionRate'].mean()), 4)},
    ])


def _oos_comparison_text(cfg: Mahoraga8Config, wf: Dict[str, Any]) -> str:
    base_r = wf['stitched_base']
    h8_r = wf['stitched_h8']
    if len(base_r) == 0:
        return 'No OOS results.'
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    h8_eq = cfg.capital_initial * (1.0 + h8_r).cumprod()
    sb = m6.summarize(base_r, base_eq, pd.Series(1.0, index=base_r.index), pd.Series(0.0, index=base_r.index), cfg, 'BASE_OOS')
    so = m6.summarize(h8_r, h8_eq, pd.Series(1.0, index=h8_r.index), pd.Series(0.0, index=h8_r.index), cfg, 'H8_OOS')
    return f"OOS COMPARISON\n  BASELINE CAGR={sb['CAGR']*100:.2f}%  Sharpe={sb['Sharpe']:.3f}  MaxDD={sb['MaxDD']*100:.2f}%  CVaR5={sb['CVaR_5']*100:.2f}%\n  H8       CAGR={so['CAGR']*100:.2f}%  Sharpe={so['Sharpe']:.3f}  MaxDD={so['MaxDD']*100:.2f}%  CVaR5={so['CVaR_5']*100:.2f}%\n"


def final_report_text_h8(cfg: Mahoraga8Config, wf: Dict[str, Any]) -> str:
    fold_df = build_fold_summary(wf)
    regime_df = build_regime_comparison(wf, cfg)
    sel_df = selection_audit_h8(wf)
    text = []
    text.append('MAHORAGA 8 — FINAL REPORT')
    text.append('=' * 78)
    text.append('')
    text.append(_oos_comparison_text(cfg, wf))
    text.append('SELECTION AUDIT')
    text.append(sel_df.to_string(index=False))
    text.append('')
    text.append('FOLD SUMMARY')
    text.append(fold_df.to_string(index=False))
    if len(regime_df):
        text.append('')
        text.append('REGIME COMPARISON')
        text.append(regime_df.to_string(index=False))
    return '\n'.join(text) + '\n'


def save_outputs_h8(cfg: Mahoraga8Config, wf: Dict[str, Any]) -> None:
    d = cfg.outputs_dir
    build_fold_summary(wf).to_csv(f'{d}/walk_forward_folds_8.csv', index=False)
    selection_audit_h8(wf).to_csv(f'{d}/selection_audit.csv', index=False)
    reg = build_regime_comparison(wf, cfg)
    if len(reg):
        reg.to_csv(f'{d}/regime_comparison.csv', index=False)
    pd.DataFrame([r['best_params'] for r in wf['results']]).to_csv(f'{d}/best_params_by_fold.csv', index=False)
    with open(f'{d}/final_report.txt', 'w', encoding='utf-8') as f:
        f.write(final_report_text_h8(cfg, wf))
