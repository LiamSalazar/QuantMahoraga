import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { SliderControl, SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasScatter } from "../utils/chartGuards";
import { formatMetric } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID, formatDemoMode } from "../utils/labels";
import { rowsFrom, topRows } from "../utils/rows";

export default function WhatIfLab({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("1");
  const [horizon, setHorizon] = useState("20");
  const [budget, setBudget] = useState(1.05);
  const [conviction, setConviction] = useState(1.1);
  const [leader, setLeader] = useState(1.1);
  const [backoff, setBackoff] = useState(1.05);
  const [cost, setCost] = useState(5);
  const [slippage, setSlippage] = useState(2);
  const [applied, setApplied] = useState(0);
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const grid = useApiResource<Record<string, unknown>>("/whatif/grid", { candidate_id: OFFICIAL_CANDIDATE_ID, fold, universe_id: options?.default_universe ?? "base_universe_12", horizon, cost_bps: cost, slippage_bps: slippage, limit: 1000, applied });
  if ((grid.loading || extended.loading) && !grid.data) return <LoadingState label="Loading what-if grid" />;
  if (grid.error) return <ErrorState error={grid.error} retry={grid.retry} />;
  if (extended.error) return <ErrorState error={extended.error} retry={extended.retry} />;
  const simulated = rowsFrom(grid.data);
  const observed = rowsFrom(extended.data, "extended_multiplier_summary").filter((row) => row.demo_mode !== true);
  const pareto = rowsFrom(grid.data, "pareto");
  const selected = simulated
    .map((row) => ({ row, dist: Math.abs(Number(row.budget_multiplier) - budget) + Math.abs(Number(row.conviction_multiplier) - conviction) + Math.abs(Number(row.leader_multiplier) - leader) + Math.abs(Number(row.backoff_strength) - backoff) }))
    .sort((a, b) => a.dist - b.dist)[0];

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="What-if & Stress" question="Observed audit rows are separate from simulated what-if scenarios." source="extended audit + fact_whatif" />
        <div className="chips"><span>Observed/audited scenarios</span><span>Simulated what-if · not official performance</span></div>
      </section>
      <section className="panel span-12">
        <SectionHeader title="Simulated Scenario Builder" question="Guided controls only; apply button avoids slider request storms." source="fact_whatif demo_mode=true" />
        <div className="lab-controls">
          <SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [1, 2, 3, 4, 5])]} onChange={setFold} compact />
          <SelectControl label="Horizon" value={horizon} options={options?.horizons ?? [1, 5, 20, 60]} onChange={setHorizon} compact />
          <SliderControl label="Budget" value={budget} min={0.9} max={1.15} step={0.05} onChange={setBudget} />
          <SliderControl label="Conviction" value={conviction} min={0.9} max={1.3} step={0.05} onChange={setConviction} />
          <SliderControl label="Leader" value={leader} min={0.9} max={1.3} step={0.05} onChange={setLeader} />
          <SliderControl label="Backoff" value={backoff} min={0.9} max={1.2} step={0.05} onChange={setBackoff} />
          <SliderControl label="Cost bps" value={cost} min={0} max={25} step={1} onChange={setCost} />
          <SliderControl label="Slippage bps" value={slippage} min={0} max={10} step={1} onChange={setSlippage} />
          <button className="primary-button" onClick={() => setApplied((value) => value + 1)}>Apply scenario</button>
        </div>
        <div className="metric-grid">
          <MetricCard label="Nearest distance" value={selected ? selected.dist.toFixed(3) : "n/a"} detail="selected sliders to available row" />
          <MetricCard label="Nearest CAGR" value={formatMetric(selected?.row.cagr, "CAGR")} detail={formatDemoMode(selected?.row.demo_mode)} />
          <MetricCard label="Nearest Sharpe" value={formatMetric(selected?.row.sharpe, "Sharpe")} detail="ranked by robust score" />
          <MetricCard label="Rows" value={String(simulated.length)} detail="current what-if slice" />
        </div>
      </section>
      <ChartPanel title="Simulated Pareto Frontier" question="What return/drawdown trade-off emerges under selected frictions?" source="fact_whatif" ready={hasScatter(pareto, "maxdd", "cagr", 4)}>
        <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="maxdd" type="number" /><YAxis dataKey="cagr" type="number" /><ZAxis dataKey="sharpe" range={[50, 220]} /><Tooltip /><Scatter data={pareto} fill="#80d8ff" /></ScatterChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Observed Scenario Ranking" question="Which audited scenarios were strongest without simulation?" source="extended_multiplier_summary.csv" ready={observed.length >= 4}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={topRows(observed, "Sharpe", 12)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="candidate_id" hide /><YAxis /><Tooltip /><Bar dataKey="Sharpe" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-12"><SectionHeader title="Observed / Audited Scenarios" question="These are not simulated what-if rows." source="extended_multiplier_summary.csv" /><DataTable rows={observed} columns={["candidate_id", "sweep_role", "CAGR", "Sharpe", "MaxDD", "robust_region_flag", "severe_fold_damage_count"]} /></section>
      <section className="panel span-12"><SectionHeader title="Simulated What-if Ranking" question="Rows are explicitly flagged and must not be treated as official performance." source="fact_whatif" /><DataTable rows={simulated as Row[]} columns={["candidate_id", "fold", "horizon", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cost_bps", "slippage_bps", "cagr", "sharpe", "maxdd", "demo_mode"]} /></section>
    </div>
  );
}
