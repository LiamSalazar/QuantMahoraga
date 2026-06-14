import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasScatter } from "../utils/chartGuards";
import { asNumber, formatMetric } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID, formatDemoMode } from "../utils/labels";
import { rowsFrom, topRows } from "../utils/rows";

function rangeValues(options: Options | null, key: string, fallback: number[]) {
  return (options?.slider_ranges?.[key]?.values ?? fallback).map((value) => Number(value));
}

function complete(row: Row) {
  return asNumber(row.cagr ?? row.CAGR) !== null && asNumber(row.sharpe ?? row.Sharpe) !== null && asNumber(row.maxdd ?? row.MaxDD) !== null;
}

export default function WhatIfLab({ options }: { options: Options | null }) {
  const [tab, setTab] = useState<"observed" | "simulated">("observed");
  const [fold, setFold] = useState("1");
  const [horizon, setHorizon] = useState("20");
  const [budget, setBudget] = useState("1.05");
  const [conviction, setConviction] = useState("1.1");
  const [leader, setLeader] = useState("1.1");
  const [backoff, setBackoff] = useState("1.05");
  const [cost, setCost] = useState("5");
  const [slippage, setSlippage] = useState("2");
  const [applied, setApplied] = useState(0);
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const grid = useApiResource<Record<string, unknown>>("/whatif/grid", { candidate_id: OFFICIAL_CANDIDATE_ID, fold, universe_id: options?.default_universe ?? "base_universe_12", horizon, cost_bps: cost, slippage_bps: slippage, limit: 1000, applied }, tab === "simulated");
  if (((grid.loading && tab === "simulated") || extended.loading) && !extended.data) return <LoadingState label="Loading what-if grid" />;
  if (grid.error) return <ErrorState error={grid.error} retry={grid.retry} />;
  if (extended.error) return <ErrorState error={extended.error} retry={extended.retry} />;

  const simulated = rowsFrom(grid.data).filter(complete);
  const observed = rowsFrom(extended.data, "extended_multiplier_summary").filter((row) => row.demo_mode !== true && complete(row));
  const pareto = rowsFrom(grid.data, "pareto").filter(complete);
  const selected = simulated
    .map((row) => ({
      row,
      dist: Math.abs(Number(row.budget_multiplier) - Number(budget)) + Math.abs(Number(row.conviction_multiplier) - Number(conviction)) + Math.abs(Number(row.leader_multiplier) - Number(leader)) + Math.abs(Number(row.backoff_strength) - Number(backoff)),
    }))
    .sort((a, b) => a.dist - b.dist)[0];
  const officialSharpe = asNumber(observed.find((row) => row.candidate_id === OFFICIAL_CANDIDATE_ID || row.CandidateId === OFFICIAL_CANDIDATE_ID)?.Sharpe);
  const selectedSharpe = asNumber(selected?.row.sharpe);
  const insight = selectedSharpe !== null && officialSharpe !== null ? `Selected scenario is ${selectedSharpe < officialSharpe ? "below" : "above"} the official Sharpe reference in this what-if slice.` : null;
  const activeRows = tab === "observed" ? observed : simulated;

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="What-if & Stress" question="Observed audit rows are separate from simulated what-if scenarios." source="extended audit + fact_whatif" />
        <div className="chips">
          <button className={tab === "observed" ? "primary-button" : "ghost-button"} onClick={() => setTab("observed")}>Observed/audited scenarios</button>
          <button className={tab === "simulated" ? "primary-button" : "ghost-button"} onClick={() => setTab("simulated")}>Simulated what-if</button>
          <span>{tab === "simulated" ? "Simulated what-if · not official performance" : "Observed audit scenario"}</span>
        </div>
      </section>

      {tab === "simulated" ? (
        <section className="panel span-12">
          <SectionHeader title="Simulated Scenario Builder" question="Guided discrete controls only; apply button avoids slider request storms." source="fact_whatif demo_mode=true" />
          <div className="lab-controls">
            <SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [1, 2, 3, 4, 5])]} onChange={setFold} compact />
            <SelectControl label="Horizon" value={horizon} options={options?.horizons ?? [1, 5, 20, 60]} onChange={setHorizon} compact />
            <SelectControl label="Budget" value={budget} options={rangeValues(options, "budget_multiplier", [0.9, 0.95, 1, 1.05, 1.1, 1.15])} onChange={setBudget} compact />
            <SelectControl label="Conviction" value={conviction} options={rangeValues(options, "conviction_multiplier", [0.9, 1, 1.1, 1.2, 1.3])} onChange={setConviction} compact />
            <SelectControl label="Leader" value={leader} options={rangeValues(options, "leader_multiplier", [0.9, 1, 1.1, 1.2, 1.3])} onChange={setLeader} compact />
            <SelectControl label="Backoff" value={backoff} options={rangeValues(options, "backoff_strength", [0.9, 1, 1.05, 1.1, 1.2])} onChange={setBackoff} compact />
            <SelectControl label="Cost bps" value={cost} options={rangeValues(options, "cost_bps", [0, 5, 10, 25])} onChange={setCost} compact />
            <SelectControl label="Slippage bps" value={slippage} options={rangeValues(options, "slippage_bps", [0, 2, 5, 10])} onChange={setSlippage} compact />
            <button className="primary-button" onClick={() => setApplied((value) => value + 1)} disabled={!grid.data && grid.loading}>Apply scenario</button>
          </div>
          <div className="metric-grid">
            <MetricCard label="Nearest scenario distance" value={selected ? selected.dist.toFixed(3) : "—"} detail="selected controls to available row" />
            <MetricCard label="Nearest CAGR" value={formatMetric(selected?.row.cagr, "CAGR")} detail={formatDemoMode(selected?.row.demo_mode)} />
            <MetricCard label="Nearest Sharpe" value={formatMetric(selected?.row.sharpe, "Sharpe")} detail={insight ?? "ranked by robust score"} />
            <MetricCard label="Valid rows" value={String(simulated.length)} detail="current what-if slice" />
          </div>
        </section>
      ) : null}

      <ChartPanel title={tab === "observed" ? "Observed Scenario Ranking" : "Simulated Pareto Frontier"} question={tab === "observed" ? "Which audited scenarios were strongest without simulation?" : "What return/drawdown trade-off emerges under selected frictions?"} source={tab === "observed" ? "extended_multiplier_summary.csv" : "fact_whatif"} ready={tab === "observed" ? observed.length >= 4 : hasScatter(pareto, "maxdd", "cagr", 4)}>
        {tab === "observed" ? (
          <ResponsiveContainer width="100%" height="100%"><BarChart data={topRows(observed, "Sharpe", 12)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="candidate_id" hide label={axisLabel("Candidate")} /><YAxis label={axisLabel("Sharpe", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="Sharpe" fill="#72f0b1" /></BarChart></ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="maxdd" type="number" label={axisLabel("Max drawdown")} /><YAxis dataKey="cagr" type="number" label={axisLabel("CAGR", true)} /><ZAxis dataKey="sharpe" range={[50, 220]} /><Tooltip content={<ChartTooltip />} /><Scatter data={pareto} fill="#80d8ff" /></ScatterChart></ResponsiveContainer>
        )}
      </ChartPanel>

      <section className="panel span-12">
        <SectionHeader title={tab === "observed" ? "Observed / Audited Scenario Ranking" : "Simulated What-if Ranking"} question={tab === "observed" ? "These are observed audit rows, not simulated what-if rows." : "Rows are explicitly flagged and must not be treated as official performance."} source={tab === "observed" ? "extended_multiplier_summary.csv" : "fact_whatif"} />
        <DataTable rows={activeRows} columns={tab === "observed" ? ["candidate_id", "sweep_role", "CAGR", "Sharpe", "MaxDD", "robust_region_flag", "severe_fold_damage_count"] : ["candidate_id", "fold", "horizon", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cost_bps", "slippage_bps", "cagr", "sharpe", "maxdd", "demo_mode"]} />
      </section>
    </div>
  );
}
