import { useEffect, useMemo, useState } from "react";
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

type ScenarioControls = {
  fold: string;
  horizon: string;
  budget: string;
  conviction: string;
  leader: string;
  backoff: string;
  cost: string;
  slippage: string;
};

function rangeValues(options: Options | null, key: string, fallback: number[]) {
  const values = (options?.slider_ranges?.[key]?.values ?? fallback).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  return values.length ? values : fallback;
}

function complete(row: Row) {
  return asNumber(row.cagr ?? row.CAGR) !== null && asNumber(row.sharpe ?? row.Sharpe) !== null && asNumber(row.maxdd ?? row.MaxDD) !== null;
}

function sameControls(a: ScenarioControls, b: ScenarioControls) {
  return Object.keys(a).every((key) => a[key as keyof ScenarioControls] === b[key as keyof ScenarioControls]);
}

function nearestOption(value: string, options: (string | number)[]) {
  const valid = options.map((option) => String(option));
  if (!valid.length) return value;
  if (valid.includes(value)) return value;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return valid
      .map((option) => ({ option, distance: Math.abs(Number(option) - numeric) }))
      .filter((item) => Number.isFinite(item.distance))
      .sort((a, b) => a.distance - b.distance)[0]?.option ?? valid[0];
  }
  return valid[0];
}

function scenarioDistance(row: Row, controls: ScenarioControls) {
  const values = [
    [row.budget_multiplier, controls.budget],
    [row.conviction_multiplier, controls.conviction],
    [row.leader_multiplier, controls.leader],
    [row.backoff_strength, controls.backoff],
  ];
  const distances = values.map(([left, right]) => {
    const a = asNumber(left);
    const b = asNumber(right);
    return a === null || b === null ? null : Math.abs(a - b);
  });
  if (distances.some((value) => value === null)) return null;
  return (distances as number[]).reduce((sum, value) => sum + value, 0);
}

export default function WhatIfLab({ options }: { options: Options | null }) {
  const [tab, setTab] = useState<"observed" | "simulated">("observed");
  const [draft, setDraft] = useState<ScenarioControls>({ fold: "1", horizon: "20", budget: "1.05", conviction: "1.1", leader: "1.1", backoff: "1.05", cost: "5", slippage: "2" });
  const [appliedControls, setAppliedControls] = useState<ScenarioControls>(draft);
  const [appliedNonce, setAppliedNonce] = useState(0);
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const grid = useApiResource<Record<string, unknown>>(
    "/whatif/grid",
    { candidate_id: OFFICIAL_CANDIDATE_ID, fold: appliedControls.fold, universe_id: options?.default_universe ?? "base_universe_12", horizon: appliedControls.horizon, cost_bps: appliedControls.cost, slippage_bps: appliedControls.slippage, limit: 1000, appliedNonce },
    tab === "simulated",
  );

  const foldOptions = useMemo(() => ["all", ...(options?.folds ?? [1, 2, 3, 4, 5])], [options?.folds]);
  const horizonOptions = useMemo(() => options?.horizons ?? [1, 5, 20, 60], [options?.horizons]);
  const budgetOptions = useMemo(() => rangeValues(options, "budget_multiplier", [0.9, 0.95, 1, 1.05, 1.1, 1.15]), [options]);
  const convictionOptions = useMemo(() => rangeValues(options, "conviction_multiplier", [0.9, 1, 1.1, 1.2, 1.3]), [options]);
  const leaderOptions = useMemo(() => rangeValues(options, "leader_multiplier", [0.9, 1, 1.1, 1.2, 1.3]), [options]);
  const backoffOptions = useMemo(() => rangeValues(options, "backoff_strength", [0.9, 1, 1.05, 1.1, 1.2]), [options]);
  const costOptions = useMemo(() => rangeValues(options, "cost_bps", [0, 5, 10, 25]), [options]);
  const slippageOptions = useMemo(() => rangeValues(options, "slippage_bps", [0, 2, 5, 10]), [options]);
  const simulated = rowsFrom(grid.data).filter(complete);
  const observed = rowsFrom(extended.data, "extended_multiplier_summary").filter((row) => row.demo_mode !== true && complete(row));
  const pareto = rowsFrom(grid.data, "pareto").filter(complete);
  const selected = useMemo(
    () =>
      simulated
        .map((row) => ({ row, dist: scenarioDistance(row, appliedControls) }))
        .filter((item): item is { row: Row; dist: number } => item.dist !== null)
        .sort((a, b) => a.dist - b.dist)[0],
    [simulated, appliedControls],
  );
  const officialSharpe = asNumber(observed.find((row) => row.candidate_id === OFFICIAL_CANDIDATE_ID || row.CandidateId === OFFICIAL_CANDIDATE_ID)?.Sharpe);
  const selectedSharpe = asNumber(selected?.row.sharpe);
  const insight = selectedSharpe !== null && officialSharpe !== null ? `Selected scenario is ${selectedSharpe < officialSharpe ? "below" : "above"} the official Sharpe reference in this what-if slice.` : null;
  const activeRows = tab === "observed" ? observed : simulated;
  const hasDraftChanges = !sameControls(draft, appliedControls);
  const applyStatus = hasDraftChanges ? "Pending changes" : "Scenario already applied";

  useEffect(() => {
    const normalize = (current: ScenarioControls): ScenarioControls => ({
      fold: nearestOption(current.fold, foldOptions),
      horizon: nearestOption(current.horizon, horizonOptions),
      budget: nearestOption(current.budget, budgetOptions),
      conviction: nearestOption(current.conviction, convictionOptions),
      leader: nearestOption(current.leader, leaderOptions),
      backoff: nearestOption(current.backoff, backoffOptions),
      cost: nearestOption(current.cost, costOptions),
      slippage: nearestOption(current.slippage, slippageOptions),
    });
    setDraft((current) => {
      const next = normalize(current);
      return sameControls(current, next) ? current : next;
    });
    setAppliedControls((current) => {
      const next = normalize(current);
      return sameControls(current, next) ? current : next;
    });
  }, [foldOptions, horizonOptions, budgetOptions, convictionOptions, leaderOptions, backoffOptions, costOptions, slippageOptions]);

  function updateDraft(key: keyof ScenarioControls, value: string) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  function applyScenario() {
    if (!hasDraftChanges) return;
    setAppliedControls(draft);
    setAppliedNonce((value) => value + 1);
  }

  if (((grid.loading && tab === "simulated") || extended.loading) && !extended.data) return <LoadingState label="Loading what-if grid" />;
  if (grid.error) return <ErrorState error={grid.error} retry={grid.retry} />;
  if (extended.error) return <ErrorState error={extended.error} retry={extended.retry} />;

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
            <SelectControl label="Fold" value={draft.fold} options={foldOptions} onChange={(value) => updateDraft("fold", value)} compact />
            <SelectControl label="Horizon" value={draft.horizon} options={horizonOptions} onChange={(value) => updateDraft("horizon", value)} compact />
            <SelectControl label="Budget" value={draft.budget} options={budgetOptions} onChange={(value) => updateDraft("budget", value)} compact />
            <SelectControl label="Conviction" value={draft.conviction} options={convictionOptions} onChange={(value) => updateDraft("conviction", value)} compact />
            <SelectControl label="Leader" value={draft.leader} options={leaderOptions} onChange={(value) => updateDraft("leader", value)} compact />
            <SelectControl label="Backoff" value={draft.backoff} options={backoffOptions} onChange={(value) => updateDraft("backoff", value)} compact />
            <SelectControl label="Cost bps" value={draft.cost} options={costOptions} onChange={(value) => updateDraft("cost", value)} compact />
            <SelectControl label="Slippage bps" value={draft.slippage} options={slippageOptions} onChange={(value) => updateDraft("slippage", value)} compact />
            <button className="primary-button" onClick={applyScenario} disabled={!hasDraftChanges || (!grid.data && grid.loading)}>Apply scenario</button>
            <span>{applyStatus}</span>
          </div>
          <div className="metric-grid">
            <MetricCard label="Nearest scenario distance" value={selected ? selected.dist.toFixed(3) : "—"} detail={selected && selected.dist > 0 ? "nearest valid scenario shown" : "selected controls to available row"} />
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
