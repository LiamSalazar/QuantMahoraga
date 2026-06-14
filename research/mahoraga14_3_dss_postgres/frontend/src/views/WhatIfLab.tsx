import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
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
import { asNumber, formatMetric, formatText } from "../utils/format";
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

const initialControls: ScenarioControls = { fold: "1", horizon: "20", budget: "1.05", conviction: "1.1", leader: "1.1", backoff: "1.05", cost: "5", slippage: "2" };

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

function uniqueValues(rows: Row[], key: string, fallback: (string | number)[]) {
  const values = Array.from(new Set(rows.map((row) => row[key]).filter((value) => value !== null && value !== undefined).map((value) => String(value))));
  return values.length ? values : fallback.map((value) => String(value));
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

function nearestScenario(rows: Row[], controls: ScenarioControls) {
  return rows
    .map((row) => ({ row, dist: scenarioDistance(row, controls) }))
    .filter((item): item is { row: Row; dist: number } => item.dist !== null)
    .sort((a, b) => a.dist - b.dist)[0] ?? null;
}

function delta(value: unknown, reference: unknown, key: string) {
  const a = asNumber(value);
  const b = asNumber(reference);
  if (a === null || b === null) return "No reference";
  return formatMetric(a - b, key);
}

function scenarioId(row?: Row | null) {
  return String(row?.scenario_id ?? `${row?.budget_multiplier}-${row?.conviction_multiplier}-${row?.leader_multiplier}-${row?.backoff_strength}-${row?.cost_bps}-${row?.slippage_bps}-${row?.fold}-${row?.horizon}`);
}

function marked(row: Row, preview?: Row | null, applied?: Row | null, best?: Row | null) {
  const id = scenarioId(row);
  if (applied && id === scenarioId(applied)) return "Applied scenario";
  if (preview && id === scenarioId(preview)) return "Preview scenario";
  if (best && id === scenarioId(best)) return "Best simulated scenario";
  return "Simulated scenario";
}

export default function WhatIfLab({ options }: { options: Options | null }) {
  const [tab, setTab] = useState<"observed" | "simulated">("observed");
  const [yMetric, setYMetric] = useState<"cagr" | "sharpe">("cagr");
  const [draftControls, setDraftControls] = useState<ScenarioControls>(initialControls);
  const [appliedControls, setAppliedControls] = useState<ScenarioControls>(initialControls);
  const [appliedNonce, setAppliedNonce] = useState(0);
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const previewGrid = useApiResource<Record<string, unknown>>(
    "/whatif/grid",
    { candidate_id: OFFICIAL_CANDIDATE_ID, fold: draftControls.fold, universe_id: options?.default_universe ?? "base_universe_12", horizon: draftControls.horizon, cost_bps: draftControls.cost, slippage_bps: draftControls.slippage, limit: 1000 },
    tab === "simulated",
  );
  const appliedGrid = useApiResource<Record<string, unknown>>(
    "/whatif/grid",
    { candidate_id: OFFICIAL_CANDIDATE_ID, fold: appliedControls.fold, universe_id: options?.default_universe ?? "base_universe_12", horizon: appliedControls.horizon, cost_bps: appliedControls.cost, slippage_bps: appliedControls.slippage, limit: 1000, appliedNonce },
    tab === "simulated",
  );
  const reference = useApiResource<Record<string, unknown>>(
    "/research/whatif-reference",
    { candidate_id: OFFICIAL_CANDIDATE_ID, fold: appliedControls.fold, universe_id: options?.default_universe ?? "base_universe_12", horizon: appliedControls.horizon, cost_bps: appliedControls.cost, slippage_bps: appliedControls.slippage },
    tab === "simulated",
  );

  const foldOptions = useMemo(() => ["all", ...(options?.folds ?? [1, 2, 3, 4, 5])], [options?.folds]);
  const horizonOptions = useMemo(() => options?.horizons ?? [1, 5, 20, 60], [options?.horizons]);
  const simulatedPreviewRows = rowsFrom(previewGrid.data).filter(complete);
  const simulatedAppliedRows = rowsFrom(appliedGrid.data).filter(complete);
  const observed = rowsFrom(extended.data, "extended_multiplier_summary").filter((row) => row.demo_mode !== true && complete(row));
  const pareto = rowsFrom(appliedGrid.data, "pareto").filter(complete);
  const budgetOptions = useMemo(() => uniqueValues(simulatedPreviewRows, "budget_multiplier", rangeValues(options, "budget_multiplier", [0.9, 0.95, 1, 1.05, 1.1, 1.15])), [simulatedPreviewRows, options]);
  const convictionOptions = useMemo(() => uniqueValues(simulatedPreviewRows, "conviction_multiplier", rangeValues(options, "conviction_multiplier", [0.9, 1, 1.1, 1.2, 1.3])), [simulatedPreviewRows, options]);
  const leaderOptions = useMemo(() => uniqueValues(simulatedPreviewRows, "leader_multiplier", rangeValues(options, "leader_multiplier", [0.9, 1, 1.1, 1.2, 1.3])), [simulatedPreviewRows, options]);
  const backoffOptions = useMemo(() => uniqueValues(simulatedPreviewRows, "backoff_strength", rangeValues(options, "backoff_strength", [0.9, 1, 1.05, 1.1, 1.2])), [simulatedPreviewRows, options]);
  const costOptions = useMemo(() => rangeValues(options, "cost_bps", [0, 5, 10, 25]), [options]);
  const slippageOptions = useMemo(() => rangeValues(options, "slippage_bps", [0, 2, 5, 10]), [options]);
  const previewScenario = useMemo(() => nearestScenario(simulatedPreviewRows, draftControls), [simulatedPreviewRows, draftControls]);
  const appliedScenario = useMemo(() => nearestScenario(simulatedAppliedRows, appliedControls), [simulatedAppliedRows, appliedControls]);
  const officialBaselineReference = (reference.data?.official ?? null) as Row | null;
  const bestScenario = ((reference.data?.best_simulated ?? null) as Row | null) ?? topRows(simulatedAppliedRows, "robust_score", 1)[0] ?? null;
  const hasDraftChanges = !sameControls(draftControls, appliedControls);
  const canApply = Boolean(previewScenario?.row) && hasDraftChanges;
  const activeRows = tab === "observed" ? observed : simulatedAppliedRows;
  const chartRows = pareto.map((row) => ({ ...row, scenario_type: marked(row, previewScenario?.row, appliedScenario?.row, bestScenario) }));
  const officialPoint = officialBaselineReference ? [{ ...officialBaselineReference, scenario_type: "Official baseline", scenario_id: "official-baseline" }] : [];
  const appliedPoint = appliedScenario?.row ? [{ ...appliedScenario.row, scenario_type: "Applied scenario" }] : [];
  const previewPoint = previewScenario?.row ? [{ ...previewScenario.row, scenario_type: "Preview scenario" }] : [];
  const bestPoint = bestScenario ? [{ ...bestScenario, scenario_type: "Best simulated scenario" }] : [];
  const applyLabel = !previewScenario ? "No valid scenario" : hasDraftChanges ? "Apply scenario" : "Scenario applied";

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
    setDraftControls((current) => {
      const next = normalize(current);
      return sameControls(current, next) ? current : next;
    });
    setAppliedControls((current) => {
      const next = normalize(current);
      return sameControls(current, next) ? current : next;
    });
  }, [foldOptions, horizonOptions, budgetOptions, convictionOptions, leaderOptions, backoffOptions, costOptions, slippageOptions]);

  function updateDraft(key: keyof ScenarioControls, value: string) {
    setDraftControls((current) => ({ ...current, [key]: value }));
  }

  function applyScenario() {
    if (!canApply) return;
    setAppliedControls(draftControls);
    setAppliedNonce((value) => value + 1);
  }

  if (((previewGrid.loading || appliedGrid.loading) && tab === "simulated" && !appliedGrid.data) || (extended.loading && !extended.data)) return <LoadingState label="Loading what-if grid" />;
  if (previewGrid.error) return <ErrorState error={previewGrid.error} retry={previewGrid.retry} />;
  if (appliedGrid.error) return <ErrorState error={appliedGrid.error} retry={appliedGrid.retry} />;
  if (reference.error) return <ErrorState error={reference.error} retry={reference.retry} />;
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
          <SectionHeader title="Simulated Scenario Builder" question="Draft controls preview the nearest valid scenario before Apply." source="fact_whatif demo_mode=true" />
          <div className="lab-controls">
            <SelectControl label="Fold" value={draftControls.fold} options={foldOptions} onChange={(value) => updateDraft("fold", value)} compact />
            <SelectControl label="Horizon" value={draftControls.horizon} options={horizonOptions} onChange={(value) => updateDraft("horizon", value)} compact />
            <SelectControl label="Budget" value={draftControls.budget} options={budgetOptions} onChange={(value) => updateDraft("budget", value)} compact />
            <SelectControl label="Conviction" value={draftControls.conviction} options={convictionOptions} onChange={(value) => updateDraft("conviction", value)} compact />
            <SelectControl label="Leader" value={draftControls.leader} options={leaderOptions} onChange={(value) => updateDraft("leader", value)} compact />
            <SelectControl label="Backoff" value={draftControls.backoff} options={backoffOptions} onChange={(value) => updateDraft("backoff", value)} compact />
            <SelectControl label="Cost bps" value={draftControls.cost} options={costOptions} onChange={(value) => updateDraft("cost", value)} compact />
            <SelectControl label="Slippage bps" value={draftControls.slippage} options={slippageOptions} onChange={(value) => updateDraft("slippage", value)} compact />
            <SelectControl label="Y metric" value={yMetric} options={["cagr", "sharpe"]} onChange={(value) => setYMetric(value === "sharpe" ? "sharpe" : "cagr")} compact />
            <button className="primary-button" onClick={applyScenario} disabled={!canApply}>{applyLabel}</button>
          </div>
          <div className="metric-grid">
            <MetricCard label="Official baseline" value={formatMetric(officialBaselineReference?.sharpe, "Sharpe")} detail={`CAGR ${formatMetric(officialBaselineReference?.cagr, "CAGR")}`} />
            <MetricCard label="Preview scenario" value={formatMetric(previewScenario?.row.sharpe, "Sharpe")} detail={`${previewScenario ? previewScenario.dist.toFixed(3) : "No"} distance`} />
            <MetricCard label="Applied scenario" value={formatMetric(appliedScenario?.row.sharpe, "Sharpe")} detail={formatDemoMode(appliedScenario?.row.demo_mode)} />
            <MetricCard label="Best simulated" value={formatMetric(bestScenario?.sharpe, "Sharpe")} detail={formatText(bestScenario?.scenario_id ?? bestScenario?.candidate_id)} />
            <MetricCard label="Preview vs official" value={delta(previewScenario?.row.sharpe, officialBaselineReference?.sharpe, "Sharpe")} detail={`CAGR ${delta(previewScenario?.row.cagr, officialBaselineReference?.cagr, "CAGR")}`} />
            <MetricCard label="Preview vs applied" value={delta(previewScenario?.row.sharpe, appliedScenario?.row.sharpe, "Sharpe")} detail={`MaxDD ${delta(previewScenario?.row.maxdd, appliedScenario?.row.maxdd, "MaxDD")}`} />
          </div>
        </section>
      ) : null}

      <ChartPanel title={tab === "observed" ? "Observed Scenario Ranking" : "Simulated Pareto Frontier"} question={tab === "observed" ? "Which audited scenarios were strongest without simulation?" : "Official, preview, applied and best simulated are explicitly marked."} source={tab === "observed" ? "extended_multiplier_summary.csv" : "fact_whatif"} ready={tab === "observed" ? observed.length >= 4 : hasScatter(chartRows, "maxdd", yMetric, 4)}>
        {tab === "observed" ? (
          <ResponsiveContainer width="100%" height="100%"><BarChart data={topRows(observed, "Sharpe", 12)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="candidate_id" hide label={axisLabel("Candidate")} /><YAxis label={axisLabel("Sharpe", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="Sharpe" fill="#72f0b1" /></BarChart></ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="maxdd" type="number" label={axisLabel("Max drawdown")} /><YAxis dataKey={yMetric} type="number" label={axisLabel(yMetric.toUpperCase(), true)} /><ZAxis dataKey="sharpe" range={[50, 220]} /><Tooltip content={<ChartTooltip />} /><Scatter data={chartRows}>{chartRows.map((row, index) => <Cell key={index} fill={row.scenario_type === "Applied scenario" ? "#f7c76a" : row.scenario_type === "Preview scenario" ? "#80d8ff" : row.scenario_type === "Best simulated scenario" ? "#72f0b1" : "#4d6873"} />)}</Scatter><Scatter data={officialPoint} fill="#ffffff" shape="cross" /><Scatter data={bestPoint} fill="#72f0b1" shape="star" /><Scatter data={previewPoint} fill="#80d8ff" shape="triangle" /><Scatter data={appliedPoint} fill="#f7c76a" shape="diamond" /></ScatterChart></ResponsiveContainer>
        )}
      </ChartPanel>

      <section className="panel span-12">
        <SectionHeader title={tab === "observed" ? "Observed / Audited Scenario Ranking" : "Simulated What-if Ranking"} question={tab === "observed" ? "These are observed audit rows, not simulated what-if rows." : "Rows are explicitly flagged and the applied scenario is highlighted."} source={tab === "observed" ? "extended_multiplier_summary.csv" : "fact_whatif"} />
        <DataTable
          rows={activeRows}
          columns={tab === "observed" ? ["candidate_id", "sweep_role", "CAGR", "Sharpe", "MaxDD", "robust_region_flag", "severe_fold_damage_count"] : ["scenario_id", "fold", "horizon", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cost_bps", "slippage_bps", "cagr", "sharpe", "maxdd", "demo_mode"]}
          rowClassName={(row) => (appliedScenario?.row && scenarioId(row) === scenarioId(appliedScenario.row) ? "highlight-row" : "")}
        />
      </section>
    </div>
  );
}
