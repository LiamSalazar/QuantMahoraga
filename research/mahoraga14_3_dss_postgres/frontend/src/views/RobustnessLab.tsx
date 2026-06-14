import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { DataTable } from "../components/DataTable";
import { SelectControl } from "../components/Controls";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasHeatmap, hasScatter } from "../utils/chartGuards";
import { asNumber } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID, formatCandidateLabel } from "../utils/labels";
import { rowsFrom, topRows } from "../utils/rows";

const axes = ["budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength"];
const metricOptions = ["Sharpe", "CAGR", "MaxDD", "AlphaNW_QQQ", "robust_score"];

export default function RobustnessLab({ options }: { options: Options | null }) {
  const [metric, setMetric] = useState("Sharpe");
  const [xAxis, setXAxis] = useState("budget_multiplier");
  const [yAxis, setYAxis] = useState("leader_multiplier");
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const surface = useApiResource<Record<string, unknown>>("/robustness/surface", { metric, universe_id: options?.default_universe ?? "base_universe_12", limit: 1200 });
  const compare = useApiResource<Record<string, unknown>>("/research/robustness-compare", { universe_id: options?.default_universe ?? "base_universe_12" });
  if ((extended.loading || surface.loading || compare.loading) && !extended.data) return <LoadingState label="Loading robustness evidence" />;
  if (extended.error) return <ErrorState error={extended.error} retry={extended.retry} />;
  if (surface.error) return <ErrorState error={surface.error} retry={surface.retry} />;
  if (compare.error) return <ErrorState error={compare.error} retry={compare.retry} />;

  const summary = rowsFrom(extended.data, "extended_multiplier_summary");
  const sensitivity = rowsFrom(extended.data, "sensitivity_ranking").filter((row) => row.sensitivity_score !== null);
  const plateau = rowsFrom(extended.data, "plateau_radius");
  const points = rowsFrom(surface.data).filter((row) => row[xAxis] !== null && row[yAxis] !== null && row.metric_value !== null);
  const pareto = summary
    .map((row) => ({ ...row, candidate_id: row.candidate_id ?? row.CandidateId, cagr: row.CAGR, sharpe: row.Sharpe, maxdd: row.MaxDD }))
    .filter((row) => row.cagr !== null && row.sharpe !== null && row.maxdd !== null);
  const compareRows = rowsFrom(compare.data);
  const usableSurface = hasHeatmap(points, xAxis, yAxis, "metric_value", 6);
  const insight = useMemo(() => {
    const dominant = topRows(sensitivity, "sensitivity_score", 1)[0];
    return dominant ? `${String(dominant.axis).replaceAll("_", " ")} is the dominant sensitivity axis in the current sweep.` : null;
  }, [sensitivity]);

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Robustness Lab" question="Is the official baseline robust or dependent on a fragile configuration?" source="fact_robustness_surface + extended multiplier audit" />
        <div className="callout">
          <b>{formatCandidateLabel(OFFICIAL_CANDIDATE_ID)}</b>
          <span>Best observed candidates are audit evidence. Compare deltas against the frozen official point before treating a sweep candidate as a replacement.</span>
        </div>
      </section>

      {insight ? <section className="panel span-12"><div className="insight-card"><b>Insight</b><span>{insight}</span></div></section> : null}

      <ChartPanel title="Sensitivity Tornado" question="Which multiplier axis causes the largest degradation?" source="sensitivity_ranking.csv" ready={sensitivity.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={sensitivity} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Sensitivity score")} /><YAxis dataKey="axis" type="category" width={170} label={axisLabel("Parameter", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="sensitivity_score" fill="#f7c76a" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Pareto Frontier" question="What return/drawdown trade-off did the observed sweep expose?" source="extended_multiplier_summary.csv" ready={hasScatter(pareto, "maxdd", "cagr", 4)}>
        <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="maxdd" type="number" name="MaxDD" label={axisLabel("Max drawdown")} /><YAxis dataKey="cagr" type="number" name="CAGR" label={axisLabel("CAGR", true)} /><ZAxis dataKey="sharpe" range={[50, 240]} /><Tooltip content={<ChartTooltip />} /><Scatter data={pareto}>{pareto.map((row: Row, index) => <Cell key={index} fill={row.candidate_id === OFFICIAL_CANDIDATE_ID ? "#f7c76a" : "#80d8ff"} />)}</Scatter></ScatterChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title={usableSurface ? "Robustness Surface" : "Sparse Sweep Fallback"}
        question={usableSurface ? "Does the metric form a usable plateau?" : "Sparse sweep on this pair; showing ranked sensitivity instead."}
        source="mart.mv_robustness_surface"
        ready={usableSurface || sensitivity.length >= 2}
        action={<><SelectControl label="Metric" value={metric} options={metricOptions} onChange={setMetric} compact /><SelectControl label="X axis" value={xAxis} options={axes} onChange={setXAxis} compact /><SelectControl label="Y axis" value={yAxis} options={axes.filter((axis) => axis !== xAxis)} onChange={setYAxis} compact /></>}
      >
        {usableSurface ? (
          <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey={xAxis} type="number" label={axisLabel(xAxis.replaceAll("_", " "))} /><YAxis dataKey={yAxis} type="number" label={axisLabel(yAxis.replaceAll("_", " "), true)} /><ZAxis dataKey="metric_value" range={[40, 280]} /><Tooltip content={<ChartTooltip />} /><Scatter data={points}>{points.map((row, index) => <Cell key={index} fill={row.candidate_id === OFFICIAL_CANDIDATE_ID || (asNumber(row[xAxis]) === 1.05 && asNumber(row[yAxis]) === 1.1) ? "#f7c76a" : "#72f0b1"} />)}</Scatter></ScatterChart></ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%"><BarChart data={sensitivity} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Sensitivity score")} /><YAxis dataKey="axis" type="category" width={170} label={axisLabel("Parameter", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="sensitivity_score" fill="#80d8ff" /></BarChart></ResponsiveContainer>
        )}
      </ChartPanel>

      <section className="panel span-6">
        <SectionHeader title="Best / Official / Worst Compare" question="Deltas explain why a higher-scoring candidate may not replace the official baseline." source="/research/robustness-compare" />
        <DataTable rows={compareRows} columns={["compare_role", "candidate_id", "cagr", "delta_cagr", "sharpe", "delta_sharpe", "sortino", "delta_sortino", "maxdd", "delta_maxdd", "alpha_qqq", "delta_alpha_qqq", "robust_score", "severe_fold_damage_count"]} />
      </section>
      <section className="panel span-6">
        <SectionHeader title="Plateau Radius" question="How far can each axis move before robust-region decay?" source="plateau_radius_by_axis.csv" />
        <DataTable rows={plateau} />
      </section>
      <section className="panel span-12">
        <SectionHeader title="Candidate Ranking and Worst-Fold Damage" question="Observed sweep rows with severe fold damage and robust region flags." source="extended_multiplier_summary.csv" />
        <DataTable rows={summary} columns={["candidate_id", "sweep_role", "CAGR", "Sharpe", "Sortino", "MaxDD", "robust_region_flag", "severe_fold_damage_count", "worst_fold_sharpe_delta_vs_official"]} />
      </section>
    </div>
  );
}
