import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { DataTable } from "../components/DataTable";
import { SelectControl } from "../components/Controls";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasHeatmap, hasScatter } from "../utils/chartGuards";
import { OFFICIAL_CANDIDATE_ID, formatCandidateLabel } from "../utils/labels";
import { rowsFrom } from "../utils/rows";
import { useState } from "react";

export default function RobustnessLab({ options }: { options: Options | null }) {
  const [metric, setMetric] = useState("Sharpe");
  const extended = useApiResource<Record<string, unknown>>("/research/extended-summary");
  const surface = useApiResource<Record<string, unknown>>("/robustness/surface", { metric, universe_id: options?.default_universe ?? "base_universe_12", limit: 1200 });
  if ((extended.loading || surface.loading) && !extended.data) return <LoadingState label="Loading robustness evidence" />;
  if (extended.error) return <ErrorState error={extended.error} retry={extended.retry} />;
  if (surface.error) return <ErrorState error={surface.error} retry={surface.retry} />;
  const summary = rowsFrom(extended.data, "extended_multiplier_summary");
  const sensitivity = rowsFrom(extended.data, "sensitivity_ranking");
  const plateau = rowsFrom(extended.data, "plateau_radius");
  const points = rowsFrom(surface.data);
  const pareto = summary.map((row) => ({ ...row, candidate_id: row.candidate_id ?? row.CandidateId, cagr: row.CAGR, sharpe: row.Sharpe, maxdd: row.MaxDD }));
  const official = pareto.find((row) => row.candidate_id === OFFICIAL_CANDIDATE_ID);

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Robustness Lab" question="Is the official baseline robust or dependent on a fragile configuration?" source="fact_robustness_surface + extended multiplier audit" action={<SelectControl label="Metric" value={metric} options={options?.metrics?.includes("Sharpe") ? ["Sharpe", "CAGR", "MaxDD", "AlphaNW_QQQ", "robust_score"] : ["Sharpe"]} onChange={setMetric} compact />} />
        <div className="callout">
          <b>{formatCandidateLabel(OFFICIAL_CANDIDATE_ID)}</b>
          <span>Official marker remains visible; best observed candidates are audit evidence, not a replacement baseline.</span>
        </div>
      </section>
      <ChartPanel title="Sensitivity Tornado" question="Which multiplier axis causes the largest degradation?" source="sensitivity_ranking.csv" ready={sensitivity.length >= 2}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={sensitivity} layout="vertical">
            <CartesianGrid stroke="#22303a" />
            <XAxis type="number" />
            <YAxis dataKey="axis" type="category" width={150} />
            <Tooltip />
            <Bar dataKey="sensitivity_score" fill="#f7c76a" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Pareto Frontier" question="What return/drawdown trade-off did the observed sweep expose?" source="extended_multiplier_summary.csv" ready={hasScatter(pareto, "maxdd", "cagr", 4)}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="maxdd" type="number" name="MaxDD" />
            <YAxis dataKey="cagr" type="number" name="CAGR" />
            <ZAxis dataKey="sharpe" range={[50, 240]} />
            <Tooltip />
            <Scatter data={pareto}>
              {pareto.map((row: Row, index) => <Cell key={index} fill={row.candidate_id === official?.candidate_id ? "#f7c76a" : "#80d8ff"} />)}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Robustness Surface" question="Does the metric form a usable plateau?" source="mart.mv_robustness_surface" ready={hasHeatmap(points, "budget_multiplier", "leader_multiplier", "metric_value")}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="budget_multiplier" type="number" />
            <YAxis dataKey="leader_multiplier" type="number" />
            <ZAxis dataKey="metric_value" range={[40, 280]} />
            <Tooltip />
            <Scatter data={points} fill="#72f0b1" />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartPanel>
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
