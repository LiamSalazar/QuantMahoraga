import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { formatMetric } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

export default function ModuleAttribution({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("all");
  const modules = useApiResource<Record<string, unknown>>("/module/effectiveness", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold });
  if (modules.loading && !modules.data) return <LoadingState label="Loading module attribution" />;
  if (modules.error) return <ErrorState error={modules.error} retry={modules.retry} />;
  const activation = rowsFrom(modules.data, "activation");
  const byHorizon = rowsFrom(modules.data, "by_horizon");
  const bestHelped = [...byHorizon].filter((row) => row.helped_rate !== null).sort((a, b) => Number(b.helped_rate) - Number(a.helped_rate))[0];
  const topActivation = [...activation].filter((row) => row.activation_rate !== null).sort((a, b) => Number(b.activation_rate) - Number(a.activation_rate))[0];
  const weak = [...byHorizon].filter((row) => Number(row.avg_alpha_vs_qqq) < 0).sort((a, b) => Number(a.avg_alpha_vs_qqq) - Number(b.avg_alpha_vs_qqq))[0];
  return (
    <div className="view-grid">
      <section className="panel span-12"><SectionHeader title="Module Attribution" question="Which modules help, hurt or change behavior by horizon?" source="mart.mv_module_effectiveness" action={<SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact />} /><div className="metric-grid">{activation.slice(0, 7).map((row) => <MetricCard key={String(row.module_name)} label={String(row.module_name)} value={formatMetric(row.activation_rate, "activation_rate")} detail={`${formatMetric(row.observations, "observations")} observations`} />)}</div></section>
      <section className="panel span-12"><div className="insight-strip">{bestHelped ? <article className="insight-card"><b>Best helped rate</b><span>{String(bestHelped.module_name)} leads at horizon {String(bestHelped.horizon)}.</span></article> : null}{topActivation ? <article className="insight-card"><b>Highest activation</b><span>{String(topActivation.module_name)} is most frequently active.</span></article> : null}{weak ? <article className="insight-card"><b>Weak contribution</b><span>{String(weak.module_name)} has negative average alpha in this slice.</span></article> : null}</div></section>
      <ChartPanel title="Module x Horizon Matrix" question="Does effectiveness change by forecast horizon?" source="mart.mv_module_effectiveness" ready={byHorizon.length >= 4}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={byHorizon.slice(0, 80)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="horizon" label={axisLabel("Horizon")} /><YAxis label={axisLabel("Module rate", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="helped_rate" fill="#72f0b1" /><Bar dataKey="activation_rate" fill="#80d8ff" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Activation Ranking" question="Which modules most frequently alter the official policy path?" source="fact_module_trace" ready={activation.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={activation} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Activation rate")} /><YAxis dataKey="module_name" type="category" width={190} label={axisLabel("Module", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="activation_rate" fill="#f7c76a" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-12"><SectionHeader title="Module Table" question="Activation, helped rate, alpha and drawdown diagnostics by module/horizon." source="mart.mv_module_effectiveness" /><DataTable rows={byHorizon} columns={["module_name", "horizon", "activation_rate", "helped_rate", "avg_alpha_vs_qqq", "avg_drawdown_change", "avg_exposure_effect", "observations", "demo_mode"]} /></section>
    </div>
  );
}
