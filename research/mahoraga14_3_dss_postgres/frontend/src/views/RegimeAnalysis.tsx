import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options, ViewKey } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { formatMetric } from "../utils/format";
import { rowsFrom } from "../utils/rows";

export default function RegimeAnalysis({ options, onOpenView }: { options: Options | null; onOpenView?: (view: ViewKey) => void }) {
  const [fold, setFold] = useState("all");
  const data = useApiResource<Record<string, unknown>>("/regime/behavior", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold });
  if (data.loading && !data.data) return <LoadingState label="Loading regime analysis" />;
  if (data.error) return <ErrorState error={data.error} retry={data.retry} />;
  const rows = rowsFrom(data.data);
  const best = [...rows].filter((row) => row.avg_net_return !== null).sort((a, b) => Number(b.avg_net_return) - Number(a.avg_net_return))[0];
  const weakest = [...rows].filter((row) => row.avg_net_return !== null).sort((a, b) => Number(a.avg_net_return) - Number(b.avg_net_return))[0];
  const exposure = [...rows].filter((row) => row.avg_exposure !== null).sort((a, b) => Number(b.avg_exposure) - Number(a.avg_exposure))[0];
  const backoff = [...rows].filter((row) => row.backoff_activation_rate !== null).sort((a, b) => Number(b.backoff_activation_rate) - Number(a.backoff_activation_rate))[0];
  function openOlap(regime: unknown) {
    if (!onOpenView || !regime) return;
    sessionStorage.setItem("mahoragaOlapPreset", "regime-best-alpha");
    sessionStorage.setItem("mahoragaOlapRegime", String(regime));
    onOpenView("olap");
  }
  return (
    <div className="view-grid">
      <section className="panel span-12"><SectionHeader title="Regime Analysis" question={fold === "all" ? "Fold-all view is aggregated to one bar per regime." : "Selected fold regime behavior with drill-through."} source="mart.mv_regime_behavior" action={<SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact />} /></section>
      <section className="panel span-12"><div className="insight-strip">{best ? <article className="insight-card"><b>Best regime</b><span>{String(best.regime)} leads average net return at {formatMetric(best.avg_net_return, "return")}.</span></article> : null}{weakest ? <article className="insight-card"><b>Weakest regime</b><span>{String(weakest.regime)} is weakest at {formatMetric(weakest.avg_net_return, "return")}.</span></article> : null}{exposure ? <article className="insight-card"><b>Highest exposure</b><span>{String(exposure.regime)} carries {formatMetric(exposure.avg_exposure, "exposure")} average exposure.</span></article> : null}{backoff ? <article className="insight-card"><b>Backoff activation</b><span>{String(backoff.regime)} has {formatMetric(backoff.backoff_activation_rate, "rate")} backoff activation.</span></article> : null}</div></section>
      <ChartPanel title="Regime Return Matrix" question="Which regimes produce positive excess return?" source="fact_decision_state" ready={rows.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows}><CartesianGrid stroke="#22303a" /><XAxis dataKey="regime" label={axisLabel("Regime")} /><YAxis label={axisLabel("Return", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="avg_net_return" fill="#72f0b1" /><Bar dataKey="avg_benchmark_return" fill="#80d8ff" /></BarChart></ResponsiveContainer></ChartPanel>
      <ChartPanel title="Exposure / Backoff Behavior" question="How does risk management respond by state?" source="fact_decision_state" ready={rows.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows}><CartesianGrid stroke="#22303a" /><XAxis dataKey="regime" label={axisLabel("Regime")} /><YAxis label={axisLabel("Rate / exposure", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="avg_exposure" fill="#f7c76a" /><Bar dataKey="backoff_activation_rate" fill="#ff8a7a" /><Bar dataKey="continuation_activation_rate" fill="#80d8ff" /></BarChart></ResponsiveContainer></ChartPanel>
      <section className="panel span-12"><SectionHeader title="Regime Matrix Table" question="Return, benchmark, exposure, drawdown, backoff, continuation and leader blend by regime." source="mart.mv_regime_behavior" /><DataTable rows={rows} columns={["regime", "avg_net_return", "avg_benchmark_return", "avg_exposure", "avg_turnover", "avg_drawdown", "backoff_activation_rate", "continuation_activation_rate", "avg_leader_blend", "observations", "demo_mode"]} rowAction={(row) => (onOpenView ? <button className="ghost-button" onClick={() => openOlap(row.regime)}>OLAP</button> : null)} /></section>
    </div>
  );
}
