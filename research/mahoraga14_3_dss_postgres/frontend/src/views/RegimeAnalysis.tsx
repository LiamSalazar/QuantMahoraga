import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

export default function RegimeAnalysis({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("all");
  const data = useApiResource<Record<string, unknown>>("/regime/behavior", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold });
  if (data.loading && !data.data) return <LoadingState label="Loading regime analysis" />;
  if (data.error) return <ErrorState error={data.error} retry={data.retry} />;
  const rows = rowsFrom(data.data);
  return (
    <div className="view-grid">
      <section className="panel span-12"><SectionHeader title="Regime Analysis" question="In which states does Mahoraga work better or worse?" source="mart.mv_regime_behavior" action={<SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact />} /></section>
      <ChartPanel title="Return vs Benchmark vs Alpha Proxy" question="Which regimes produce positive excess return?" source="fact_decision_state" ready={rows.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows}><CartesianGrid stroke="#22303a" /><XAxis dataKey="regime" /><YAxis /><Tooltip /><Bar dataKey="avg_net_return" fill="#72f0b1" /><Bar dataKey="avg_benchmark_return" fill="#80d8ff" /></BarChart></ResponsiveContainer></ChartPanel>
      <ChartPanel title="Exposure / Backoff Behavior" question="How does risk management respond by state?" source="fact_decision_state" ready={rows.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows}><CartesianGrid stroke="#22303a" /><XAxis dataKey="regime" /><YAxis /><Tooltip /><Bar dataKey="avg_exposure" fill="#f7c76a" /><Bar dataKey="backoff_activation_rate" fill="#ff8a7a" /><Bar dataKey="continuation_activation_rate" fill="#80d8ff" /></BarChart></ResponsiveContainer></ChartPanel>
      <section className="panel span-12"><SectionHeader title="Regime Matrix Table" question="Return, benchmark, exposure, drawdown, backoff, continuation and leader blend by regime." source="mart.mv_regime_behavior" /><DataTable rows={rows} /></section>
    </div>
  );
}
