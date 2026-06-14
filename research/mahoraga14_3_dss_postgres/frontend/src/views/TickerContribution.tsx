import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasScatter } from "../utils/chartGuards";
import { formatMetric, formatNumber } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom, topRows } from "../utils/rows";

export default function TickerContribution({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("all");
  const data = useApiResource<Record<string, unknown>>("/ticker/contribution", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold, limit: 200 });
  if (data.loading && !data.data) return <LoadingState label="Loading ticker contribution" />;
  if (data.error) return <ErrorState error={data.error} retry={data.retry} />;
  const rows = rowsFrom(data.data);
  const positives = topRows(rows, "total_pnl_contribution", 12);
  const negatives = topRows(rows, "total_pnl_contribution", 12, false);
  const top3Share = positives.slice(0, 3).reduce((sum, row) => sum + Number(row.total_pnl_contribution ?? 0), 0) / Math.max(0.000001, rows.reduce((sum, row) => sum + Math.max(0, Number(row.total_pnl_contribution ?? 0)), 0));
  return (
    <div className="view-grid">
      <section className="panel span-12"><SectionHeader title="Ticker Contribution" question="Which tickers explain results and where is concentration?" source="mart.mv_ticker_contribution" action={<SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact />} /><div className="metric-grid"><MetricCard label="Tickers" value={formatNumber(rows.length, 0)} /><MetricCard label="Top 3 contribution share" value={formatMetric(top3Share, "share")} /><MetricCard label="Highest contributor" value={String(positives[0]?.ticker ?? "n/a")} detail={formatMetric(positives[0]?.total_pnl_contribution, "return")} /><MetricCard label="Largest drag" value={String(negatives[0]?.ticker ?? "n/a")} detail={formatMetric(negatives[0]?.total_pnl_contribution, "return")} /></div></section>
      <ChartPanel title="Positive Contributors" question="Which names added most PnL contribution?" source="mart.mv_ticker_contribution" ready={positives.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={positives} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" /><YAxis dataKey="ticker" type="category" width={70} /><Tooltip /><Bar dataKey="total_pnl_contribution" fill="#72f0b1" /></BarChart></ResponsiveContainer></ChartPanel>
      <ChartPanel title="Negative Contributors" question="Which names were the largest drags?" source="mart.mv_ticker_contribution" ready={negatives.length >= 2}><ResponsiveContainer width="100%" height="100%"><BarChart data={negatives} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" /><YAxis dataKey="ticker" type="category" width={70} /><Tooltip /><Bar dataKey="total_pnl_contribution" fill="#ff8a7a" /></BarChart></ResponsiveContainer></ChartPanel>
      <ChartPanel title="Selection vs Leadership" question="Does frequent selection translate into leadership and contribution?" source="fact_position_daily roll-up" ready={hasScatter(rows, "selection_rate", "leader_flag_rate", 4)}><ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="selection_rate" type="number" /><YAxis dataKey="leader_flag_rate" type="number" /><ZAxis dataKey="avg_final_weight" range={[40, 240]} /><Tooltip /><Scatter data={rows} fill="#80d8ff" /></ScatterChart></ResponsiveContainer></ChartPanel>
      <section className="panel span-6"><SectionHeader title="Sortable Evidence Table" question="Use ticker rows to drill into Decision Replay or OLAP manually." source="mart.mv_ticker_contribution" /><DataTable rows={rows} /></section>
    </div>
  );
}
