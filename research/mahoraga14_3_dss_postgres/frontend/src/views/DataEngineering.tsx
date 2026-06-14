import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import type { HealthSummary, Row } from "../api/types";
import { formatMetric, formatNumber } from "../utils/format";
import { rowsFrom } from "../utils/rows";

export default function DataEngineering() {
  const health = useApiResource<HealthSummary>("/data/health-summary", undefined, true, false);
  const perf = useApiResource<Record<string, unknown>>("/query/performance", undefined, true, false);
  if ((health.loading || perf.loading) && !health.data) return <LoadingState label="Loading data engineering evidence" />;
  if (health.error) return <ErrorState error={health.error} retry={health.retry} />;
  if (perf.error) return <ErrorState error={perf.error} retry={perf.retry} />;
  const rows = rowsFrom(perf.data);
  const counts = health.data?.row_counts ?? {};
  const tableRows: Row[] = Object.entries(counts).map(([table_name, row_count]) => ({ table_name, row_count, layer: table_name.includes(".") ? table_name.split(".")[0] : table_name.startsWith("fact_") || table_name.startsWith("dim_") ? "dw" : "oltp/parquet" }));
  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Data Engineering" question="What architecture supports the DSS?" source="OLTP + DW + mart materialized views + query logs" />
        <div className="metric-grid">
          <MetricCard label="Backend active" value={health.data?.backend ?? "n/a"} detail={health.data?.profile ?? "profile"} />
          <MetricCard label="Latest run_id" value={String(health.data?.latest_run_id ?? "n/a")} />
          <MetricCard label="Real rows" value={formatNumber(health.data?.real_rows, 0)} detail="artifact-derived estimate" />
          <MetricCard label="Simulated rows" value={formatNumber(health.data?.simulated_rows, 0)} detail="flagged what-if" />
          <MetricCard label="Validation" value={String(health.data?.validation_passed ?? "n/a")} />
          <MetricCard label="Query logs" value={String(health.data?.query_logs_active ? "active" : "warming")} detail={`${formatNumber(health.data?.query_log_count, 0)} grouped endpoints`} />
        </div>
      </section>
      <ChartPanel title="Latency by Endpoint" question="Which API slices are expensive?" source="oltp.dss_query_log / mart.mv_query_performance" ready={rows.length >= 1}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 24)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="endpoint" hide /><YAxis /><Tooltip /><Bar dataKey="avg_elapsed_ms" fill="#f7c76a" /><Bar dataKey="p95_elapsed_ms" fill="#ff8a7a" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Query Count by Endpoint" question="Which research slices are being used most?" source="oltp.dss_query_log" ready={rows.length >= 1}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 24)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="endpoint" hide /><YAxis /><Tooltip /><Bar dataKey="query_count" fill="#80d8ff" /><Bar dataKey="avg_rows_returned" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-6"><SectionHeader title="Layer Row Counts" question="Logical row counts avoid treating empty partition parents as fatal." source="/data/health-summary" /><DataTable rows={tableRows} columns={["layer", "table_name", "row_count"]} pageSize={14} /></section>
      <section className="panel span-6"><SectionHeader title="Query Performance Table" question="Endpoint, source relation, materialized-view use, latency and rows returned." source="/query/performance" /><DataTable rows={rows} columns={["endpoint", "source_relation", "used_materialized_view", "query_count", "avg_elapsed_ms", "p95_elapsed_ms", "avg_rows_returned", "last_seen_at"]} /></section>
      <section className="panel span-12"><SectionHeader title="Architecture Interpretation" question="Evidence of engineering depth, not a decorative dashboard." source="DSS run artifacts" /><div className="architecture-strip"><span>OLTP ingestion snapshots</span><span>Dimensional DW facts</span><span>Refreshable marts</span><span>FastAPI audit endpoints</span><span>Lazy React command center</span></div><p className="muted">The UI distinguishes audited artifact rows from simulated what-if rows and queries marts/endpoints instead of loading raw facts on startup.</p></section>
    </div>
  );
}
