import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import type { HealthSummary, Row } from "../api/types";
import { formatMetric, formatNumber, formatText } from "../utils/format";
import { rowsFrom } from "../utils/rows";

function ms(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(1)} ms` : "No latency";
}

function shortRunId(value: unknown) {
  const text = String(value ?? "No run id");
  return text.length > 22 ? `${text.slice(0, 22)}...` : text;
}

function shortLabel(value: unknown, size = 30) {
  const text = String(value ?? "Unknown");
  return text.length > size ? `${text.slice(0, size - 3)}...` : text;
}

function sourceLayer(value: unknown) {
  const text = String(value ?? "");
  if (text.includes("mart.")) return "Mart";
  if (text.includes("dw.fact") || text.includes("fact_")) return "DW fact";
  if (text.includes("oltp.")) return "OLTP";
  if (text.includes("+")) return "Mixed";
  return "Mixed";
}

function suggestion(row: Row) {
  const endpoint = String(row.endpoint ?? "");
  const source = String(row.source_relation ?? "");
  if (endpoint.includes("decision/replay")) return "materialized replay mart / covering index / cache";
  if (source.includes("dw.fact")) return "index or mart roll-up";
  if (Number(row.avg_rows_returned ?? 0) > 1000) return "pagination / lower limit";
  return "cache hot path";
}

export default function DataEngineering() {
  const health = useApiResource<HealthSummary>("/data/health-summary", undefined, true, false);
  const evidence = useApiResource<Record<string, unknown>>("/data/execution-evidence", undefined, true, false);
  if ((health.loading || evidence.loading) && !health.data) return <LoadingState label="Loading data engineering evidence" />;
  if (health.error) return <ErrorState error={health.error} retry={health.retry} />;
  if (evidence.error) return <ErrorState error={evidence.error} retry={evidence.retry} />;
  const rows = rowsFrom(evidence.data, "query_performance");
  const sourceRows = rowsFrom(evidence.data, "source_usage")
    .sort((a, b) => Number(b.query_count ?? 0) - Number(a.query_count ?? 0))
    .slice(0, 10)
    .map((row) => ({ ...row, source_label: shortLabel(row.source_relation) }));
  const counts = (evidence.data?.row_counts ?? health.data?.row_counts ?? {}) as Record<string, number>;
  const tableRows: Row[] = Object.entries(counts).map(([table_name, row_count]) => ({ table_name, row_count, layer: table_name.includes(".") ? table_name.split(".")[0] : table_name.startsWith("fact_") || table_name.startsWith("dim_") ? "dw" : "oltp/parquet" }));
  const logical = health.data?.logical_counts ?? {};
  const layerRows: Row[] = [
    { layer: "OLTP", row_count: logical.oltp_rows },
    { layer: "DW", row_count: logical.dw_rows },
    { layer: "Mart", row_count: logical.mart_rows },
  ].filter((row) => Number(row.row_count) > 0);
  const totalQueries = rows.reduce((sum, row) => sum + Number(row.query_count ?? 0), 0);
  const avgLatency = rows.reduce((sum, row) => sum + Number(row.avg_elapsed_ms ?? 0), 0) / Math.max(1, rows.length);
  const p95Latency = Math.max(...rows.map((row) => Number(row.p95_elapsed_ms ?? 0)), 0);
  const slowest = evidence.data?.slowest_endpoint as Row | undefined;
  const highestP95 = evidence.data?.highest_p95_endpoint as Row | undefined;
  const fastest = evidence.data?.fastest_endpoint as Row | undefined;
  const mostUsed = evidence.data?.most_used_endpoint as Row | undefined;
  const mostSource = evidence.data?.most_used_source_relation as Row | undefined;
  const optimizationTargets = rows
    .filter((row) => Number(row.p95_elapsed_ms ?? 0) > 250 || Number(row.avg_elapsed_ms ?? 0) > 100 || String(row.endpoint ?? "").includes("decision/replay"))
    .sort((a, b) => Number(b.p95_elapsed_ms ?? 0) - Number(a.p95_elapsed_ms ?? 0))
    .slice(0, 8)
    .map((row) => ({
      endpoint: row.endpoint,
      avg_latency: ms(row.avg_elapsed_ms),
      p95_latency: ms(row.p95_elapsed_ms),
      query_count: row.query_count,
      source_relation: row.source_relation,
      source_layer: sourceLayer(row.source_relation),
      suggested_optimization: suggestion(row),
    }));
  const insight = rows.length ? "Most DSS endpoints are served from Postgres facts/marts with query logging active." : null;

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Data Engineering" question="What execution evidence supports the DSS?" source="OLTP + DW + mart materialized views + query logs" />
        <div className="metric-grid">
          <MetricCard label="Backend active" value={health.data?.backend ?? "No backend"} />
          <MetricCard label="Validation" value={formatText(evidence.data?.validation_status ?? health.data?.validation_passed)} />
          <MetricCard label="Logical rows" value={formatNumber(health.data?.logical_counts?.total_rows, 0)} />
          <MetricCard label="OLTP tables loaded" value={formatNumber(evidence.data?.oltp_tables_loaded, 0)} />
          <MetricCard label="DW facts loaded" value={formatNumber(evidence.data?.dw_facts_loaded, 0)} />
          <MetricCard label="Marts loaded" value={formatNumber(evidence.data?.marts_loaded, 0)} />
          <MetricCard label="Real rows" value={formatNumber(evidence.data?.real_rows ?? health.data?.real_rows, 0)} />
          <MetricCard label="Simulated what-if rows" value={formatNumber(evidence.data?.simulated_whatif_rows ?? health.data?.simulated_rows, 0)} />
          <MetricCard label="Query logs active" value={formatText(evidence.data?.query_logs_active ?? health.data?.query_logs_active)} />
          <MetricCard label="Endpoint groups" value={formatNumber(health.data?.query_log_count, 0)} />
          <MetricCard label="Materialized views" value={formatNumber(evidence.data?.materialized_views_count, 0)} />
          <MetricCard label="Most used source" value={String(mostSource?.source_relation ?? "No source")} detail={formatMetric(mostSource?.query_count, "count")} />
        </div>
      </section>

      {insight ? <section className="panel span-12"><div className="insight-card"><b>Insight</b><span>{insight}</span></div></section> : null}

      <section className="panel span-12">
        <SectionHeader title="Query Performance Summary" question="Aggregated evidence from active DSS endpoint logs." source="oltp.dss_query_log" />
        <div className="metric-grid">
          <MetricCard label="Total queries" value={formatNumber(totalQueries, 0)} />
          <MetricCard label="Avg latency" value={ms(avgLatency)} />
          <MetricCard label="P95 latency" value={ms(p95Latency)} />
          <MetricCard label="Slowest avg endpoint" value={String(slowest?.endpoint ?? "No endpoint")} detail={ms(slowest?.avg_elapsed_ms)} />
          <MetricCard label="Highest p95 endpoint" value={String(highestP95?.endpoint ?? "No endpoint")} detail={ms(highestP95?.p95_elapsed_ms)} />
          <MetricCard label="Fastest endpoint" value={String(fastest?.endpoint ?? "No endpoint")} detail={ms(fastest?.avg_elapsed_ms)} />
          <MetricCard label="Most queried endpoint" value={String(mostUsed?.endpoint ?? "No endpoint")} detail={formatMetric(mostUsed?.query_count, "count")} />
        </div>
      </section>

      <ChartPanel title="Rows by Layer" question="How much data is available in OLTP, DW and mart layers?" source="/data/health-summary" ready={layerRows.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={layerRows}><CartesianGrid stroke="#22303a" /><XAxis dataKey="layer" label={axisLabel("Layer")} /><YAxis label={axisLabel("Rows", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="row_count" fill="#80d8ff" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Latency by Endpoint" question="Which API slices are expensive?" source="oltp.dss_query_log / mart.mv_query_performance" ready={rows.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 24)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="endpoint" hide label={axisLabel("Endpoint")} /><YAxis label={axisLabel("Milliseconds", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="avg_elapsed_ms" fill="#f7c76a" /><Bar dataKey="p95_elapsed_ms" fill="#ff8a7a" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Query Count by Endpoint" question="Which research slices are being used most?" source="oltp.dss_query_log" ready={rows.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 24)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="endpoint" hide label={axisLabel("Endpoint")} /><YAxis label={axisLabel("Count / rows", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="query_count" fill="#80d8ff" /><Bar dataKey="avg_rows_returned" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Source Relation Usage" question="Which facts or marts support active DSS views?" source="oltp.dss_query_log" ready={sourceRows.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={sourceRows} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Query count")} /><YAxis dataKey="source_label" type="category" width={210} label={axisLabel("Source relation", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="query_count" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-12"><SectionHeader title="Optimization Targets" question="Endpoints whose latency profile justifies indexing, marting, pagination or cache attention." source="oltp.dss_query_log" /><DataTable rows={optimizationTargets} columns={["endpoint", "avg_latency", "p95_latency", "query_count", "source_layer", "source_relation", "suggested_optimization"]} pageSize={8} /></section>
      <section className="panel span-6"><SectionHeader title="Layer Row Counts" question="Logical row counts avoid treating empty partition parents as fatal." source="/data/execution-evidence" /><DataTable rows={tableRows} columns={["layer", "table_name", "row_count"]} pageSize={14} /></section>
      <section className="panel span-6"><SectionHeader title="Query Performance Table" question="Endpoint, source relation, materialized-view use, latency and rows returned." source="/data/execution-evidence" /><DataTable rows={rows.map((row) => ({ ...row, source_layer: sourceLayer(row.source_relation) }))} columns={["endpoint", "source_relation", "source_layer", "used_materialized_view", "query_count", "avg_elapsed_ms", "p95_elapsed_ms", "avg_rows_returned", "last_seen_at"]} /></section>
    </div>
  );
}
