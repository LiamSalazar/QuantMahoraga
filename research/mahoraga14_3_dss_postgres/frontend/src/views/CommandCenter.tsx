import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasHeatmap, hasScatter, hasSeries } from "../utils/chartGuards";
import { asNumber, formatMetric, formatNumber } from "../utils/format";
import { formatCandidateLabel, formatDemoMode, formatPercent, OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { pick, rowsFrom } from "../utils/rows";

const metricKeys: [string, string[]][] = [
  ["CAGR", ["cagr", "CAGR"]],
  ["Sharpe", ["sharpe", "Sharpe"]],
  ["Sortino", ["sortino", "Sortino"]],
  ["MaxDD", ["maxdd", "MaxDD"]],
  ["Alpha vs QQQ", ["alpha_qqq", "AlphaNW_QQQ"]],
  ["Alpha vs SPY", ["alpha_spy", "AlphaNW_SPY"]],
  ["Avg exposure", ["avg_exposure", "AvgExposure"]],
  ["Avg turnover", ["avg_turnover", "AvgTurnover"]],
  ["Return / exposure", ["return_per_exposure", "ReturnPerExposure"]],
];

function normalizeMetricRow(row: Row): Row {
  return {
    candidate_id: pick(row, ["candidate_id", "CandidateId"]),
    cagr: pick(row, ["cagr", "CAGR"]),
    sharpe: pick(row, ["sharpe", "Sharpe"]),
    maxdd: pick(row, ["maxdd", "MaxDD"]),
    robust_score: pick(row, ["robust_score", "robust_region_flag"]),
    role: row.research_role ?? row.sweep_role,
    demo_mode: row.demo_mode,
  };
}

export default function CommandCenter({ options }: { options: Options | null }) {
  const resource = useApiResource<Record<string, unknown>>("/research/command-center", {
    candidate_id: options?.default_candidate ?? OFFICIAL_CANDIDATE_ID,
    universe_id: options?.default_universe ?? "base_universe_12",
  });
  if (resource.loading && !resource.data) return <LoadingState label="Loading command center evidence" />;
  if (resource.error) return <ErrorState error={resource.error} retry={resource.retry} />;

  const data = resource.data ?? {};
  const overview = (data.overview ?? {}) as Record<string, unknown>;
  const health = (data.health ?? {}) as Record<string, unknown>;
  const scorecard = rowsFrom(overview, "scorecard");
  const official = scorecard[0] ?? rowsFrom(data.best_official_worst, "rows").find((row) => pick(row, ["candidate_id", "CandidateId"]) === OFFICIAL_CANDIDATE_ID) ?? {};
  const triad = rowsFrom(data.best_official_worst, "rows").map(normalizeMetricRow);
  const equity = rowsFrom(overview, "equity_curve");
  const folds = rowsFrom(overview, "fold_performance").map((row) => ({ ...row, metric_value: row.avg_alpha_vs_qqq ?? row.avg_realized_return }));
  const comparison = rowsFrom(data, "baseline_comparison");
  const questions = rowsFrom(data, "research_questions");

  return (
    <div className="view-grid">
      <section className="hero-card span-12">
        <div>
          <span className="eyebrow">Official frozen point</span>
          <h2>{formatCandidateLabel(OFFICIAL_CANDIDATE_ID)}</h2>
          <p>B1.05_C1.10_L1.10_R1.05 · Universe: base_universe_12 · Frozen · promoted · audited</p>
          <div className="chips">
            <span>Backend: {String(health.backend ?? "Postgres")}</span>
            <span>{String((data.identity as Row | undefined)?.data_badge ?? "Postgres · audited artifacts + flagged simulated what-if")}</span>
            <span>Real rows: {formatNumber(health.real_rows, 0)}</span>
            <span>Simulated what-if rows: {formatNumber(health.simulated_rows, 0)}</span>
            <span>Marts: {formatNumber((health.marts_available as unknown[] | undefined)?.length, 0)}</span>
            <span>Query logs: {health.query_logs_active ? "active" : "warming up"}</span>
          </div>
        </div>
      </section>

      <div className="metric-grid span-12">
        {metricKeys.map(([label, keys]) => (
          <MetricCard key={label} label={label} value={formatMetric(pick(official, keys), label)} detail="official baseline" />
        ))}
      </div>

      <section className="panel span-12">
        <SectionHeader title="Best / Official / Worst Observed" question="Where does the frozen point sit in the audited sweep?" source="extended_multiplier_summary.csv" />
        <div className="triad-grid">
          {triad.map((row) => (
            <article className="candidate-card" key={String(row.candidate_id)}>
              <span>{String(row.role ?? "Observed candidate")}</span>
              <h3>{formatCandidateLabel(row.candidate_id)}</h3>
              <small>{String(row.candidate_id)}</small>
              <div className="mini-metrics">
                <b>CAGR {formatPercent(row.cagr)}</b>
                <b>Sharpe {formatNumber(row.sharpe, 3)}</b>
                <b>MaxDD {formatPercent(row.maxdd)}</b>
                <b>{formatDemoMode(row.demo_mode)}</b>
              </div>
            </article>
          ))}
        </div>
      </section>

      <ChartPanel title="Equity Curve" question="How did cumulative value evolve across the official path?" source="mart.mv_drawdown_replay" ready={hasSeries(equity, "date_value", "equity")} emptyDetail="Equity series is not materialized for this slice.">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={equity}>
            <defs>
              <linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#72f0b1" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#72f0b1" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="date_value" minTickGap={42} />
            <YAxis />
            <Tooltip />
            <Area dataKey="equity" stroke="#72f0b1" fill="url(#equityFill)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Drawdown Replay" question="Where did path risk concentrate?" source="mart.mv_drawdown_replay" ready={hasSeries(equity, "date_value", "drawdown")} emptyDetail="Drawdown series is not available for this slice.">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={equity}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="date_value" minTickGap={42} />
            <YAxis />
            <Tooltip />
            <Area dataKey="drawdown" stroke="#ff8a7a" fill="#ff8a7a33" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Fold Evidence" question="Is the official point stable across walk-forward folds?" source="mart.mv_performance_by_fold" ready={hasHeatmap(folds, "fold", "horizon", "metric_value") || folds.length >= 4}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={folds.slice(0, 40)}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="fold" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="metric_value" fill="#80d8ff" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Mini Pareto" question="What is the trade-off between return and drawdown for best/official/worst?" source="extended_multiplier_summary.csv" ready={hasScatter(triad, "maxdd", "cagr", 3)}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="maxdd" name="MaxDD" type="number" />
            <YAxis dataKey="cagr" name="CAGR" type="number" />
            <ZAxis dataKey="sharpe" range={[80, 260]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={triad}>
              {triad.map((row, index) => (
                <Cell key={index} fill={row.candidate_id === OFFICIAL_CANDIDATE_ID ? "#f7c76a" : asNumber(row.cagr) && Number(row.cagr) > 30 ? "#72f0b1" : "#ff8a7a"} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </ChartPanel>

      <section className="panel span-12">
        <SectionHeader title="Quick Benchmark Comparison" question="How does official Mahoraga compare with QQQ, SPY and 14.1 control?" source="stitched_comparison_official.csv" />
        <DataTable rows={comparison} columns={["Variant", "GateRole", "CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "AvgExposure"]} />
      </section>

      <section className="panel span-12">
        <SectionHeader title="Research Questions Answered" question="Each card maps a user question to an audited fact/mart and OLAP operation." source="metadata/questions registry" />
        <div className="question-grid">
          {questions.length ? questions.map((row) => (
            <article key={String(row.id)} className="question-card">
              <h3>{String(row.question)}</h3>
              <span>{String(row.endpoint)}</span>
              <small>{Array.isArray(row.operations) ? row.operations.join(" · ") : String(row.operations ?? "slice")}</small>
              <small>{Array.isArray(row.facts) ? row.facts.join(", ") : String(row.facts ?? "facts")}</small>
            </article>
          )) : <EmptyState />}
        </div>
      </section>
    </div>
  );
}
