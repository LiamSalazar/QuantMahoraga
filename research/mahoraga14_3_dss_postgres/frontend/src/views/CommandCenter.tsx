import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { Options, Row, ViewKey } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasScatter, hasSeries } from "../utils/chartGuards";
import { asNumber, formatMetric, formatNumber } from "../utils/format";
import { formatCandidateLabel, formatDemoMode, formatPercent, OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { pick, rowsFrom, topRows } from "../utils/rows";

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
    role: row.research_role ?? row.sweep_role,
    demo_mode: row.demo_mode,
  };
}

function bestWorst(rows: Row[], key: string) {
  return { best: topRows(rows, key, 1)[0], worst: topRows(rows, key, 1, false)[0] };
}

export default function CommandCenter({ options, onOpenView }: { options: Options | null; onOpenView: (view: ViewKey) => void }) {
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
  const official = scorecard.find((row) => row.candidate_id === OFFICIAL_CANDIDATE_ID) ?? scorecard[0] ?? {};
  const triad = rowsFrom(data.best_official_worst, "rows").map(normalizeMetricRow);
  const equity = rowsFrom(overview, "equity_curve").filter((row) => row.equity !== null && row.drawdown !== null);
  const folds = rowsFrom(overview, "fold_performance").map((row) => ({ ...row, metric_value: row.avg_alpha_vs_qqq ?? row.avg_realized_return })).filter((row) => row.metric_value !== null && row.metric_value !== undefined);
  const comparison = rowsFrom(data, "baseline_comparison");
  const winsDrags = (data.top_wins_drags ?? {}) as Record<string, unknown>;
  const tickerBW = bestWorst(rowsFrom(winsDrags, "tickers"), "total_pnl_contribution");
  const foldBW = bestWorst(rowsFrom(winsDrags, "folds"), "avg_alpha_vs_qqq");
  const moduleBW = bestWorst(rowsFrom(winsDrags, "modules"), "helped_rate");
  const regimeBW = bestWorst(rowsFrom(winsDrags, "regimes"), "avg_net_return");
  const candidateBW = bestWorst(rowsFrom(winsDrags, "candidates"), "sharpe");
  const qqq = comparison.find((row) => String(row.Variant).includes("QQQ"));
  const insights = [
    asNumber(pick(official, ["cagr", "CAGR"])) !== null && asNumber(pick(qqq, ["CAGR"])) !== null && Number(pick(official, ["cagr", "CAGR"])) > Number(pick(qqq, ["CAGR"]))
      ? "Official Mahoraga improves CAGR over QQQ in the stitched comparison."
      : null,
    tickerBW.best ? `Top positive ticker in this slice is ${tickerBW.best.ticker}.` : null,
    moduleBW.best ? `${moduleBW.best.module_name} has the strongest helped-rate evidence by horizon.` : null,
  ].filter(Boolean);

  const highlights: Array<{ title: string; detail: string; view: ViewKey }> = [
    { title: "Compare official vs best/worst observed", detail: "Audit why the frozen baseline is not automatically replaced by a higher-scoring sweep point.", view: "robustness" },
    { title: "Audit a rich decision", detail: "Open a decision with positions, module trace and outcomes already materialized.", view: "replay" },
    { title: "Inspect module contribution", detail: "See helped rate, activation and horizon effects by overlay module.", view: "modules" },
    { title: "Explore mining questions", detail: "Run guided OLAP presets with tables, charts and drill-through actions.", view: "olap" },
    { title: "Check data evidence", detail: "Review row origin, query logs, marts and execution evidence.", view: "engineering" },
  ];

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
            <span>Query logs: {health.query_logs_active ? "active" : "warming up"}</span>
          </div>
        </div>
      </section>

      <div className="metric-grid span-12">
        {metricKeys.map(([label, keys]) => <MetricCard key={label} label={label} value={formatMetric(pick(official, keys), label)} detail="official baseline" />)}
      </div>

      {insights.length ? (
        <section className="panel span-12">
          <SectionHeader title="Command Insights" question="Short conclusions generated from the current evidence slice." source="current DSS aggregates" />
          <div className="insight-strip">{insights.map((text) => <article className="insight-card" key={text}><b>Insight</b><span>{text}</span></article>)}</div>
        </section>
      ) : null}

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
        <ResponsiveContainer width="100%" height="100%"><AreaChart data={equity}><defs><linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#72f0b1" stopOpacity={0.35} /><stop offset="100%" stopColor="#72f0b1" stopOpacity={0.02} /></linearGradient></defs><CartesianGrid stroke="#22303a" /><XAxis dataKey="date_value" minTickGap={42} label={axisLabel("Date")} /><YAxis label={axisLabel("Cumulative value", true)} /><Tooltip content={<ChartTooltip />} /><Area dataKey="equity" stroke="#72f0b1" fill="url(#equityFill)" strokeWidth={2} /></AreaChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Drawdown Replay" question="Where did path risk concentrate?" source="mart.mv_drawdown_replay" ready={hasSeries(equity, "date_value", "drawdown")} emptyDetail="Drawdown series is not available for this slice.">
        <ResponsiveContainer width="100%" height="100%"><AreaChart data={equity}><CartesianGrid stroke="#22303a" /><XAxis dataKey="date_value" minTickGap={42} label={axisLabel("Date")} /><YAxis label={axisLabel("Drawdown", true)} /><Tooltip content={<ChartTooltip />} /><Area dataKey="drawdown" stroke="#ff8a7a" fill="#ff8a7a33" strokeWidth={2} /></AreaChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Fold Evidence" question="Is the official point stable across walk-forward folds?" source="mart.mv_performance_by_fold" ready={folds.length >= 4}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={folds.slice(0, 40)}><CartesianGrid stroke="#22303a" /><XAxis dataKey="fold" label={axisLabel("Fold")} /><YAxis label={axisLabel("Alpha / return", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="metric_value" fill="#80d8ff" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Mini Pareto" question="What is the trade-off between return and drawdown for best/official/worst?" source="extended_multiplier_summary.csv" ready={hasScatter(triad, "maxdd", "cagr", 3)}>
        <ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid stroke="#22303a" /><XAxis dataKey="maxdd" name="MaxDD" type="number" label={axisLabel("Max drawdown")} /><YAxis dataKey="cagr" name="CAGR" type="number" label={axisLabel("CAGR", true)} /><ZAxis dataKey="sharpe" range={[80, 260]} /><Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ChartTooltip />} /><Scatter data={triad}>{triad.map((row, index) => <Cell key={index} fill={row.candidate_id === OFFICIAL_CANDIDATE_ID ? "#f7c76a" : asNumber(row.cagr) && Number(row.cagr) > 30 ? "#72f0b1" : "#ff8a7a"} />)}</Scatter></ScatterChart></ResponsiveContainer>
      </ChartPanel>

      <section className="panel span-12">
        <SectionHeader title="Top Wins / Top Drags" question="Compact data-mined anchors for further investigation." source="marts over folds, tickers, modules, regimes and candidates" />
        <div className="metric-grid">
          <MetricCard label="Best fold" value={formatMetric(foldBW.best?.fold, "fold")} detail={formatMetric(foldBW.best?.avg_alpha_vs_qqq, "alpha")} />
          <MetricCard label="Best ticker contributor" value={String(tickerBW.best?.ticker ?? "—")} detail={formatMetric(tickerBW.best?.total_pnl_contribution, "return")} />
          <MetricCard label="Best module/horizon" value={String(moduleBW.best?.module_name ?? "—")} detail={`H${String(moduleBW.best?.horizon ?? "—")} · ${formatMetric(moduleBW.best?.helped_rate, "rate")}`} />
          <MetricCard label="Best observed candidate" value={formatCandidateLabel(candidateBW.best?.candidate_id)} detail={formatMetric(candidateBW.best?.sharpe, "Sharpe")} />
          <MetricCard label="Worst fold" value={formatMetric(foldBW.worst?.fold, "fold")} detail={formatMetric(foldBW.worst?.avg_alpha_vs_qqq, "alpha")} />
          <MetricCard label="Largest drag ticker" value={String(tickerBW.worst?.ticker ?? "—")} detail={formatMetric(tickerBW.worst?.total_pnl_contribution, "return")} />
          <MetricCard label="Weakest regime" value={String(regimeBW.worst?.regime ?? "—")} detail={formatMetric(regimeBW.worst?.avg_net_return, "return")} />
          <MetricCard label="Worst observed candidate" value={formatCandidateLabel(candidateBW.worst?.candidate_id)} detail={formatMetric(candidateBW.worst?.sharpe, "Sharpe")} />
        </div>
      </section>

      <section className="panel span-12">
        <SectionHeader title="Quick Benchmark Comparison" question="How does official Mahoraga compare with QQQ, SPY and 14.1 control?" source="stitched_comparison_official.csv" />
        <DataTable rows={comparison} columns={["Variant", "GateRole", "CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "AvgExposure"]} />
      </section>

      <section className="panel span-12">
        <SectionHeader title="Research Workbench Highlights" question="Fast paths into auditable research workflows." source="DSS navigation" />
        <div className="workbench-grid">
          {highlights.map((item) => (
            <article className="workbench-card" key={item.title}>
              <b>{item.title}</b>
              <span>{item.detail}</span>
              <button className="ghost-button" onClick={() => onOpenView(item.view)}>Open</button>
            </article>
          ))}
        </div>
      </section>

      {!comparison.length ? <EmptyState /> : null}
    </div>
  );
}
