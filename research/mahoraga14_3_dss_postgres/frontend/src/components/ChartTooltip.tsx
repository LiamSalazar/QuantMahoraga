import { formatCandidateLabel } from "../utils/labels";
import { formatDate, formatMetric, formatText } from "../utils/format";

const labelMap: Record<string, string> = {
  avg_alpha_vs_qqq: "Alpha vs QQQ",
  avg_benchmark_return: "Benchmark return",
  avg_drawdown: "Drawdown",
  avg_elapsed_ms: "Avg latency",
  avg_exposure: "Avg exposure",
  avg_final_weight: "Avg final weight",
  avg_net_return: "Net return",
  avg_realized_return: "Realized return",
  avg_rows_returned: "Rows returned",
  backoff_activation_rate: "Backoff activation",
  budget_multiplier: "Budget",
  cagr: "CAGR",
  CAGR: "CAGR",
  conviction_multiplier: "Conviction",
  drawdown: "Drawdown",
  equity: "Cumulative value",
  expected_exposure: "Expected exposure",
  final_weight: "Final weight",
  helped_rate: "Helped rate",
  leader_flag_rate: "Leader rate",
  leader_multiplier: "Leader",
  maxdd: "Max drawdown",
  MaxDD: "Max drawdown",
  metric_value: "Metric value",
  p95_elapsed_ms: "P95 latency",
  query_count: "Query count",
  robust_score: "Robust score",
  selection_rate: "Selection rate",
  sharpe: "Sharpe",
  Sharpe: "Sharpe",
  Sortino: "Sortino",
  total_pnl_contribution: "Contribution",
};

export function metricLabel(key: unknown): string {
  const text = String(key ?? "");
  return labelMap[text] ?? text.replaceAll("_", " ");
}

function formatLabel(label: unknown, payloadRow?: Record<string, unknown>): string {
  const candidate = payloadRow?.candidate_id ?? payloadRow?.CandidateId;
  if (candidate) return formatCandidateLabel(candidate);
  if (typeof label === "string" && /^\d{4}-\d{2}-\d{2}/.test(label)) return formatDate(label);
  return formatText(label);
}

export function axisLabel(value: string, vertical = false) {
  return {
    value,
    angle: vertical ? -90 : 0,
    position: vertical ? "insideLeft" : "insideBottom",
    offset: vertical ? 8 : -2,
    fill: "#91a0a8",
    fontSize: 12,
  } as const;
}

export function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<Record<string, unknown>>; label?: unknown }) {
  if (!active || !payload?.length) return null;
  const first = payload[0];
  const row = (first.payload ?? {}) as Record<string, unknown>;
  const entries = payload.filter((item) => item.value !== null && item.value !== undefined && Number.isFinite(Number(item.value)));
  return (
    <div className="chart-tooltip">
      <strong>{formatLabel(label, row)}</strong>
      {row.ticker ? <span>Ticker: {formatText(row.ticker)}</span> : null}
      {row.candidate_id || row.CandidateId ? <small>{formatText(row.candidate_id ?? row.CandidateId)}</small> : null}
      {entries.map((item) => {
        const key = String(item.dataKey ?? item.name ?? "");
        return (
          <span key={key}>
            {metricLabel(key)}: <b>{formatMetric(item.value, key)}</b>
          </span>
        );
      })}
      {row.demo_mode !== undefined ? <em>{row.demo_mode ? "Simulated what-if · not official performance" : "Observed audit scenario"}</em> : null}
    </div>
  );
}
