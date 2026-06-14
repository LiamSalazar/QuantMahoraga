import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import type { Row } from "../api/types";
import { hasSeries } from "../utils/chartGuards";
import { asNumber, formatMetric, formatNumber, formatPercent } from "../utils/format";
import { pick, rowsFrom } from "../utils/rows";

function pValue(value: unknown) {
  const n = asNumber(value);
  if (n === null) return "";
  return n < 0.001 ? "<0.001" : n.toFixed(3);
}

function statsRow(row: Record<string, unknown>): Row | null {
  const benchmark = row.Benchmark ?? row.benchmark;
  const alpha = row.alpha_ann ?? row.AlphaNW_QQQ ?? row.AlphaNW_SPY;
  const tAlpha = row.t_alpha;
  const pAlpha = row.p_alpha;
  const beta = row.beta;
  const r2 = row.r2;
  if (!benchmark || [alpha, tAlpha, pAlpha, beta, r2].some((value) => asNumber(value) === null)) return null;
  return {
    benchmark: String(benchmark),
    alpha_ann: formatPercent(alpha),
    t_alpha: formatNumber(tAlpha, 2),
    p_alpha: pValue(pAlpha),
    beta: formatNumber(beta, 2),
    r2: formatNumber(r2, 2),
  };
}

export default function BaselineEvidence() {
  const resource = useApiResource<Record<string, unknown>>("/research/baseline-evidence");
  const distributions = useApiResource<Record<string, unknown>>("/research/distributions");
  const data = resource.data ?? {};
  const stitched = rowsFrom(data, "stitched_comparison");
  const official = stitched.find((row) => row.CandidateId === "B1.05_C1.10_L1.10_R1.05") ?? stitched[stitched.length - 1] ?? {};
  const folds = rowsFrom(data, "fold_summary").filter((row) => row.CandidateId === "B1.05_C1.10_L1.10_R1.05");
  const alpha = rowsFrom(data, "alpha_newey_west").filter((row) => row.Variant === "MAHORAGA14_3_BASELINE_OFFICIAL");
  const cost = [...rowsFrom(data, "cost_sensitivity"), ...rowsFrom(data, "slippage_sensitivity")];
  const baseCost = cost[0] ?? {};
  const costDelta = cost.map((row) => ({ ...row, delta_cagr: Number(row.CAGR ?? 0) - Number(baseCost.CAGR ?? 0), delta_sharpe: Number(row.Sharpe ?? 0) - Number(baseCost.Sharpe ?? 0) }));
  const outcomePercentiles = rowsFrom(distributions.data, "outcome_percentiles");
  const decisionPercentiles = rowsFrom(distributions.data, "decision_percentiles");
  const statisticalRows = alpha.map(statsRow).filter((row): row is Row => row !== null);
  const qqqStats = statisticalRows.find((row) => String(row.benchmark).includes("QQQ"));
  const spyStats = statisticalRows.find((row) => String(row.benchmark).includes("SPY"));
  if ((resource.loading || distributions.loading) && !resource.data) return <LoadingState label="Loading official baseline evidence" />;
  if (resource.error) return <ErrorState error={resource.error} retry={resource.retry} />;
  if (distributions.error) return <ErrorState error={distributions.error} retry={distributions.retry} />;

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Baseline Evidence" question="What formal evidence supports the frozen baseline?" source="official baseline outputs, read-only" />
        <div className="metric-grid">
          {["CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "AvgExposure", "AvgTurnover"].map((key) => (
            <MetricCard key={key} label={key} value={formatMetric(pick(official, [key]), key)} detail="official stitched" />
          ))}
        </div>
      </section>
      {qqqStats && spyStats ? (
        <section className="panel span-12">
          <div className="insight-card">
            <b>Statistical edge</b>
            <span>Official Mahoraga shows positive Newey-West alpha vs QQQ and SPY, with beta near {String(qqqStats.beta)} vs QQQ and {String(spyStats.beta)} vs SPY.</span>
          </div>
        </section>
      ) : null}
      <section className="panel span-12">
        <SectionHeader title="Stitched Comparison Table" question="Official vs QQQ, SPY and 14.1 control." source="stitched_comparison_official.csv" />
        <DataTable rows={stitched} columns={["Variant", "GateRole", "CandidateId", "CAGR", "Sharpe", "Sortino", "MaxDD", "BetaQQQ", "BetaSPY", "AlphaNW_QQQ", "AlphaNW_SPY", "AvgExposure"]} />
      </section>
      <ChartPanel title="Fold x Metric Summary" question="Do folds retain positive risk-adjusted behavior?" source="fold_summary_official.csv" ready={folds.length >= 4}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={folds}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="Fold" label={axisLabel("Fold")} />
            <YAxis label={axisLabel("Value", true)} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="Sharpe" fill="#80d8ff" />
            <Bar dataKey="AlphaNW_QQQ" fill="#72f0b1" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Alpha & Beta" question="Is alpha positive after benchmark adjustment?" source="alpha_nw_official.csv" ready={alpha.length >= 2}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={alpha}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="Benchmark" label={axisLabel("Benchmark")} />
            <YAxis label={axisLabel("Alpha / beta", true)} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="alpha_ann" fill="#72f0b1" />
            <Bar dataKey="beta" fill="#f7c76a" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Cost / Slippage CAGR Delta" question="How much CAGR decays when friction rises?" source="cost_sensitivity_official.csv + slippage_sensitivity_official.csv" ready={hasSeries(costDelta, "Scenario", "delta_cagr", 2)}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={costDelta}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="Scenario" label={axisLabel("Scenario")} />
            <YAxis label={axisLabel("Delta CAGR", true)} />
            <Tooltip content={<ChartTooltip />} />
            <Line dataKey="delta_cagr" stroke="#72f0b1" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Cost / Slippage Sharpe Delta" question="Does Sharpe degrade smoothly as trading frictions rise?" source="cost_sensitivity_official.csv + slippage_sensitivity_official.csv" ready={hasSeries(costDelta, "Scenario", "delta_sharpe", 2)}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={costDelta}>
            <CartesianGrid stroke="#22303a" />
            <XAxis dataKey="Scenario" label={axisLabel("Scenario")} />
            <YAxis label={axisLabel("Delta Sharpe", true)} />
            <Tooltip content={<ChartTooltip />} />
            <Line dataKey="delta_sharpe" stroke="#80d8ff" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Outcome Percentiles by Horizon" question="Does performance depend on extreme outcomes or stable horizon distributions?" source="/research/distributions" ready={outcomePercentiles.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={outcomePercentiles}><CartesianGrid stroke="#22303a" /><XAxis dataKey="horizon" label={axisLabel("Horizon")} /><YAxis label={axisLabel("Outcome percentiles", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="p5_outcome" fill="#ff8a7a" /><Bar dataKey="median_outcome" fill="#72f0b1" /><Bar dataKey="p95_outcome" fill="#80d8ff" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-12">
        <SectionHeader title="Statistical and Operating Evidence" question="Newey-West alpha, p/q values, exposure, turnover and return per exposure." source="official outputs + audit CSVs" />
        <DataTable rows={statisticalRows} columns={["benchmark", "alpha_ann", "t_alpha", "p_alpha", "beta", "r2"]} pageSize={10} />
      </section>
      <section className="panel span-12">
        <SectionHeader title="Distribution Percentile Table" question="Postgres aggregations over outcome and decision-state facts." source="dw.fact_outcome + dw.fact_decision_state" />
        <DataTable rows={[...outcomePercentiles, ...decisionPercentiles]} pageSize={10} />
      </section>
    </div>
  );
}
