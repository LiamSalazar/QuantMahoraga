import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasSeries } from "../utils/chartGuards";
import { formatMetric, formatText } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID, formatCandidateLabel } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

export default function DecisionReplay({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("all");
  const [ticker, setTicker] = useState("all");
  const replay = useApiResource<Record<string, unknown>>("/decision/replay", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold, ticker });
  if (replay.loading && !replay.data) return <LoadingState label="Loading decision replay" />;
  if (replay.error) return <ErrorState error={replay.error} retry={replay.retry} />;
  const data = replay.data ?? {};
  const decision = (data.decision ?? null) as Record<string, unknown> | null;
  const positions = rowsFrom(data, "positions");
  const modules = rowsFrom(data, "modules");
  const outcomes = rowsFrom(data, "outcomes");
  const market = rowsFrom(data, "market_context");
  const timeline = rowsFrom(data, "timeline");

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Decision Replay" question="Why did Mahoraga take this decision and what happened after?" source="fact_decision_state + fact_position_daily + fact_module_trace + fact_outcome" action={<><SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact /><SelectControl label="Ticker" value={ticker} options={["all", ...(options?.tickers ?? [])]} onChange={setTicker} compact /></>} />
        {decision ? (
          <>
            <h3>{formatCandidateLabel(decision.candidate_id)} <small>{formatText(decision.date_value)}</small></h3>
            <div className="metric-grid">
              {["fold", "regime", "participation_state", "expected_exposure", "expected_turnover", "drawdown", "backoff_strength", "leader_blend"].map((key) => <MetricCard key={key} label={key} value={formatMetric(decision[key], key)} />)}
            </div>
          </>
        ) : <EmptyState title="No decision found" detail="Reset filters or choose a broader fold/ticker slice." />}
      </section>
      <section className="panel span-12">
        <SectionHeader title="Decision Flow Stepper" question="Auditable transformation from market context to outcomes." source="DSS semantic flow" />
        <div className="stepper">{["Market context", "Signals & rank", "Selected names", "Base weights", "Overlay modules", "Risk/backoff", "Final exposure", "Outcomes"].map((step) => <span key={step}>{step}</span>)}</div>
      </section>
      <ChartPanel title="Portfolio Weights" question="Which tickers drove the final allocation?" source="fact_position_daily" ready={positions.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={positions.slice(0, 18)} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" /><YAxis dataKey="ticker" type="category" width={70} /><Tooltip /><Bar dataKey="final_weight" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <ChartPanel title="Decision Timeline" question="Was a path timeline materialized for this decision?" source="fact_decision_state" ready={hasSeries(timeline, "date_value", "drawdown", 10)} emptyDetail="Timeline not materialized for this decision. Showing decision trace, weights, modules and outcomes.">
        <ResponsiveContainer width="100%" height="100%"><LineChart data={timeline}><CartesianGrid stroke="#22303a" /><XAxis dataKey="date_value" minTickGap={34} /><YAxis /><Tooltip /><Line dataKey="expected_exposure" stroke="#80d8ff" /><Line dataKey="drawdown" stroke="#ff8a7a" /></LineChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-6"><SectionHeader title="Module Trace" question="Which overlays were active and with what effect?" source="fact_module_trace" /><DataTable rows={modules} columns={["module_name", "module_active", "intensity_score", "effect_on_exposure", "state_label", "raw_value"]} /></section>
      <section className="panel span-6"><SectionHeader title="Outcomes and Market Context" question="What happened over 1d/5d/20d/60d horizons?" source="fact_outcome + fact_market_bar" /><DataTable rows={[...outcomes, ...market]} /></section>
    </div>
  );
}
