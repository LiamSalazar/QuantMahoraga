import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options, Row } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { formatDate, formatMetric, formatText } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID, formatCandidateLabel } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

function replaySeed(key: string) {
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

export default function DecisionReplay({ options }: { options: Options | null }) {
  const [fold, setFold] = useState(replaySeed("mahoragaReplayFold") || "all");
  const [date, setDate] = useState(replaySeed("mahoragaReplayDate"));
  const [ticker, setTicker] = useState(replaySeed("mahoragaReplayTicker") || "all");
  const casebook = useApiResource<Record<string, unknown>>("/research/decision-casebook", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold });
  const replay = useApiResource<Record<string, unknown>>("/decision/replay", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold, date, ticker });
  if ((replay.loading || casebook.loading) && !replay.data) return <LoadingState label="Loading decision replay" />;
  if (replay.error) return <ErrorState error={replay.error} retry={replay.retry} />;
  if (casebook.error) return <ErrorState error={casebook.error} retry={casebook.retry} />;

  const data = replay.data ?? {};
  const decision = (data.decision ?? null) as Record<string, unknown> | null;
  const positions = rowsFrom(data, "positions").filter((row) => row.final_weight !== null);
  const availableTickers = rowsFrom(data, "available_tickers");
  const modules = rowsFrom(data, "modules");
  const outcomes = rowsFrom(data, "outcomes").filter((row) => row.horizon !== null && (row.realized_return !== null || row.alpha_vs_qqq !== null));
  const market = rowsFrom(data, "market_context");
  const cases = rowsFrom(casebook.data);
  const tickerOptions = useMemo(() => ["all", ...availableTickers.map((row) => String(row.ticker)).filter(Boolean)], [availableTickers]);

  useEffect(() => {
    if (ticker !== "all" && !tickerOptions.includes(ticker)) setTicker("all");
  }, [ticker, tickerOptions]);

  function loadCase(row: Row) {
    const nextFold = String(row.fold ?? "all");
    const nextDate = String(row.date_value ?? "");
    setFold(nextFold);
    setDate(nextDate);
    setTicker("all");
    sessionStorage.setItem("mahoragaReplayFold", nextFold);
    sessionStorage.setItem("mahoragaReplayDate", nextDate);
    sessionStorage.removeItem("mahoragaReplayTicker");
  }

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Decision Replay" question="Why did Mahoraga take this decision and what happened after?" source="fact_decision_state + fact_position_daily + fact_module_trace + fact_outcome" action={<><SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact /><SelectControl label="Decision date" value={date || "auto"} options={["auto", ...cases.map((row) => String(row.date_value)).filter(Boolean)]} onChange={(value) => setDate(value === "auto" ? "" : value)} compact /><SelectControl label="Ticker" value={ticker} options={tickerOptions} onChange={setTicker} compact /></>} />
        {decision ? (
          <>
            <h3>{formatCandidateLabel(decision.candidate_id)} <small>{formatDate(decision.date_value)}</small></h3>
            <div className="metric-grid">
              {["fold", "regime", "participation_state", "expected_exposure", "expected_turnover", "drawdown", "hard_backoff_flag", "leader_blend"].map((key) => <MetricCard key={key} label={key.replaceAll("_", " ")} value={formatMetric(decision[key], key)} />)}
            </div>
          </>
        ) : <EmptyState title="No decision found" detail="Reset filters or choose a broader fold slice." />}
      </section>

      <section className="panel span-12">
        <SectionHeader title="Decision Casebook" question="Curated replay cases with positions, module traces and outcomes available." source="/research/decision-casebook" />
        <div className="casebook-grid">
          {cases.map((row) => (
            <article className="casebook-card" key={`${row.case_label}-${row.date_value}-${row.fold}`}>
              <b>{formatText(row.case_label)}</b>
              <span>{formatText(row.rationale)}</span>
              <small>{formatDate(row.date_value)} · fold {formatText(row.fold)} · 20d {formatMetric(row.return_20d, "return")}</small>
              <button className="ghost-button" onClick={() => loadCase(row)}>Load replay</button>
            </article>
          ))}
        </div>
      </section>

      <section className="panel span-12">
        <SectionHeader title="Decision Path Summary" question="Auditable transformation from market context to final exposure and outcomes." source="DSS semantic flow" />
        <div className="stepper">{["Market context", "Signals & rank", "Selected names", "Base weights", "Overlay modules", "Risk/backoff", "Final exposure", "Outcomes"].map((step) => <span key={step}>{step}</span>)}</div>
      </section>

      <ChartPanel title="Portfolio Weights" question="Which tickers drove the final allocation?" source="fact_position_daily" ready={positions.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={positions.slice(0, 18)} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Final weight")} /><YAxis dataKey="ticker" type="category" width={70} label={axisLabel("Ticker", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="final_weight" fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Outcome Strip" question="Which forward horizons have real outcomes for this decision?" source="fact_outcome" ready={outcomes.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={outcomes}><CartesianGrid stroke="#22303a" /><XAxis dataKey="horizon" label={axisLabel("Horizon")} /><YAxis label={axisLabel("Return / alpha", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="realized_return" fill="#72f0b1" /><Bar dataKey="alpha_vs_qqq" fill="#80d8ff" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <section className="panel span-6"><SectionHeader title="Module Trace" question="Which overlays were active and with what effect?" source="fact_module_trace" /><DataTable rows={modules} columns={["module_name", "module_active", "intensity_score", "effect_on_exposure", "state_label", "raw_value"]} /></section>
      <section className="panel span-6"><SectionHeader title="Outcomes and Market Context" question="Only horizons and market rows with materialized values are shown." source="fact_outcome + fact_market_bar" /><DataTable rows={[...outcomes, ...market]} /></section>
    </div>
  );
}
