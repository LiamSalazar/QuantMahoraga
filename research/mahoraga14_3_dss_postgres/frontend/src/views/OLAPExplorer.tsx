import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasSeries } from "../utils/chartGuards";
import { formatMetric } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

const presets = [
  { id: "alpha-fold", label: "Alpha by fold", question: "Where is benchmark-adjusted return strongest?", operation: "roll-up", dimensions: ["fold"], measure: "alpha", facts: "fact_outcome" },
  { id: "outcome-horizon", label: "Outcome by horizon", question: "How do forward outcomes change by horizon?", operation: "roll-up", dimensions: ["horizon"], measure: "return", facts: "fact_outcome" },
  { id: "contribution-ticker", label: "Contribution by ticker", question: "Which tickers contribute most?", operation: "drill-down", dimensions: ["ticker"], measure: "return", facts: "fact_position_daily" },
  { id: "helped-module", label: "Helped rate by module", question: "Which modules activate most?", operation: "roll-up", dimensions: ["module_name"], measure: "helped_rate", facts: "fact_module_trace" },
  { id: "exposure-regime", label: "Exposure by regime", question: "Where does exposure concentrate?", operation: "slice", dimensions: ["regime"], measure: "exposure", facts: "fact_decision_state" },
  { id: "drawdown-state", label: "Drawdown by participation state", question: "Which participation states carry drawdown?", operation: "dice", dimensions: ["participation_state"], measure: "drawdown", facts: "fact_decision_state" },
  { id: "candidate-universe", label: "Candidate vs universe", question: "How does performance vary by universe?", operation: "pivot", dimensions: ["candidate_id", "universe_id"], measure: "alpha", facts: "fact_outcome" },
  { id: "backoff-regime", label: "Backoff effect by regime", question: "Where does risk backoff activate?", operation: "slice", dimensions: ["regime"], measure: "turnover", facts: "fact_decision_state" },
  { id: "worst-horizon", label: "Worst decisions by horizon", question: "Where are weak outcomes clustered?", operation: "drill-down", dimensions: ["horizon", "fold"], measure: "return", facts: "fact_outcome" },
  { id: "robust-ranking", label: "Candidate robustness ranking", question: "Which candidates rank by alpha?", operation: "roll-up", dimensions: ["candidate_id"], measure: "alpha", facts: "fact_outcome" },
] as const;

export default function OLAPExplorer({ options }: { options: Options | null }) {
  const [presetId, setPresetId] = useState<string>(presets[0].id);
  const [fold, setFold] = useState("all");
  const preset = useMemo(() => presets.find((item) => item.id === presetId) ?? presets[0], [presetId]);
  const data = useApiResource<Record<string, unknown>>("/slice", { dimensions: preset.dimensions, measure: preset.measure, operation: preset.operation, candidate_id: preset.id === "candidate-universe" || preset.id === "robust-ranking" ? undefined : OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold, limit: 500 });
  if (data.loading && !data.data) return <LoadingState label="Running OLAP preset" />;
  if (data.error) return <ErrorState error={data.error} retry={data.retry} />;
  const rows = rowsFrom(data.data);
  const xKey = preset.dimensions[0];
  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="OLAP Explorer" question="Guided cube operations over Mahoraga facts and marts." source={preset.facts} action={<><SelectControl label="Preset" value={presetId} options={presets.map((item) => item.id)} onChange={setPresetId} /><SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact /></>} />
        <div className="metric-grid">
          <MetricCard label="Question" value={preset.label} detail={preset.question} />
          <MetricCard label="OLAP operation" value={preset.operation} detail={preset.dimensions.join(" / ")} />
          <MetricCard label="Facts/marts" value={preset.facts} detail={preset.measure} />
          <MetricCard label="Rows" value={formatMetric(rows.length, "rows")} detail="current result" />
        </div>
      </section>
      <ChartPanel title={preset.label} question={preset.question} source={preset.facts} ready={hasSeries(rows, xKey, preset.measure, 2)} emptyDetail="Single-value result. Use additional dimensions or broaden filters for a useful chart.">
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 40)}><CartesianGrid stroke="#22303a" /><XAxis dataKey={xKey} /><YAxis /><Tooltip /><Bar dataKey={preset.measure} fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-6"><SectionHeader title="Pivot Table" question="Auditable rows returned by the guided OLAP preset." source={String((data.data ?? {}).table ?? preset.facts)} /><DataTable rows={rows} /></section>
    </div>
  );
}
