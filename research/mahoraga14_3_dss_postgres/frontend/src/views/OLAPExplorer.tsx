import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options, Row, ViewKey } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { hasSeries } from "../utils/chartGuards";
import { formatMetric, formatText } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

const presets = [
  ["fold-best-performance", "Which fold contributes most to official performance?", "performance", "robustness"],
  ["fold-worst-drawdown", "Which fold carries the worst drawdown?", "performance", "baseline"],
  ["sharpe-stable-folds", "Is Sharpe stable across folds?", "performance", "baseline"],
  ["candidate-cagr-maxdd", "Which candidate has the best CAGR/MaxDD tradeoff?", "robustness", "robustness"],
  ["candidate-best-sharpe", "Which candidate has the best Sharpe?", "robustness", "robustness"],
  ["candidate-severe-fold-damage", "Which candidate has severe fold damage?", "robustness", "robustness"],
  ["axis-degrades-most", "Which multiplier axis degrades the model most?", "robustness", "robustness"],
  ["module-helps-horizon", "Which module helps most by horizon?", "modules", "modules"],
  ["module-active-low-value", "Which module activates often but adds little?", "modules", "modules"],
  ["ticker-top-contribution", "Which tickers contribute most?", "tickers", "tickers"],
  ["ticker-largest-drags", "Which tickers drag most?", "tickers", "tickers"],
  ["ticker-frequent-leaders", "Which tickers are frequent leaders?", "tickers", "tickers"],
  ["regime-best-alpha", "Which regime has the best alpha proxy?", "regime", "regimes"],
  ["regime-exposure-concentration", "Where is exposure concentrated?", "regime", "regimes"],
  ["regime-backoff-most", "Where does backoff activate most?", "regime", "regimes"],
  ["decision-best-20d", "Best decisions by 20d outcome.", "decisions", "replay"],
  ["decision-worst-20d", "Worst decisions by 20d outcome.", "decisions", "replay"],
  ["engineering-slowest-endpoint", "Which endpoint is slowest?", "data engineering", "engineering"],
  ["engineering-source-most-used", "Which source relation is used most?", "data engineering", "engineering"],
] as const;

function initialPreset() {
  try {
    return sessionStorage.getItem("mahoragaOlapPreset") ?? presets[0][0];
  } catch {
    return presets[0][0];
  }
}

export default function OLAPExplorer({ options, onOpenView }: { options: Options | null; onOpenView: (view: ViewKey) => void }) {
  const [presetId, setPresetId] = useState<string>(initialPreset());
  const [fold, setFold] = useState("all");
  const preset = useMemo(() => presets.find((item) => item[0] === presetId) ?? presets[0], [presetId]);
  const data = useApiResource<Record<string, unknown>>("/research/olap-preset", { preset_id: preset[0], candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold, limit: 500 });
  if (data.loading && !data.data) return <LoadingState label="Running mining question" />;
  if (data.error) return <ErrorState error={data.error} retry={data.retry} />;
  const rows = rowsFrom(data.data);
  const payload = data.data ?? {};
  const xKey = String(payload.dimension ?? Object.keys(rows[0] ?? {})[0] ?? "dimension");
  const yKey = String(payload.measure ?? Object.keys(rows[0] ?? {})[1] ?? "value");
  const chartReady = hasSeries(rows, xKey, yKey, 2);
  const top = rows[0];
  const insight = top ? `${formatText(top[xKey])} ranks first by ${String(yKey).replaceAll("_", " ")} in this mining question.` : null;
  const drillView = preset[3] as ViewKey;
  function drill(row?: Row) {
    if (row?.date_value) {
      sessionStorage.setItem("mahoragaReplayDate", String(row.date_value));
      if (row.fold) sessionStorage.setItem("mahoragaReplayFold", String(row.fold));
    }
    onOpenView(drillView);
  }

  return (
    <div className="view-grid">
      <section className="panel span-12">
        <SectionHeader title="Mining Questions Workbench" question="Guided OLAP presets over audited Mahoraga facts and marts." source={String(payload.source ?? "guided DSS marts")} action={<><SelectControl label="Preset" value={presetId} options={presets.map((item) => item[0])} onChange={(value) => { sessionStorage.setItem("mahoragaOlapPreset", value); setPresetId(value); }} /><SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact /></>} />
        <div className="metric-grid">
          <MetricCard label="Question" value={preset[2]} detail={String(payload.question ?? preset[1])} />
          <MetricCard label="Operation" value={String(payload.operation ?? "slice")} detail="OLAP operation" />
          <MetricCard label="Rows" value={formatMetric(rows.length, "rows")} detail="valid result rows" />
          <MetricCard label="Next action" value={`Open ${drillView}`} detail="drill-through available" />
        </div>
      </section>

      {insight ? <section className="panel span-12"><div className="insight-card"><b>Insight</b><span>{insight}</span></div></section> : null}

      <ChartPanel title={String(payload.question ?? preset[1])} question={`Operation: ${String(payload.operation ?? "slice")}`} source={String(payload.source ?? "guided marts")} ready={chartReady} emptyDetail="Single-value result. Showing KPI and table instead of a low-value chart.">
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows.slice(0, 40)}><CartesianGrid stroke="#22303a" /><XAxis dataKey={xKey} label={axisLabel(xKey.replaceAll("_", " "))} /><YAxis label={axisLabel(yKey.replaceAll("_", " "), true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey={yKey} fill="#72f0b1" /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <section className="panel span-6">
        <SectionHeader title="Result Table" question="Auditable rows returned by the selected mining question." source={String(payload.source ?? "guided marts")} action={<button className="ghost-button" onClick={() => drill(top)}>Open in {drillView}</button>} />
        <DataTable rows={rows} rowAction={(row) => <button className="ghost-button" onClick={() => drill(row)}>Open</button>} />
      </section>
    </div>
  );
}
