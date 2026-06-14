import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Options } from "../api/types";
import { ChartPanel } from "../components/ChartPanel";
import { ChartTooltip, axisLabel } from "../components/ChartTooltip";
import { SelectControl } from "../components/Controls";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { ErrorState, LoadingState } from "../components/States";
import { useApiResource } from "../hooks/useApiResource";
import { formatMetric } from "../utils/format";
import { OFFICIAL_CANDIDATE_ID } from "../utils/labels";
import { rowsFrom } from "../utils/rows";

export default function ModuleAttribution({ options }: { options: Options | null }) {
  const [fold, setFold] = useState("all");
  const [matrixMetric, setMatrixMetric] = useState("helped_rate");
  const modules = useApiResource<Record<string, unknown>>("/module/effectiveness", { candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: options?.default_universe ?? "base_universe_12", fold });
  const activation = rowsFrom(modules.data, "activation");
  const byHorizon = rowsFrom(modules.data, "by_horizon");
  const bestHelped = [...byHorizon].filter((row) => row.helped_rate !== null).sort((a, b) => Number(b.helped_rate) - Number(a.helped_rate))[0];
  const topActivation = [...activation].filter((row) => row.activation_rate !== null).sort((a, b) => Number(b.activation_rate) - Number(a.activation_rate))[0];
  const weak = [...byHorizon].filter((row) => Number(row.avg_alpha_vs_qqq) < 0).sort((a, b) => Number(a.avg_alpha_vs_qqq) - Number(b.avg_alpha_vs_qqq))[0];
  const horizons = useMemo(() => Array.from(new Set(byHorizon.map((row) => String(row.horizon)).filter(Boolean))).sort((a, b) => Number(a) - Number(b)), [byHorizon]);
  const moduleNames = useMemo(() => Array.from(new Set(byHorizon.map((row) => String(row.module_name)).filter(Boolean))).sort(), [byHorizon]);
  const matrixRows = useMemo(() => moduleNames.map((module) => ({
    module,
    cells: horizons.map((horizon) => byHorizon.find((row) => String(row.module_name) === module && String(row.horizon) === horizon) ?? null),
  })), [moduleNames, horizons, byHorizon]);
  const values = byHorizon.map((row) => Number(row[matrixMetric])).filter(Number.isFinite);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 1);
  if (modules.loading && !modules.data) return <LoadingState label="Loading module attribution" />;
  if (modules.error) return <ErrorState error={modules.error} retry={modules.retry} />;

  function heat(cell: Record<string, unknown> | null) {
    if (!cell) return "rgba(38, 50, 57, 0.35)";
    const value = Number(cell[matrixMetric]);
    if (!Number.isFinite(value)) return "rgba(38, 50, 57, 0.35)";
    const t = Math.max(0, Math.min(1, (value - minValue) / Math.max(0.000001, maxValue - minValue)));
    return `rgba(${Math.round(40 + 74 * t)}, ${Math.round(74 + 166 * t)}, ${Math.round(88 + 89 * t)}, ${0.35 + t * 0.55})`;
  }
  return (
    <div className="view-grid">
      <section className="panel span-12"><SectionHeader title="Module Attribution" question="Which modules help, hurt or change behavior by horizon?" source="mart.mv_module_effectiveness" action={<><SelectControl label="Fold" value={fold} options={["all", ...(options?.folds ?? [])]} onChange={setFold} compact /><SelectControl label="Matrix metric" value={matrixMetric} options={["helped_rate", "activation_rate", "avg_alpha_vs_qqq"]} onChange={setMatrixMetric} compact /></>} /><div className="metric-grid">{activation.slice(0, 7).map((row) => <MetricCard key={String(row.module_name)} label={String(row.module_name)} value={formatMetric(row.activation_rate, "activation_rate")} detail={`${formatMetric(row.observations, "observations")} observations`} />)}</div></section>
      <section className="panel span-12"><div className="insight-strip">{bestHelped ? <article className="insight-card"><b>Best helped rate</b><span>{String(bestHelped.module_name)} leads at horizon {String(bestHelped.horizon)}.</span></article> : null}{topActivation ? <article className="insight-card"><b>Highest activation</b><span>{String(topActivation.module_name)} is most frequently active.</span></article> : null}{weak ? <article className="insight-card"><b>Weak contribution</b><span>{String(weak.module_name)} has negative average alpha in this slice.</span></article> : null}</div></section>
      <section className="panel span-12">
        <SectionHeader title="Module x Horizon Matrix" question="Rows are modules, columns are horizons, cell color is the selected metric." source="mart.mv_module_effectiveness" />
        <div className="heatmap-table">
          <div className="heatmap-row header"><span>Module</span>{horizons.map((horizon) => <b key={horizon}>{horizon}d</b>)}</div>
          {matrixRows.map((row) => (
            <div className="heatmap-row" key={row.module}>
              <span>{row.module}</span>
              {row.cells.map((cell, index) => (
                <b key={`${row.module}-${horizons[index]}`} style={{ background: heat(cell) }} title={cell ? `${matrixMetric}: ${formatMetric(cell[matrixMetric], matrixMetric)} · observations ${formatMetric(cell.observations, "observations")}` : "No materialized row"}>
                  {cell ? formatMetric(cell[matrixMetric], matrixMetric) : ""}
                </b>
              ))}
            </div>
          ))}
        </div>
      </section>
      <ChartPanel title="Activation Ranking" question="Which modules most frequently alter the official policy path?" source="fact_module_trace" ready={activation.length >= 2}>
        <ResponsiveContainer width="100%" height="100%"><BarChart data={activation} layout="vertical"><CartesianGrid stroke="#22303a" /><XAxis type="number" label={axisLabel("Activation rate")} /><YAxis dataKey="module_name" type="category" width={190} label={axisLabel("Module", true)} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="activation_rate" fill="#f7c76a" /></BarChart></ResponsiveContainer>
      </ChartPanel>
      <section className="panel span-12"><SectionHeader title="Module Table" question="Activation, helped rate, alpha and drawdown diagnostics by module/horizon." source="mart.mv_module_effectiveness" /><DataTable rows={byHorizon} columns={["module_name", "horizon", "activation_rate", "helped_rate", "avg_alpha_vs_qqq", "avg_drawdown_change", "avg_exposure_effect", "observations", "demo_mode"]} /></section>
    </div>
  );
}
