import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  BarChart3,
  Database,
  FileText,
  Filter,
  GitBranch,
  Layers3,
  LineChart,
  RefreshCcw,
  Search,
  ShieldCheck,
  Table2,
  Target,
  Workflow,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const NA = "Not available in current cube";
const OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05";
const OFFICIAL_UNIVERSE_ID = "base_universe_12";

type Row = Record<string, unknown>;

type ApiRows = {
  count: number;
  rows: Row[];
};

type Metric = {
  metric_name: string;
  value: unknown;
  display_value: string;
  category: string;
  source_file?: string | null;
  source_section?: string | null;
  interpretation: string;
  limitation?: string | null;
};

type ScorecardData = {
  identity: Row;
  nomenclature: Row[];
  categories: Record<string, Metric[]>;
  metrics: Metric[];
  unavailable_metrics: Metric[];
  sources_discovered: Row[];
  summary: Row;
};

type ResearchQuestion = {
  id: string;
  question: string;
  data_sources_used: string[];
  methodology: string[];
  evidence_values: Row;
  conclusion: string;
  confidence_level: string;
  limitations: string;
};

type ResearchQuestionResponse = {
  questions: ResearchQuestion[];
};

type CandidateMetadata = {
  official: Row;
  nomenclature: Row[];
  families: Row[];
  representative_candidates: string[];
  candidates: Row[];
};

type FoldResponse = {
  folds: Row[];
  sources: string[];
  interpretation: string[];
};

type OverviewData = {
  official_candidate_id: string;
  official_universe_id: string;
  official_metrics: Row | null;
  robustness_summary: Row;
  main_sensitivity: Row | null;
  universe_summary: Row[];
  best_universe: Row | null;
  artifacts: Row[];
  narrative: string[];
};

type BudgetData = {
  rows: Row[];
  interpretation: string[];
};

type PlateauData = {
  plateau: Row[];
  sensitivity: Row[];
  worst_fold_degradation: Row[];
  interpretation: string[];
};

type Preset = {
  id: string;
  label: string;
  description: string;
  count: number;
  sample_decisions: Row[];
};

type PresetResponse = {
  presets: Preset[];
};

type MetadataOptions = {
  candidates: string[];
  universes: string[];
  folds: number[];
  tickers: string[];
  modules: string[];
  horizons: number[];
};

type DecisionDetail = {
  decision: Row | null;
  positions: Row[];
  modules: Row[];
  outcomes: Row[];
  market_context: Row | null;
  interpretation: string[];
  comparison_chips?: Row[];
  data_sources?: string[];
};

type ModuleEffectiveness = {
  continuation: Row[];
  leader: Row[];
  backoff: Row[];
  backoff_counts: Row;
  top_leader_tickers: Row[];
  module_states: Row[];
  fold_behavior: Row[];
  interpretation: string[];
  explanations?: Record<string, string[]>;
};

type TickerContribution = {
  top_positive: Row[];
  top_negative: Row[];
  all: Row[];
};

type CubeData = {
  problem?: string;
  evidence_chain?: string[];
  analytical_axes?: Row[];
  operations?: Row[];
  files: Row[];
  schemas: Record<string, string[]>;
  logical_dimensions: string[];
  relationships: string[];
  sample_queries: string[];
};

type DecisionCase = {
  date: string;
  fold: number;
  candidate_id: string;
  candidate_label: string;
  universe_id: string;
  participation_state?: string;
  long_budget?: number;
  market_regime?: string;
  outcome_20d_vs_qqq?: number;
  beat_qqq_20d?: boolean | number | null;
  beat_control_20d?: boolean | number | null;
  key_module_state?: string;
};

type DecisionPreset = {
  id: string;
  title: string;
  count: number;
  what_it_means: string;
  research_question: string;
  tables_used: string[];
  expected_interpretation: string;
  selected_explanation: string;
};

type DecisionCaseResponse = {
  presets: DecisionPreset[];
  active_preset: string;
  count: number;
  cases: DecisionCase[];
  result_text: string;
  explanation: string;
};

type ViewKey = "overview" | "robustness" | "decision" | "modules" | "cubes";

const views: { key: ViewKey; label: string; icon: ReactNode }[] = [
  { key: "overview", label: "Research Overview", icon: <ShieldCheck size={15} /> },
  { key: "robustness", label: "Robustness DSS", icon: <Activity size={15} /> },
  { key: "decision", label: "Decision Investigation", icon: <Search size={15} /> },
  { key: "modules", label: "Module and Outcome Audit", icon: <Workflow size={15} /> },
  { key: "cubes", label: "Data Cubes", icon: <Database size={15} /> },
];

const emptyScorecard: ScorecardData = {
  identity: {},
  nomenclature: [],
  categories: {},
  metrics: [],
  unavailable_metrics: [],
  sources_discovered: [],
  summary: {},
};

const emptyQuestions: ResearchQuestionResponse = { questions: [] };
const emptyCandidates: CandidateMetadata = { official: {}, nomenclature: [], families: [], representative_candidates: [], candidates: [] };
const emptyFolds: FoldResponse = { folds: [], sources: [], interpretation: [] };
const emptyDecisionCases: DecisionCaseResponse = { presets: [], active_preset: "official-baseline", count: 0, cases: [], result_text: "Showing 0 cases matching current filters.", explanation: "" };

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asText(value: unknown): string {
  if (value === null || value === undefined || value === "") return NA;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return NA;
  if (Number.isInteger(value) && Math.abs(value) < 10000) return String(value);
  if (Math.abs(value) >= 100) return value.toFixed(2);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(digits);
}

function formatMetric(value: unknown, key = ""): string {
  const numeric = asNumber(value);
  if (numeric === null) return asText(value);
  const lower = key.toLowerCase();
  if (lower.includes("share") || lower.includes("helped_rate") || lower.includes("rate")) {
    return `${(numeric * 100).toFixed(1)}%`;
  }
  if (["cagr", "maxdd"].includes(key) || lower.includes("cagr") || lower.includes("maxdd")) {
    return `${numeric.toFixed(2)}%`;
  }
  if (lower.includes("distance")) return numeric.toFixed(4);
  if (lower.includes("alpha") || lower.includes("return") || lower.includes("drawdown")) return numeric.toFixed(4);
  return formatNumber(numeric);
}

function formatHorizon(value: unknown): string {
  const numeric = asNumber(value);
  return numeric === null ? asText(value) : `${numeric}d`;
}

function buildDetailPath(params: Row): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") query.set(key, String(value));
  });
  const qs = query.toString();
  return `/dss/decision-detail${qs ? `?${qs}` : ""}`;
}

function useData<T>(path: string, initial: T): [T, string, () => void] {
  const [data, setData] = useState(initial);
  const [error, setError] = useState("");

  const load = () => {
    fetchJson<T>(path)
      .then((next) => {
        setData(next);
        setError("");
      })
      .catch((err) => setError(String(err)));
  };

  useEffect(load, [path]);
  return [data, error, load];
}

function Panel({
  title,
  icon,
  children,
  actions,
  compact = false,
}: {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  compact?: boolean;
}) {
  return (
    <section className={`border border-line bg-panel ${compact ? "p-3" : "p-4"}`}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-accent">{icon}</span>
          <h2 className="truncate text-sm font-semibold text-ink">{title}</h2>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

function ErrorBanner({ message }: { message: string }) {
  if (!message) return null;
  return <div className="border border-risk bg-risk/10 px-3 py-2 text-sm text-risk">{message}</div>;
}

function KpiGrid({ items }: { items: { label: string; value: unknown; keyName?: string; hint?: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden border border-line bg-line md:grid-cols-4 xl:grid-cols-6">
      {items.map((item) => (
        <div key={item.label} className="min-h-[86px] bg-panel-strong p-3">
          <div className="text-xs text-muted">{item.label}</div>
          <div className="mt-2 break-words text-lg font-semibold text-ink">{formatMetric(item.value, item.keyName ?? item.label)}</div>
          {item.hint && <div className="mt-1 text-xs text-muted">{item.hint}</div>}
        </div>
      ))}
    </div>
  );
}

function MetricSection({ title, metrics, limit = 12 }: { title: string; metrics: Metric[]; limit?: number }) {
  const visible = metrics.slice(0, limit);
  return (
    <div className="space-y-2">
      <div className="text-xs font-medium uppercase text-muted">{title}</div>
      <div className="grid grid-cols-1 gap-px overflow-hidden border border-line bg-line md:grid-cols-2 xl:grid-cols-3">
        {visible.map((metric) => (
          <div key={`${metric.category}-${metric.metric_name}`} className="min-h-[118px] bg-panel-strong p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 text-xs text-muted">{metric.metric_name}</div>
              {metric.value === null || metric.value === undefined ? <span className="shrink-0 border border-line px-1.5 py-0.5 text-[10px] uppercase text-muted">N/A</span> : null}
            </div>
            <div className="mt-2 break-words text-lg font-semibold text-ink">{metric.display_value}</div>
            <div className="mt-2 line-clamp-2 text-xs leading-5 text-muted">{metric.interpretation}</div>
            {metric.source_file && <div className="mt-2 truncate text-[11px] text-muted">{metric.source_file}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceBlock({ question }: { question: ResearchQuestion | undefined }) {
  if (!question) return <div className="border border-line bg-panel-strong p-4 text-sm text-muted">{NA}</div>;
  const evidenceRows = Object.entries(question.evidence_values ?? {}).map(([key, value]) => ({ metric: key, value: Array.isArray(value) ? `${value.length} rows` : asText(value) }));
  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
      <div className="space-y-3">
        <div className="border border-line bg-panel-strong p-3">
          <div className="text-xs font-medium uppercase text-muted">Methodology</div>
          <BulletList items={question.methodology} />
        </div>
        <div className="border border-line bg-panel-strong p-3">
          <div className="text-xs font-medium uppercase text-muted">Conclusion</div>
          <div className="mt-2 text-sm leading-6 text-ink">{question.conclusion}</div>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="border border-accent/50 bg-accent/10 px-2 py-1 text-ink">{question.confidence_level}</span>
            <span className="border border-line bg-panel px-2 py-1 text-muted">{question.limitations}</span>
          </div>
        </div>
      </div>
      <div className="space-y-3">
        <DataTable rows={evidenceRows} limit={12} columns={["metric", "value"]} />
        <div className="border border-line bg-panel-strong p-3">
          <div className="text-xs font-medium uppercase text-muted">Data Sources</div>
          <div className="mt-2 space-y-1">
            {question.data_sources_used.map((source) => (
              <div key={source} className="break-all text-xs leading-5 text-muted">
                {source}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Badge({ children }: { children: ReactNode }) {
  return <span className="inline-flex items-center border border-accent bg-accent/10 px-2 py-1 text-xs font-semibold uppercase text-ink">{children}</span>;
}

function ComparisonChips({ chips }: { chips?: Row[] }) {
  if (!chips?.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip) => {
        const status = String(chip.status ?? "unknown");
        const active = status === "positive" || status === "active";
        const negative = status === "negative";
        return (
          <span key={String(chip.label)} className={`border px-2 py-1 text-xs ${active ? "border-accent bg-accent/10 text-ink" : negative ? "border-risk bg-risk/10 text-risk" : "border-line bg-panel-strong text-muted"}`}>
            {asText(chip.label)} {chip.value === null || chip.value === undefined ? "" : asText(chip.value)}
          </span>
        );
      })}
    </div>
  );
}

function QuestionCards({
  questions,
  active,
  onSelect,
}: {
  questions: { id: string; label: string; note?: string; count?: number }[];
  active: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
      {questions.map((question) => (
        <button
          key={question.id}
          onClick={() => onSelect(question.id)}
          className={`min-h-[86px] border p-3 text-left transition-colors ${
            active === question.id ? "border-accent bg-accent/10 text-ink" : "border-line bg-panel-strong text-muted hover:border-muted"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <span className="text-sm font-medium text-ink">{question.label}</span>
            {typeof question.count === "number" && <span className="text-xs text-muted">{question.count}</span>}
          </div>
          {question.note && <div className="mt-2 text-xs leading-5 text-muted">{question.note}</div>}
        </button>
      ))}
    </div>
  );
}

function DataTable({ rows, limit = 12, columns }: { rows: Row[]; limit?: number; columns?: string[] }) {
  const displayRows = rows.slice(0, limit);
  const tableColumns = useMemo(() => {
    if (columns?.length) return columns.filter((column) => displayRows.some((row) => column in row));
    const preferred = [
      "CandidateId",
      "candidate_id",
      "universe_id",
      "sweep_role",
      "date",
      "decision_date",
      "fold",
      "horizon",
      "ticker",
      "module_name",
      "branch_taken",
      "CAGR",
      "Sharpe",
      "Sortino",
      "MaxDD",
      "AlphaNW_QQQ",
      "budget_multiplier",
      "long_budget",
      "robust_region_flag",
      "severe_fold_damage_count",
      "helped_rate",
      "realized_alpha_vs_qqq",
      "final_weight",
      "pnl_contribution",
    ];
    const keys = new Set<string>();
    displayRows.forEach((row) => Object.keys(row).forEach((key) => keys.add(key)));
    const ordered = preferred.filter((key) => keys.has(key));
    Array.from(keys)
      .filter((key) => !ordered.includes(key) && !["run_id", "analysis_phase", "baseline_reference", "generated_at"].includes(key))
      .slice(0, 10)
      .forEach((key) => ordered.push(key));
    return ordered.slice(0, 14);
  }, [displayRows, columns]);

  if (!displayRows.length) {
    return <div className="border border-line bg-panel-strong p-4 text-sm text-muted">No rows loaded.</div>;
  }

  return (
    <div className="table-scroll overflow-auto border border-line">
      <table className="min-w-full border-collapse text-left text-xs">
        <thead className="bg-panel-strong text-muted">
          <tr>
            {tableColumns.map((column) => (
              <th key={column} className="whitespace-nowrap border-b border-line px-3 py-2 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, rowIndex) => (
            <tr key={rowIndex} className="odd:bg-panel even:bg-panel-strong">
              {tableColumns.map((column) => (
                <td key={column} className="whitespace-nowrap border-b border-line px-3 py-2 text-ink">
                  {column === "horizon" ? formatHorizon(row[column]) : formatMetric(row[column], column)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FieldGrid({ row, fields }: { row: Row | null; fields: { label: string; key: string; kind?: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden border border-line bg-line md:grid-cols-3">
      {fields.map((field) => (
        <div key={field.key} className="min-h-[68px] bg-panel-strong p-3">
          <div className="text-xs text-muted">{field.label}</div>
          <div className="mt-1 break-words text-sm font-medium text-ink">
            {field.kind === "horizon" ? formatHorizon(row?.[field.key]) : formatMetric(row?.[field.key], field.key)}
          </div>
        </div>
      ))}
    </div>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <div className="space-y-2">
      {items.map((item, index) => (
        <div key={index} className="border-l border-accent/60 pl-3 text-sm leading-6 text-ink">
          {item}
        </div>
      ))}
    </div>
  );
}

function HelpedRateBars({ title, rows }: { title: string; rows: Row[] }) {
  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted">{title}</div>
      {rows.map((row) => {
        const rate = asNumber(row.helped_rate) ?? 0;
        const width = Math.max(0, Math.min(100, rate * 100));
        return (
          <div key={`${title}-${String(row.horizon)}`} className="grid grid-cols-[42px_1fr_72px] items-center gap-2 text-xs">
            <span className="text-muted">{formatHorizon(row.horizon)}</span>
            <div className="h-2 bg-panel-strong">
              <div className="h-2 bg-accent" style={{ width: `${width}%` }} />
            </div>
            <span className="text-right text-ink">{formatMetric(rate, "helped_rate")}</span>
          </div>
        );
      })}
    </div>
  );
}

function BudgetLineChart({ rows }: { rows: Row[] }) {
  const points = rows
    .map((row) => ({
      x: asNumber(row.budget_multiplier),
      y: asNumber(row.Sharpe),
      label: String(row.CandidateId ?? ""),
      robust: asNumber(row.robust_region_flag) === 1,
    }))
    .filter((point): point is { x: number; y: number; label: string; robust: boolean } => point.x !== null && point.y !== null);
  if (points.length < 2) return <div className="text-sm text-muted">{NA}</div>;

  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const xScale = (x: number) => 42 + ((x - minX) / Math.max(maxX - minX, 0.001)) * 554;
  const yScale = (y: number) => 188 - ((y - minY) / Math.max(maxY - minY, 0.001)) * 144;
  const path = points.map((point) => `${xScale(point.x)},${yScale(point.y)}`).join(" ");

  return (
    <svg viewBox="0 0 640 220" className="h-[240px] w-full overflow-visible">
      <line x1="42" y1="188" x2="596" y2="188" stroke="#2a3037" />
      <line x1="42" y1="44" x2="42" y2="188" stroke="#2a3037" />
      <polyline fill="none" stroke="#91a9b8" strokeWidth="2" points={path} />
      {points.map((point) => (
        <g key={point.label}>
          <circle cx={xScale(point.x)} cy={yScale(point.y)} r="4" fill={point.robust ? "#91a9b8" : "#8a8f96"} />
          <text x={xScale(point.x)} y="207" textAnchor="middle" fill="#8c949e" fontSize="11">
            {point.x.toFixed(2)}
          </text>
        </g>
      ))}
      <text x="42" y="28" fill="#8c949e" fontSize="11">
        Sharpe by budget multiplier
      </text>
    </svg>
  );
}

function ContributionBars({ rows, valueKey }: { rows: Row[]; valueKey: string }) {
  const max = Math.max(0.0001, ...rows.map((row) => Math.abs(asNumber(row[valueKey]) ?? 0)));
  return (
    <div className="space-y-2">
      {rows.slice(0, 10).map((row) => {
        const value = asNumber(row[valueKey]) ?? 0;
        const width = Math.max(3, (Math.abs(value) / max) * 100);
        return (
          <div key={String(row.ticker)} className="grid grid-cols-[58px_1fr_82px] items-center gap-2 text-xs">
            <span className="text-ink">{asText(row.ticker)}</span>
            <div className="h-2 bg-panel-strong">
              <div className="h-2 bg-accent/80" style={{ width: `${width}%` }} />
            </div>
            <span className="text-right text-muted">{formatMetric(value, valueKey)}</span>
          </div>
        );
      })}
    </div>
  );
}

function FoldHeatmap({ rows }: { rows: Row[] }) {
  const folds = Array.from(new Set(rows.map((row) => String(row.fold)))).sort((a, b) => Number(a) - Number(b));
  const horizons = Array.from(new Set(rows.map((row) => String(row.horizon)))).sort((a, b) => Number(a) - Number(b));
  return (
    <div className="overflow-auto border border-line">
      <table className="min-w-full text-xs">
        <thead className="bg-panel-strong text-muted">
          <tr>
            <th className="border-b border-line px-3 py-2 text-left">Fold</th>
            {horizons.map((horizon) => (
              <th key={horizon} className="border-b border-line px-3 py-2 text-left">
                {formatHorizon(horizon)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {folds.map((fold) => (
            <tr key={fold}>
              <td className="border-b border-line bg-panel px-3 py-2 text-ink">{fold}</td>
              {horizons.map((horizon) => {
                const row = rows.find((item) => String(item.fold) === fold && String(item.horizon) === horizon);
                const rate = asNumber(row?.helped_rate) ?? 0;
                return (
                  <td key={horizon} className="border-b border-line px-3 py-2 text-ink" style={{ backgroundColor: `rgba(145, 169, 184, ${0.08 + rate * 0.28})` }}>
                    {row ? formatMetric(rate, "helped_rate") : NA}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
  allowEmpty = true,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: (string | number)[];
  allowEmpty?: boolean;
}) {
  return (
    <label className="text-xs text-muted">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 h-9 w-full border border-line bg-panel-strong px-2 text-sm text-ink outline-none focus:border-accent">
        {allowEmpty && <option value="">All</option>}
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function Overview() {
  const [scorecard, scoreError] = useData<ScorecardData>("/dss/scorecard", emptyScorecard);
  const [questions, questionError] = useData<ResearchQuestionResponse>("/dss/research-questions", emptyQuestions);
  const [folds, foldError] = useData<FoldResponse>("/dss/folds", emptyFolds);
  const [activeQuestion, setActiveQuestion] = useState("baseline-support");

  const identity = scorecard.identity ?? {};
  const active = questions.questions.find((item) => item.id === activeQuestion) ?? questions.questions[0];
  const questionCards = questions.questions.slice(0, 9).map((item) => ({
    id: item.id,
    label: item.question,
    note: item.confidence_level,
  }));
  const foldRows = folds.folds.map((fold) => ({
    Fold: fold.Fold,
    TestStart: fold.TestStart,
    TestEnd: fold.TestEnd,
    CAGR: fold.CAGR,
    Sharpe: fold.Sharpe,
    Sortino: fold.Sortino,
    MaxDD: fold.MaxDD,
    AlphaNW_QQQ: fold.AlphaNW_QQQ,
    Exposure: fold.Exposure,
    weak_spots: Array.isArray(fold.weak_spots) ? fold.weak_spots.join("; ") : fold.weak_spots,
  }));

  return (
    <div className="space-y-4">
      <ErrorBanner message={scoreError || questionError || foldError} />
      <Panel title="Official Baseline Identity" icon={<ShieldCheck size={16} />}>
        <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
          <div className="border border-line bg-panel-strong p-4">
            <Badge>{asText(identity.badge)}</Badge>
            <div className="mt-4 text-2xl font-semibold text-ink">{asText(identity.short_label)}</div>
            <div className="mt-2 break-all text-sm text-muted">{asText(identity.technical_id)}</div>
            <div className="mt-4 grid gap-2 text-sm">
              <div className="flex justify-between gap-3 border-b border-line pb-2"><span className="text-muted">Role</span><span className="text-right text-ink">{asText(identity.candidate_role)}</span></div>
              <div className="flex justify-between gap-3 border-b border-line pb-2"><span className="text-muted">Universe</span><span className="text-right text-ink">{asText(identity.universe)}</span></div>
              <div className="flex justify-between gap-3"><span className="text-muted">Source</span><span className="text-right text-ink">{asText(identity.source)}</span></div>
            </div>
          </div>
          <div className="grid gap-px overflow-hidden border border-line bg-line md:grid-cols-2">
            {scorecard.nomenclature.map((item) => (
              <div key={String(item.code)} className="bg-panel-strong p-4">
                <div className="text-lg font-semibold text-ink">{asText(item.code)} = {asText(item.parameter)}</div>
                <div className="mt-1 text-xs text-muted">Official value {asText(item.official_value)}</div>
                <div className="mt-3 text-sm leading-6 text-ink">{asText(item.meaning)}</div>
              </div>
            ))}
          </div>
        </div>
      </Panel>

      <div className="grid gap-4">
        <Panel title="Performance Scorecard" icon={<BarChart3 size={16} />}>
          <MetricSection title="Performance" metrics={scorecard.categories.Performance ?? []} />
        </Panel>
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title="Risk Scorecard" icon={<Activity size={16} />}>
            <MetricSection title="Risk" metrics={scorecard.categories.Risk ?? []} />
          </Panel>
          <Panel title="Statistical Evidence Scorecard" icon={<LineChart size={16} />}>
            <MetricSection title="Benchmark and Statistical Evidence" metrics={scorecard.categories["Benchmark and Statistical Evidence"] ?? []} />
          </Panel>
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title="Portfolio / Execution Scorecard" icon={<Workflow size={16} />}>
            <MetricSection title="Portfolio and Execution Diagnostics" metrics={scorecard.categories["Portfolio and Execution Diagnostics"] ?? []} />
          </Panel>
          <Panel title="ML / Signal Diagnostics Scorecard" icon={<Target size={16} />}>
            <MetricSection title="ML / Signal Diagnostics" metrics={scorecard.categories["ML / Signal Diagnostics"] ?? []} />
            <div className="mt-4 border border-line bg-panel-strong p-3 text-sm leading-6 text-ink">
              In financial systems, raw classification accuracy alone can be misleading. A signal with near-50% accuracy may still add value if payoff asymmetry, exposure timing, drawdown control, or position sizing improves portfolio-level outcomes. These diagnostics should be interpreted together with alpha, drawdown, exposure, turnover, robustness and fold behavior.
            </div>
          </Panel>
        </div>
        <Panel title="Robustness Scorecard" icon={<ShieldCheck size={16} />}>
          <MetricSection title="Robustness" metrics={scorecard.categories.Robustness ?? []} />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Panel title="Research Questions" icon={<Target size={16} />}>
          <QuestionCards questions={questionCards} active={active?.id ?? ""} onSelect={setActiveQuestion} />
          <div className="mt-4">
            <EvidenceBlock question={active} />
          </div>
        </Panel>
        <Panel title="Fold Summary" icon={<GitBranch size={16} />}>
          <BulletList items={folds.interpretation} />
          <div className="mt-3">
            <DataTable rows={foldRows} limit={8} columns={["Fold", "TestStart", "TestEnd", "CAGR", "Sharpe", "MaxDD", "AlphaNW_QQQ", "Exposure", "weak_spots"]} />
          </div>
        </Panel>
      </div>

      <Panel title="Data Source Inventory" icon={<FileText size={16} />}>
        <DataTable rows={scorecard.sources_discovered} limit={12} columns={["group", "path", "available", "files"]} />
      </Panel>
    </div>
  );
}

function RobustnessDss() {
  const [overview] = useData<OverviewData>("/dss/overview", {
    official_candidate_id: OFFICIAL_CANDIDATE_ID,
    official_universe_id: OFFICIAL_UNIVERSE_ID,
    official_metrics: null,
    robustness_summary: {},
    main_sensitivity: null,
    universe_summary: [],
    best_universe: null,
    artifacts: [],
    narrative: [],
  });
  const [budget, budgetError] = useData<BudgetData>("/dss/robustness/budget", { rows: [], interpretation: [] });
  const [plateau, plateauError] = useData<PlateauData>("/dss/robustness/plateau", { plateau: [], sensitivity: [], worst_fold_degradation: [], interpretation: [] });
  const [questions, questionError] = useData<ResearchQuestionResponse>("/dss/research-questions", emptyQuestions);
  const [candidates, candidateError] = useData<CandidateMetadata>("/dss/candidates", emptyCandidates);
  const [question, setQuestion] = useState("spike");

  const robustnessQuestionIds = ["spike", "budget-localized", "budget-reduced", "budget-increased", "most-sensitive", "widest-region", "fold-damage", "generalization", "limitations"];
  const robustnessQuestions = questions.questions.filter((item) => robustnessQuestionIds.includes(item.id));
  const activeQuestion = robustnessQuestions.find((item) => item.id === question) ?? robustnessQuestions[0];
  const questionCards = robustnessQuestions.map((item) => ({ id: item.id, label: item.question, note: item.confidence_level }));

  const official = overview.official_metrics ?? {};
  const robust = overview.robustness_summary ?? {};

  return (
    <div className="space-y-4">
      <ErrorBanner message={budgetError || plateauError || questionError || candidateError} />
      <Panel title="Robustness Questions" icon={<Target size={16} />}>
        <QuestionCards questions={questionCards} active={activeQuestion?.id ?? ""} onSelect={setQuestion} />
        <div className="mt-4">
          <EvidenceBlock question={activeQuestion} />
        </div>
      </Panel>

      <Panel title="KPI Strip" icon={<BarChart3 size={16} />}>
        <KpiGrid
          items={[
            { label: "Robust region share", value: robust.robust_region_share_extended, keyName: "robust_region_share_extended" },
            { label: "Distance to decay", value: robust.distance_to_decay },
            { label: "Sampled candidates", value: robust.sampled_candidates },
            { label: "Most sensitive axis", value: robust.most_sensitive_axis },
            { label: "Official Sharpe", value: official.Sharpe },
            { label: "Official CAGR", value: official.CAGR, keyName: "CAGR" },
          ]}
        />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Budget Sensitivity" icon={<LineChart size={16} />}>
          <BudgetLineChart rows={budget.rows} />
        </Panel>
        <Panel title="Budget Axis Evidence" icon={<Table2 size={16} />}>
          <DataTable rows={budget.rows} limit={12} columns={["CandidateId", "budget_multiplier", "CAGR", "Sharpe", "Sortino", "MaxDD", "robust_region_flag", "severe_fold_damage_count"]} />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Plateau Radius" icon={<Activity size={16} />}>
          <DataTable rows={plateau.plateau} limit={8} columns={["axis", "official_value", "robust_min_sampled_value", "robust_max_sampled_value", "plateau_radius_relative", "interpretation"]} />
        </Panel>
        <Panel title="Sensitivity Ranking" icon={<Filter size={16} />}>
          <DataTable rows={plateau.sensitivity} limit={8} columns={["axis", "sensitivity_score", "worst_candidate_id", "worst_sharpe_drop", "worst_cagr_drop", "worst_severe_fold_damage_count"]} />
        </Panel>
      </div>

      <Panel title="Worst Fold Degradation" icon={<GitBranch size={16} />}>
        <DataTable rows={plateau.worst_fold_degradation} limit={12} columns={["CandidateId", "sweep_role", "CAGR", "Sharpe", "MaxDD", "severe_fold_damage_count", "worst_fold_cagr_delta_vs_official"]} />
      </Panel>

      <Panel title="Candidate Explanation" icon={<ShieldCheck size={16} />}>
        <DataTable rows={candidates.candidates} limit={14} columns={["label", "technical_id", "role", "changed_axes", "robust_flag", "representative_cube_candidate", "interpretation"]} />
      </Panel>

      <Panel title="Static figures from reports" icon={<LineChart size={16} />}>
        <details>
          <summary className="cursor-pointer text-sm text-ink">Open static figure references</summary>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <img className="w-full border border-line bg-panel-strong" src={`${API_BASE}/figures/extended_multiplier_heatmap.png`} alt="Extended multiplier heatmap" />
            <img className="w-full border border-line bg-panel-strong" src={`${API_BASE}/figures/multiplier_1d_degradation.png`} alt="One-dimensional degradation" />
          </div>
        </details>
      </Panel>
    </div>
  );
}

function DecisionInvestigation() {
  const [options] = useData<MetadataOptions>("/metadata/options", { candidates: [], universes: [], folds: [], tickers: [], modules: [], horizons: [] });
  const [activePreset, setActivePreset] = useState("official-baseline");
  const [selectedCase, setSelectedCase] = useState<DecisionCase | null>(null);
  const [detail, setDetail] = useState<DecisionDetail>({ decision: null, positions: [], modules: [], outcomes: [], market_context: null, interpretation: [] });
  const [error, setError] = useState("");
  const [filters, setFilters] = useState({
    date_start: "",
    date_end: "",
    fold: "",
    candidate_id: OFFICIAL_CANDIDATE_ID,
    universe_id: OFFICIAL_UNIVERSE_ID,
  });
  const [appliedFilters, setAppliedFilters] = useState(filters);
  const [detailFilters, setDetailFilters] = useState({
    ticker: "",
    module_name: "",
    horizon: "",
  });

  const casePath = useMemo(() => {
    const query = new URLSearchParams({ preset_id: activePreset, limit: "120" });
    Object.entries(appliedFilters).forEach(([key, value]) => {
      if (value !== "") query.set(key, value);
    });
    return `/dss/decision-cases?${query.toString()}`;
  }, [activePreset, appliedFilters]);

  const [caseData, caseError, reloadCases] = useData<DecisionCaseResponse>(casePath, emptyDecisionCases);
  const active = caseData.presets.find((preset) => preset.id === activePreset) ?? caseData.presets[0];

  const loadDetail = (params: Row) => {
    fetchJson<DecisionDetail>(buildDetailPath(params))
      .then((next) => {
        setDetail(next);
        setError("");
      })
      .catch((err) => setError(String(err)));
  };

  useEffect(() => {
    const first = caseData.cases[0];
    if (first) {
      setSelectedCase(first);
      loadDetail({
        date: first.date,
        fold: first.fold,
        candidate_id: first.candidate_id,
        universe_id: first.universe_id,
      });
    }
  }, [casePath, caseData.cases.length]);

  useEffect(() => {
    if (detail.decision) return;
    loadDetail({ candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: OFFICIAL_UNIVERSE_ID });
  }, []);

  const filteredPositions = detail.positions.filter((row) => !detailFilters.ticker || row.ticker === detailFilters.ticker);
  const filteredModules = detail.modules.filter((row) => !detailFilters.module_name || row.module_name === detailFilters.module_name);
  const filteredOutcomes = detail.outcomes.filter((row) => !detailFilters.horizon || String(row.horizon) === detailFilters.horizon);

  return (
    <div className="space-y-4">
      <ErrorBanner message={caseError || error} />
      <Panel title="Guided Presets" icon={<Target size={16} />}>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
          {caseData.presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setActivePreset(preset.id)}
              className={`min-h-[164px] border p-3 text-left transition-colors ${activePreset === preset.id ? "border-accent bg-accent/10" : "border-line bg-panel-strong hover:border-muted"}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="text-sm font-semibold text-ink">{preset.title}</div>
                <span className="text-xs text-muted">{preset.count}</span>
              </div>
              <div className="mt-2 text-xs leading-5 text-muted">{preset.what_it_means}</div>
              <div className="mt-2 text-xs leading-5 text-ink">{preset.research_question}</div>
              <div className="mt-2 truncate text-[11px] text-muted">{preset.tables_used.join(", ")}</div>
            </button>
          ))}
        </div>
        {active && <div className="mt-4 border border-line bg-panel-strong p-3 text-sm leading-6 text-ink">{caseData.explanation || active.selected_explanation}</div>}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
        <Panel
          title="Filters"
          icon={<Filter size={16} />}
          actions={
            <button
              onClick={() => {
                setAppliedFilters(filters);
                reloadCases();
              }}
              className="flex h-9 items-center gap-2 border border-accent px-3 text-sm text-ink"
            >
              <Search size={14} />
              Update cases
            </button>
          }
        >
          <div className="grid gap-3">
            <label className="text-xs text-muted">
              Start date
              <input value={filters.date_start} onChange={(event) => setFilters({ ...filters, date_start: event.target.value })} className="mt-1 h-9 w-full border border-line bg-panel-strong px-2 text-sm text-ink outline-none focus:border-accent" />
            </label>
            <label className="text-xs text-muted">
              End date
              <input value={filters.date_end} onChange={(event) => setFilters({ ...filters, date_end: event.target.value })} className="mt-1 h-9 w-full border border-line bg-panel-strong px-2 text-sm text-ink outline-none focus:border-accent" />
            </label>
            <SelectField label="Candidate" value={filters.candidate_id} onChange={(value) => setFilters({ ...filters, candidate_id: value })} options={options.candidates} allowEmpty={false} />
            <SelectField label="Universe" value={filters.universe_id} onChange={(value) => setFilters({ ...filters, universe_id: value })} options={options.universes} allowEmpty={false} />
            <SelectField label="Fold" value={filters.fold} onChange={(value) => setFilters({ ...filters, fold: value })} options={options.folds} />
          </div>
          <div className="mt-4 border border-line bg-panel-strong p-3 text-sm text-ink">{caseData.result_text}</div>
        </Panel>

        <Panel title="Case Timeline" icon={<Search size={16} />}>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-medium text-ink">{active?.title ?? NA}</div>
            <div className="text-xs text-muted">{caseData.count} matching cases</div>
          </div>
          <div className="max-h-[520px] overflow-auto border border-line">
            {caseData.cases.map((item, index) => {
              const isActive =
                selectedCase &&
                String(selectedCase.date) === String(item.date) &&
                String(selectedCase.fold) === String(item.fold) &&
                String(selectedCase.candidate_id) === String(item.candidate_id);
              return (
                <button
                  key={`${String(item.date)}-${String(item.fold)}-${String(item.candidate_id)}-${index}`}
                  onClick={() => {
                    setSelectedCase(item);
                    loadDetail({
                      date: item.date,
                      fold: item.fold,
                      candidate_id: item.candidate_id,
                      universe_id: item.universe_id,
                    });
                  }}
                  className={`grid w-full gap-2 border-b border-line px-3 py-3 text-left text-xs md:grid-cols-[92px_48px_1fr_120px_90px_1.2fr] ${
                    isActive ? "bg-accent/10 text-ink" : "bg-panel-strong text-muted hover:bg-panel"
                  }`}
                >
                  <span>{asText(item.date)}</span>
                  <span>F{asText(item.fold)}</span>
                  <span className="truncate">{asText(item.candidate_label)}</span>
                  <span className="truncate">{asText(item.participation_state)}</span>
                  <span>{formatMetric(item.long_budget, "long_budget")}</span>
                  <span className="truncate">{asText(item.market_regime)} / 20d alpha {formatMetric(item.outcome_20d_vs_qqq, "realized_alpha_vs_qqq")}</span>
                  <span className="md:col-span-6 truncate text-muted">{asText(item.key_module_state)}</span>
                </button>
              );
            })}
          </div>
        </Panel>
      </div>

      <Panel title="Connected Detail" icon={<ShieldCheck size={16} />}>
        <ComparisonChips chips={detail.comparison_chips} />
        <div className="mt-4">
          <BulletList items={detail.interpretation} />
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Decision Summary" icon={<Activity size={16} />}>
          <FieldGrid
            row={detail.decision}
            fields={[
              { label: "Date", key: "date" },
              { label: "Fold", key: "fold" },
              { label: "Candidate", key: "candidate_id" },
              { label: "Universe", key: "universe_id" },
              { label: "Participation", key: "participation_state" },
              { label: "Long budget", key: "long_budget" },
              { label: "Gate scale", key: "gate_scale" },
              { label: "Exposure cap", key: "exp_cap" },
              { label: "Hard backoff", key: "hard_backoff_flag" },
            ]}
          />
        </Panel>
        <Panel title="Market Context" icon={<LineChart size={16} />}>
          <FieldGrid
            row={detail.market_context}
            fields={[
              { label: "QQQ return", key: "qqq_return" },
              { label: "QQQ drawdown", key: "qqq_drawdown" },
              { label: "QQQ volatility", key: "qqq_vol" },
              { label: "SPY return", key: "spy_return" },
              { label: "SPY drawdown", key: "spy_drawdown" },
              { label: "VIX", key: "vix" },
              { label: "Breadth", key: "breadth" },
              { label: "Regime proxy", key: "market_regime_proxy" },
            ]}
          />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Active Modules" icon={<Layers3 size={16} />}>
          <div className="mb-3 grid gap-3 md:grid-cols-2">
            <SelectField label="Module detail filter" value={detailFilters.module_name} onChange={(value) => setDetailFilters({ ...detailFilters, module_name: value })} options={options.modules} />
            <SelectField label="Outcome horizon filter" value={detailFilters.horizon} onChange={(value) => setDetailFilters({ ...detailFilters, horizon: value })} options={options.horizons} />
          </div>
          <FieldGrid
            row={detail.decision}
            fields={[
              { label: "Continuation trigger", key: "continuation_trigger_p" },
              { label: "Continuation pressure", key: "continuation_pressure_p" },
              { label: "Break risk", key: "continuation_break_risk_p" },
              { label: "Structural probability", key: "structural_p" },
              { label: "Leader blend", key: "leader_blend" },
              { label: "Backoff strength", key: "backoff_strength_applied" },
            ]}
          />
          <div className="mt-3">
            <DataTable rows={filteredModules} limit={10} columns={["module_name", "branch_taken", "threshold_crossed", "signal_strength"]} />
          </div>
        </Panel>
        <Panel title="Selected Positions" icon={<Table2 size={16} />}>
          <div className="mb-3">
            <SelectField label="Ticker detail filter" value={detailFilters.ticker} onChange={(value) => setDetailFilters({ ...detailFilters, ticker: value })} options={options.tickers} />
          </div>
          <DataTable rows={filteredPositions} limit={14} columns={["ticker", "rank", "base_score", "selected_flag", "leader_flag", "base_weight", "final_weight", "stop_flag", "pnl_contribution"]} />
        </Panel>
      </div>

      <Panel title="Outcome" icon={<BarChart3 size={16} />}>
        <DataTable rows={filteredOutcomes} limit={8} columns={["horizon", "realized_return", "realized_alpha_vs_qqq", "realized_alpha_vs_spy", "decision_helped_flag_vs_qqq", "decision_helped_flag_vs_control", "continuation_helped_flag", "backoff_helped_flag", "leader_helped_flag"]} />
      </Panel>

      <Panel title="Data Sources Used" icon={<FileText size={16} />}>
        <div className="grid gap-2 md:grid-cols-2">
          {(detail.data_sources ?? []).map((source) => (
            <div key={source} className="break-all border border-line bg-panel-strong p-2 text-xs text-muted">
              {source}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ModuleOutcomeAudit() {
  const [moduleData, moduleError] = useData<ModuleEffectiveness>("/dss/module-effectiveness", {
    continuation: [],
    leader: [],
    backoff: [],
    backoff_counts: {},
    top_leader_tickers: [],
    module_states: [],
    fold_behavior: [],
    interpretation: [],
  });
  const [tickerData, tickerError] = useData<TickerContribution>("/dss/ticker-contribution", { top_positive: [], top_negative: [], all: [] });
  const [question, setQuestion] = useState("continuation");

  const questions = [
    { id: "continuation", label: "Did continuation help more at longer horizons?" },
    { id: "leader", label: "Did leader participation help?" },
    { id: "backoff", label: "Did backoff help during fragile regimes?" },
    { id: "tickers", label: "Which tickers contributed most?" },
    { id: "folds", label: "Which folds were weaker?" },
    { id: "modules", label: "Which modules activate most often?" },
  ];
  const answers: Record<string, string[]> = {
    continuation: moduleData.explanations?.continuation ?? ["Continuation helped rates are grouped directly from outcome_cube by horizon."],
    leader: moduleData.explanations?.leader ?? ["Leader participation helped rates are grouped directly from outcome_cube by horizon."],
    backoff: moduleData.explanations?.backoff ?? [
      `Backoff count: ${formatMetric(moduleData.backoff_counts.backoff_count)}.`,
      `Hard backoff count: ${formatMetric(moduleData.backoff_counts.hard_backoff_count)}.`,
    ],
    tickers: moduleData.explanations?.tickers ?? ["Ticker contribution is aggregated from selected position rows and realized PnL contribution fields."],
    folds: moduleData.explanations?.folds ?? ["Fold behavior uses helped-rate and alpha aggregates by fold and outcome horizon."],
    modules: moduleData.interpretation,
  };

  return (
    <div className="space-y-4">
      <ErrorBanner message={moduleError || tickerError} />
      <Panel title="Audit Questions" icon={<Target size={16} />}>
        <QuestionCards questions={questions} active={question} onSelect={setQuestion} />
        <div className="mt-4 border border-line bg-panel-strong p-3">
          <BulletList items={answers[question] ?? []} />
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="Continuation Effectiveness" icon={<Activity size={16} />}>
          <HelpedRateBars title="Helped rate by horizon" rows={moduleData.continuation} />
          <div className="mt-4">
            <DataTable rows={moduleData.continuation} limit={6} columns={["horizon", "count", "helped_rate", "avg_alpha_vs_qqq"]} />
          </div>
        </Panel>
        <Panel title="Leader Participation" icon={<GitBranch size={16} />}>
          <HelpedRateBars title="Helped rate by horizon" rows={moduleData.leader} />
          <div className="mt-4">
            <DataTable rows={moduleData.top_leader_tickers} limit={6} columns={["ticker", "selected_frequency", "pnl_contribution", "mean_final_weight"]} />
          </div>
        </Panel>
        <Panel title="Backoff Effectiveness" icon={<ShieldCheck size={16} />}>
          <HelpedRateBars title="Helped rate by horizon" rows={moduleData.backoff} />
          <div className="mt-4">
            <KpiGrid
              items={[
                { label: "Backoff count", value: moduleData.backoff_counts.backoff_count },
                { label: "Hard backoff count", value: moduleData.backoff_counts.hard_backoff_count },
              ]}
            />
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Top Positive Contributors" icon={<BarChart3 size={16} />}>
          <ContributionBars rows={tickerData.top_positive} valueKey="total_pnl_contribution" />
        </Panel>
        <Panel title="Top Negative Contributors" icon={<BarChart3 size={16} />}>
          <ContributionBars rows={tickerData.top_negative} valueKey="total_pnl_contribution" />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Fold Behavior" icon={<GitBranch size={16} />}>
          <FoldHeatmap rows={moduleData.fold_behavior} />
        </Panel>
        <Panel title="Module State Frequency" icon={<Layers3 size={16} />}>
          <DataTable rows={moduleData.module_states} limit={18} columns={["module_name", "branch_taken", "observations", "mean_signal_strength", "threshold_cross_rate"]} />
        </Panel>
      </div>

      <Panel title="Ticker Contribution Table" icon={<Table2 size={16} />}>
        <DataTable rows={tickerData.all} limit={16} columns={["ticker", "selected_frequency", "leader_frequency", "total_pnl_contribution", "mean_final_weight", "mean_base_score"]} />
      </Panel>
    </div>
  );
}

function DataCubes() {
  const [cubeData, cubeError] = useData<CubeData>("/dss/data-cubes", { files: [], schemas: {}, logical_dimensions: [], relationships: [], sample_queries: [] });
  const [cube, setCube] = useState("decision_date_cube");
  const [rawRows, setRawRows] = useState<ApiRows>({ count: 0, rows: [] });
  const [rawError, setRawError] = useState("");

  const rawEndpoints: Record<string, string> = {
    decision_date_cube: "/decisions?limit=200",
    position_cube: "/positions?selected_only=true&limit=200",
    module_trace_cube: "/module-trace?limit=200",
    outcome_cube: "/outcomes?limit=200",
    market_context_cube: "/market-context?limit=200",
  };

  const loadRaw = () => {
    fetchJson<ApiRows>(rawEndpoints[cube] ?? "/decisions?limit=200")
      .then((next) => {
        setRawRows(next);
        setRawError("");
      })
      .catch((err) => setRawError(String(err)));
  };

  useEffect(loadRaw, [cube]);

  return (
    <div className="space-y-4">
      <ErrorBanner message={cubeError || rawError} />
      <Panel title="What problem does the DSS solve?" icon={<Target size={16} />}>
        <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="border border-line bg-panel-strong p-4 text-sm leading-6 text-ink">{cubeData.problem ?? NA}</div>
          <div className="border border-line bg-panel-strong p-4">
            <div className="text-xs font-medium uppercase text-muted">Evidence Chain</div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {(cubeData.evidence_chain ?? []).map((item, index) => (
                <span key={item} className="flex items-center gap-2 text-sm text-ink">
                  <span className="border border-line bg-panel px-2 py-1">{item}</span>
                  {index < (cubeData.evidence_chain?.length ?? 0) - 1 && <span className="text-muted">-&gt;</span>}
                </span>
              ))}
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
        <Panel title="Analytical Axes" icon={<GitBranch size={16} />}>
          <DataTable rows={cubeData.analytical_axes ?? []} limit={12} columns={["axis", "meaning"]} />
        </Panel>
        <Panel title="Supported Operations" icon={<Workflow size={16} />}>
          <DataTable rows={cubeData.operations ?? []} limit={12} columns={["operation", "meaning", "tables_used", "example_conclusion"]} />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel title="Physical Files" icon={<Database size={16} />}>
          <DataTable rows={cubeData.files} limit={10} columns={["cube", "file", "rows", "columns", "grain", "size_bytes"]} />
        </Panel>
        <Panel title="Logical Dimensions" icon={<GitBranch size={16} />}>
          <div className="grid grid-cols-2 gap-2">
            {cubeData.logical_dimensions.map((dimension) => (
              <div key={dimension} className="border border-line bg-panel-strong px-3 py-2 text-sm text-ink">
                {dimension}
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Fact Relationships" icon={<Workflow size={16} />}>
          <BulletList items={cubeData.relationships} />
        </Panel>
        <Panel title="Sample Query Shapes" icon={<FileText size={16} />}>
          <div className="space-y-2">
            {cubeData.sample_queries.map((query) => (
              <pre key={query} className="overflow-auto border border-line bg-panel-strong p-3 text-xs text-ink">
                {query}
              </pre>
            ))}
          </div>
        </Panel>
      </div>

      <Panel
        title="Schema Preview"
        icon={<Layers3 size={16} />}
        actions={
          <select value={cube} onChange={(event) => setCube(event.target.value)} className="h-9 border border-line bg-panel-strong px-2 text-sm text-ink outline-none focus:border-accent">
            {Object.keys(cubeData.schemas).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        }
      >
        <div className="flex flex-wrap gap-2">
          {(cubeData.schemas[cube] ?? []).map((column) => (
            <span key={column} className="border border-line bg-panel-strong px-2 py-1 text-xs text-muted">
              {column}
            </span>
          ))}
        </div>
      </Panel>

      <Panel
        title="Secondary Raw Cube Access"
        icon={<Table2 size={16} />}
        actions={
          <button onClick={loadRaw} className="flex h-9 items-center gap-2 border border-accent px-3 text-sm text-ink">
            <RefreshCcw size={14} />
            Refresh
          </button>
        }
      >
        <div className="mb-2 text-xs text-muted">{rawRows.count} rows matched; showing first page.</div>
        <DataTable rows={rawRows.rows} limit={18} />
      </Panel>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<ViewKey>("overview");

  return (
    <div className="min-h-screen bg-base text-ink">
      <header className="border-b border-line bg-base">
        <div className="mx-auto max-w-[1500px] px-4 py-4">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="text-xs font-medium uppercase text-muted">Mahoraga 14.3 Extended Analysis</div>
              <h1 className="mt-1 text-2xl font-semibold text-ink">Research DSS and Interpretability Audit</h1>
            </div>
            <nav className="flex flex-wrap gap-2">
              {views.map((item) => (
                <button
                  key={item.key}
                  onClick={() => setView(item.key)}
                  className={`flex h-9 items-center gap-2 border px-3 text-sm ${
                    view === item.key ? "border-accent bg-accent/10 text-ink" : "border-line bg-panel text-muted hover:border-muted hover:text-ink"
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-4 py-4">
        {view === "overview" && <Overview />}
        {view === "robustness" && <RobustnessDss />}
        {view === "decision" && <DecisionInvestigation />}
        {view === "modules" && <ModuleOutcomeAudit />}
        {view === "cubes" && <DataCubes />}
      </main>
    </div>
  );
}
