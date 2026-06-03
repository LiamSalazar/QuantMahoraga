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
};

type TickerContribution = {
  top_positive: Row[];
  top_negative: Row[];
  all: Row[];
};

type CubeData = {
  files: Row[];
  schemas: Record<string, string[]>;
  logical_dimensions: string[];
  relationships: string[];
  sample_queries: string[];
};

type ViewKey = "overview" | "robustness" | "decision" | "modules" | "cubes";

const views: { key: ViewKey; label: string; icon: ReactNode }[] = [
  { key: "overview", label: "Research Overview", icon: <ShieldCheck size={15} /> },
  { key: "robustness", label: "Robustness DSS", icon: <Activity size={15} /> },
  { key: "decision", label: "Decision Investigation", icon: <Search size={15} /> },
  { key: "modules", label: "Module and Outcome Audit", icon: <Workflow size={15} /> },
  { key: "cubes", label: "Data Cubes", icon: <Database size={15} /> },
];

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
  const [overview, error] = useData<OverviewData>("/dss/overview", {
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

  const official = overview.official_metrics ?? {};
  const robust = overview.robustness_summary ?? {};
  const kpis = [
    { label: "Official candidate", value: overview.official_candidate_id },
    { label: "CAGR", value: official.CAGR, keyName: "CAGR" },
    { label: "Sharpe", value: official.Sharpe },
    { label: "Sortino", value: official.Sortino },
    { label: "MaxDD", value: official.MaxDD, keyName: "MaxDD" },
    { label: "AlphaNW_QQQ", value: official.AlphaNW_QQQ },
    { label: "AlphaNW_SPY", value: official.AlphaNW_SPY },
    { label: "Robust region", value: robust.robust_region_share_extended, keyName: "robust_region_share_extended" },
    { label: "Distance to decay", value: robust.distance_to_decay },
    { label: "Sensitive axis", value: robust.most_sensitive_axis },
    { label: "Best universe", value: overview.best_universe?.universe_id },
    { label: "Sampled candidates", value: robust.sampled_candidates },
  ];

  return (
    <div className="space-y-4">
      <ErrorBanner message={error} />
      <Panel title="Executive Research State" icon={<ShieldCheck size={16} />}>
        <KpiGrid items={kpis} />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel title="Deterministic Reading" icon={<Target size={16} />}>
          <BulletList items={overview.narrative} />
        </Panel>
        <Panel title="Main Sensitivity" icon={<Activity size={16} />}>
          <FieldGrid
            row={overview.main_sensitivity}
            fields={[
              { label: "Axis", key: "axis" },
              { label: "Sensitivity score", key: "sensitivity_score" },
              { label: "Worst candidate", key: "worst_candidate_id" },
              { label: "Worst Sharpe drop", key: "worst_sharpe_drop" },
              { label: "Worst CAGR drop", key: "worst_cagr_drop" },
              { label: "Worst fold damage", key: "worst_severe_fold_damage_count" },
            ]}
          />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <Panel title="Universe Robustness Summary" icon={<GitBranch size={16} />}>
          <DataTable rows={overview.universe_summary} limit={8} columns={["universe_id", "run_status", "usable_count", "CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ"]} />
        </Panel>
        <Panel title="Audit Artifacts" icon={<FileText size={16} />}>
          <DataTable rows={overview.artifacts} limit={10} columns={["file", "rows", "available"]} />
        </Panel>
      </div>
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
  const [question, setQuestion] = useState("spike");

  const questionText: Record<string, string[]> = {
    spike: [
      `Robust region share is ${formatMetric(overview.robustness_summary.robust_region_share_extended, "robust_region_share_extended")}.`,
      "The sampled evidence does not describe the official candidate as a single narrow spike.",
      "The caution is local: sensitivity appears first around budget underdeployment.",
    ],
    budgetDown: budget.interpretation,
    budgetUp: [
      "Moderate upward budget samples did not collapse performance in the sampled one-dimensional range.",
      "This does not establish global optimality outside the sampled perturbations.",
    ],
    sensitive: [
      `The most sensitive axis is ${asText(overview.robustness_summary.most_sensitive_axis)}.`,
      "Sensitivity ranking is taken from the extended multiplier robustness CSV.",
    ],
    plateau: plateau.interpretation,
    fold: ["Worst-fold degradation isolates local walk-forward damage that can be hidden by stitched aggregate results."],
    symmetry: ["The plateau table shows asymmetric sampled robustness: budget is narrower on the lower side than conviction, leader, or backoff."],
  };

  const questions = [
    { id: "spike", label: "Is the official candidate a narrow parameter spike?" },
    { id: "budgetDown", label: "What happens when budget is reduced?" },
    { id: "budgetUp", label: "What happens when budget is increased?" },
    { id: "sensitive", label: "Which multiplier is most sensitive?" },
    { id: "plateau", label: "Which parameters have the widest robust region?" },
    { id: "fold", label: "Which candidates caused fold-level damage?" },
    { id: "symmetry", label: "Does robustness look symmetric or asymmetric?" },
  ];

  const official = overview.official_metrics ?? {};
  const robust = overview.robustness_summary ?? {};

  return (
    <div className="space-y-4">
      <ErrorBanner message={budgetError || plateauError} />
      <Panel title="Robustness Questions" icon={<Target size={16} />}>
        <QuestionCards questions={questions} active={question} onSelect={setQuestion} />
        <div className="mt-4 border border-line bg-panel-strong p-3">
          <BulletList items={questionText[question] ?? []} />
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

      <Panel title="Robustness Figures" icon={<LineChart size={16} />}>
        <div className="grid gap-4 lg:grid-cols-2">
          <img className="w-full border border-line bg-panel-strong" src={`${API_BASE}/figures/extended_multiplier_heatmap.png`} alt="Extended multiplier heatmap" />
          <img className="w-full border border-line bg-panel-strong" src={`${API_BASE}/figures/multiplier_1d_degradation.png`} alt="One-dimensional degradation" />
        </div>
      </Panel>
    </div>
  );
}

function DecisionInvestigation() {
  const [options] = useData<MetadataOptions>("/metadata/options", { candidates: [], universes: [], folds: [], tickers: [], modules: [], horizons: [] });
  const [presets, presetError] = useData<PresetResponse>("/dss/presets", { presets: [] });
  const [activePreset, setActivePreset] = useState("official-baseline");
  const [selectedCase, setSelectedCase] = useState<Row | null>(null);
  const [detail, setDetail] = useState<DecisionDetail>({ decision: null, positions: [], modules: [], outcomes: [], market_context: null, interpretation: [] });
  const [error, setError] = useState("");
  const [manual, setManual] = useState({
    date: "",
    fold: "",
    candidate_id: OFFICIAL_CANDIDATE_ID,
    universe_id: OFFICIAL_UNIVERSE_ID,
    ticker: "",
    module_name: "",
    horizon: "",
  });

  const active = presets.presets.find((preset) => preset.id === activePreset) ?? presets.presets[0];

  const loadDetail = (params: Row) => {
    fetchJson<DecisionDetail>(buildDetailPath(params))
      .then((next) => {
        setDetail(next);
        setError("");
      })
      .catch((err) => setError(String(err)));
  };

  useEffect(() => {
    if (!presets.presets.length) return;
    const current = presets.presets.find((preset) => preset.id === activePreset) ?? presets.presets[0];
    const first = current.sample_decisions[0];
    if (first) {
      setSelectedCase(first);
      loadDetail(first);
    }
  }, [presets.presets.length, activePreset]);

  useEffect(() => {
    if (detail.decision) return;
    loadDetail({ candidate_id: OFFICIAL_CANDIDATE_ID, universe_id: OFFICIAL_UNIVERSE_ID });
  }, []);

  const filteredPositions = detail.positions.filter((row) => !manual.ticker || row.ticker === manual.ticker);
  const filteredModules = detail.modules.filter((row) => !manual.module_name || row.module_name === manual.module_name);
  const filteredOutcomes = detail.outcomes.filter((row) => !manual.horizon || String(row.horizon) === manual.horizon);

  const presetCards = presets.presets.map((preset) => ({
    id: preset.id,
    label: preset.label,
    count: preset.count,
    note: preset.description,
  }));

  return (
    <div className="space-y-4">
      <ErrorBanner message={presetError || error} />
      <Panel title="Guided Presets" icon={<Target size={16} />}>
        <QuestionCards questions={presetCards} active={activePreset} onSelect={setActivePreset} />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
        <Panel title="Preset Cases" icon={<Search size={16} />}>
          <div className="mb-3 text-sm font-medium text-ink">{active?.label ?? NA}</div>
          <div className="max-h-[360px] overflow-auto border border-line">
            {(active?.sample_decisions ?? []).map((item, index) => {
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
                    loadDetail(item);
                  }}
                  className={`grid w-full grid-cols-[92px_54px_1fr] gap-2 border-b border-line px-3 py-2 text-left text-xs ${
                    isActive ? "bg-accent/10 text-ink" : "bg-panel-strong text-muted hover:bg-panel"
                  }`}
                >
                  <span>{asText(item.date)}</span>
                  <span>F{asText(item.fold)}</span>
                  <span className="truncate">{asText(item.candidate_id)}</span>
                </button>
              );
            })}
          </div>
        </Panel>

        <Panel
          title="Advanced Controls"
          icon={<Filter size={16} />}
          actions={
            <button
              onClick={() =>
                loadDetail({
                  date: manual.date,
                  fold: manual.fold,
                  candidate_id: manual.candidate_id,
                  universe_id: manual.universe_id,
                })
              }
              className="flex h-9 items-center gap-2 border border-accent px-3 text-sm text-ink"
            >
              <Search size={14} />
              Apply
            </button>
          }
        >
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-4">
            <label className="text-xs text-muted">
              Date
              <input value={manual.date} onChange={(event) => setManual({ ...manual, date: event.target.value })} className="mt-1 h-9 w-full border border-line bg-panel-strong px-2 text-sm text-ink outline-none focus:border-accent" />
            </label>
            <SelectField label="Candidate" value={manual.candidate_id} onChange={(value) => setManual({ ...manual, candidate_id: value })} options={options.candidates} allowEmpty={false} />
            <SelectField label="Universe" value={manual.universe_id} onChange={(value) => setManual({ ...manual, universe_id: value })} options={options.universes} allowEmpty={false} />
            <SelectField label="Fold" value={manual.fold} onChange={(value) => setManual({ ...manual, fold: value })} options={options.folds} />
            <SelectField label="Ticker" value={manual.ticker} onChange={(value) => setManual({ ...manual, ticker: value })} options={options.tickers} />
            <SelectField label="Module" value={manual.module_name} onChange={(value) => setManual({ ...manual, module_name: value })} options={options.modules} />
            <SelectField label="Horizon" value={manual.horizon} onChange={(value) => setManual({ ...manual, horizon: value })} options={options.horizons} />
          </div>
        </Panel>
      </div>

      <Panel title="Deterministic Interpretation" icon={<ShieldCheck size={16} />}>
        <BulletList items={detail.interpretation} />
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
          <DataTable rows={filteredPositions} limit={14} columns={["ticker", "rank", "base_score", "selected_flag", "leader_flag", "base_weight", "final_weight", "stop_flag", "pnl_contribution"]} />
        </Panel>
      </div>

      <Panel title="Outcome" icon={<BarChart3 size={16} />}>
        <DataTable rows={filteredOutcomes} limit={8} columns={["horizon", "realized_return", "realized_alpha_vs_qqq", "realized_alpha_vs_spy", "decision_helped_flag_vs_qqq", "decision_helped_flag_vs_control", "continuation_helped_flag", "backoff_helped_flag", "leader_helped_flag"]} />
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
    continuation: ["Continuation helped rates are grouped directly from outcome_cube by horizon."],
    leader: ["Leader participation helped rates are grouped directly from outcome_cube by horizon."],
    backoff: [
      `Backoff count: ${formatMetric(moduleData.backoff_counts.backoff_count)}.`,
      `Hard backoff count: ${formatMetric(moduleData.backoff_counts.hard_backoff_count)}.`,
    ],
    tickers: ["Ticker contribution is aggregated from selected position rows and realized PnL contribution fields."],
    folds: ["Fold behavior uses helped-rate and alpha aggregates by fold and outcome horizon."],
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
