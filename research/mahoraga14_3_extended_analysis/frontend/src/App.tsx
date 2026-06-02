import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Activity, Database, Filter, LineChart, RefreshCcw, Search, ShieldCheck, Table2 } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type ApiRows = {
  count: number;
  rows: Record<string, unknown>[];
};

type BaselineSummary = {
  official: Record<string, unknown> | null;
  robust_region_share_extended: number | null;
  sampled_candidates: number;
  universe_runs: number;
  figures: Record<string, string>;
};

type PlateauResponse = {
  plateau: Record<string, unknown>[];
  sensitivity: Record<string, unknown>[];
  report: string;
};

type ViewKey = "overview" | "robustness" | "audit";

const formatValue = (value: unknown) => {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "";
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 10) return value.toFixed(3);
    return value.toFixed(4);
  }
  return String(value);
};

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function Section({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-t border-line py-5">
      <div className="mb-4 flex items-center gap-2">
        {icon}
        <h2 className="text-sm font-semibold uppercase tracking-normal text-ink">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function MetricStrip({ row }: { row: Record<string, unknown> | null }) {
  const metrics = [
    ["CAGR", "%"],
    ["Sharpe", ""],
    ["Sortino", ""],
    ["MaxDD", "%"],
    ["AlphaNW_QQQ", ""],
    ["AlphaNW_SPY", ""],
    ["UpsideCaptureQQQ", ""],
    ["DownsideCaptureQQQ", ""],
  ];

  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden border border-line bg-line md:grid-cols-4">
      {metrics.map(([key, suffix]) => (
        <div key={key} className="bg-white p-3">
          <div className="text-xs text-muted">{key}</div>
          <div className="mt-1 text-lg font-semibold text-ink">
            {row ? `${formatValue(row[key])}${suffix}` : ""}
          </div>
        </div>
      ))}
    </div>
  );
}

function DataTable({ rows, limit = 12 }: { rows: Record<string, unknown>[]; limit?: number }) {
  const displayRows = rows.slice(0, limit);
  const columns = useMemo(() => {
    const preferred = [
      "CandidateId",
      "candidate_id",
      "universe_id",
      "sweep_role",
      "module_name",
      "date",
      "decision_date",
      "ticker",
      "fold",
      "CAGR",
      "Sharpe",
      "Sortino",
      "MaxDD",
      "robust_region_flag",
      "participation_state",
      "long_budget",
      "leader_blend",
      "backoff_strength_applied",
      "signal_strength",
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
  }, [displayRows]);

  if (!displayRows.length) {
    return <div className="border border-line bg-panel p-4 text-sm text-muted">No rows loaded.</div>;
  }

  return (
    <div className="table-scroll overflow-auto border border-line">
      <table className="min-w-full border-collapse text-left text-xs">
        <thead className="bg-panel text-muted">
          <tr>
            {columns.map((col) => (
              <th key={col} className="whitespace-nowrap border-b border-line px-3 py-2 font-medium">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, idx) => (
            <tr key={idx} className="odd:bg-white even:bg-panel/60">
              {columns.map((col) => (
                <td key={col} className="whitespace-nowrap border-b border-line px-3 py-2 text-ink">
                  {formatValue(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Overview() {
  const [summary, setSummary] = useState<BaselineSummary | null>(null);
  const [universes, setUniverses] = useState<ApiRows>({ count: 0, rows: [] });
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchJson<BaselineSummary>("/summary/baseline"), fetchJson<ApiRows>("/universes/summary?limit=20")])
      .then(([s, u]) => {
        setSummary(s);
        setUniverses(u);
      })
      .catch((err) => setError(String(err)));
  }, []);

  return (
    <>
      {error && <div className="mb-4 border border-risk/30 bg-red-50 p-3 text-sm text-risk">{error}</div>}
      <Section title="Baseline Overview" icon={<ShieldCheck size={16} />}>
        <MetricStrip row={summary?.official ?? null} />
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="border border-line p-4">
            <div className="text-xs text-muted">Promotion Reference</div>
            <div className="mt-2 text-sm font-medium text-ink">B1.05_C1.10_L1.10_R1.05</div>
            <div className="mt-1 text-xs text-muted">Mahoraga14_3R / ROBUST_MAIN</div>
          </div>
          <div className="border border-line p-4">
            <div className="text-xs text-muted">Sampled Candidates</div>
            <div className="mt-2 text-2xl font-semibold">{summary?.sampled_candidates ?? ""}</div>
          </div>
          <div className="border border-line p-4">
            <div className="text-xs text-muted">Robust Region Share</div>
            <div className="mt-2 text-2xl font-semibold">
              {summary?.robust_region_share_extended !== null && summary ? `${(summary.robust_region_share_extended * 100).toFixed(1)}%` : ""}
            </div>
          </div>
        </div>
      </Section>
      <Section title="Universe Snapshot" icon={<Database size={16} />}>
        <DataTable rows={universes.rows} limit={10} />
      </Section>
      <Section title="Generated Figures" icon={<LineChart size={16} />}>
        <div className="grid gap-4 md:grid-cols-2">
          <img className="w-full border border-line bg-white" src={`${API_BASE}/figures/extended_multiplier_heatmap.png`} alt="Multiplier heatmap" />
          <img className="w-full border border-line bg-white" src={`${API_BASE}/figures/universe_robustness_comparison.png`} alt="Universe robustness" />
        </div>
      </Section>
    </>
  );
}

function Robustness() {
  const [axis, setAxis] = useState("");
  const [robustOnly, setRobustOnly] = useState(false);
  const [rows, setRows] = useState<ApiRows>({ count: 0, rows: [] });
  const [plateau, setPlateau] = useState<PlateauResponse | null>(null);

  const load = () => {
    const params = new URLSearchParams();
    if (axis) params.set("axis", axis);
    if (robustOnly) params.set("robust_only", "true");
    params.set("limit", "300");
    fetchJson<ApiRows>(`/robustness/multipliers?${params.toString()}`).then(setRows);
    fetchJson<PlateauResponse>("/robustness/plateau").then(setPlateau);
  };

  useEffect(load, []);

  return (
    <>
      <Section title="Multiplier Robustness" icon={<Activity size={16} />}>
        <div className="mb-4 flex flex-wrap items-end gap-3">
          <label className="text-xs text-muted">
            Axis
            <select value={axis} onChange={(event) => setAxis(event.target.value)} className="mt-1 block h-9 border border-line bg-white px-2 text-sm text-ink">
              <option value="">All</option>
              <option value="budget_multiplier">Budget</option>
              <option value="conviction_multiplier">Conviction</option>
              <option value="leader_multiplier">Leader</option>
              <option value="backoff_strength">Backoff</option>
            </select>
          </label>
          <label className="flex h-9 items-center gap-2 border border-line px-3 text-sm">
            <input type="checkbox" checked={robustOnly} onChange={(event) => setRobustOnly(event.target.checked)} />
            Robust only
          </label>
          <button onClick={load} className="flex h-9 items-center gap-2 border border-ink px-3 text-sm font-medium">
            <RefreshCcw size={15} />
            Refresh
          </button>
        </div>
        <DataTable rows={rows.rows} limit={18} />
      </Section>
      <Section title="Plateau And Sensitivity" icon={<Table2 size={16} />}>
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-2 text-xs font-medium uppercase text-muted">Plateau radius</div>
            <DataTable rows={plateau?.plateau ?? []} limit={8} />
          </div>
          <div>
            <div className="mb-2 text-xs font-medium uppercase text-muted">Sensitivity ranking</div>
            <DataTable rows={plateau?.sensitivity ?? []} limit={8} />
          </div>
        </div>
      </Section>
      <Section title="Robustness Figures" icon={<LineChart size={16} />}>
        <div className="grid gap-4 lg:grid-cols-2">
          <img className="w-full border border-line bg-white" src={`${API_BASE}/figures/extended_multiplier_heatmap.png`} alt="Extended multiplier heatmap" />
          <img className="w-full border border-line bg-white" src={`${API_BASE}/figures/multiplier_1d_degradation.png`} alt="One-dimensional degradation" />
        </div>
      </Section>
    </>
  );
}

function AuditExplorer() {
  const [mode, setMode] = useState<"decisions" | "positions" | "module-trace" | "market-context">("decisions");
  const [filters, setFilters] = useState({
    date_start: "",
    date_end: "",
    fold: "",
    candidate_id: "B1.05_C1.10_L1.10_R1.05",
    ticker: "",
    module_name: "",
  });
  const [rows, setRows] = useState<ApiRows>({ count: 0, rows: [] });

  const load = () => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (!value) return;
      if (mode === "market-context" && !["date_start", "date_end"].includes(key)) return;
      if (mode !== "positions" && key === "ticker") return;
      if (mode !== "module-trace" && key === "module_name") return;
      params.set(key, String(value));
    });
    params.set("limit", "500");
    fetchJson<ApiRows>(`/${mode}?${params.toString()}`).then(setRows);
  };

  useEffect(load, [mode]);

  return (
    <>
      <Section title="Decision Audit Explorer" icon={<Filter size={16} />}>
        <div className="mb-4 flex flex-wrap gap-2">
          {(["decisions", "positions", "module-trace", "market-context"] as const).map((item) => (
            <button
              key={item}
              onClick={() => setMode(item)}
              className={`h-9 border px-3 text-sm ${mode === item ? "border-ink bg-ink text-white" : "border-line bg-white text-ink"}`}
            >
              {item}
            </button>
          ))}
        </div>
        <div className="mb-4 grid gap-3 md:grid-cols-3 lg:grid-cols-6">
          <input className="h-9 border border-line px-2 text-sm" placeholder="Date start" value={filters.date_start} onChange={(e) => setFilters({ ...filters, date_start: e.target.value })} />
          <input className="h-9 border border-line px-2 text-sm" placeholder="Date end" value={filters.date_end} onChange={(e) => setFilters({ ...filters, date_end: e.target.value })} />
          <input className="h-9 border border-line px-2 text-sm" placeholder="Fold" value={filters.fold} onChange={(e) => setFilters({ ...filters, fold: e.target.value })} />
          <input className="h-9 border border-line px-2 text-sm" placeholder="Candidate" value={filters.candidate_id} onChange={(e) => setFilters({ ...filters, candidate_id: e.target.value })} />
          <input className="h-9 border border-line px-2 text-sm" placeholder="Ticker" value={filters.ticker} onChange={(e) => setFilters({ ...filters, ticker: e.target.value.toUpperCase() })} />
          <input className="h-9 border border-line px-2 text-sm" placeholder="Module" value={filters.module_name} onChange={(e) => setFilters({ ...filters, module_name: e.target.value })} />
        </div>
        <button onClick={load} className="mb-4 flex h-9 items-center gap-2 border border-ink px-3 text-sm font-medium">
          <Search size={15} />
          Query
        </button>
        <div className="mb-2 text-xs text-muted">{rows.count} rows matched; showing first page.</div>
        <DataTable rows={rows.rows} limit={24} />
      </Section>
    </>
  );
}

export default function App() {
  const [view, setView] = useState<ViewKey>("overview");

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-line">
        <div className="mx-auto max-w-7xl px-5 py-5">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="text-xs font-medium uppercase text-muted">Mahoraga 14.3 Extended Analysis</div>
              <h1 className="mt-1 text-2xl font-semibold tracking-normal text-ink">Robustness And Decision Audit</h1>
            </div>
            <nav className="flex flex-wrap gap-2">
              {[
                ["overview", "Baseline Overview"],
                ["robustness", "Multiplier Robustness"],
                ["audit", "Decision Audit"],
              ].map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setView(key as ViewKey)}
                  className={`h-9 border px-3 text-sm ${view === key ? "border-ink bg-ink text-white" : "border-line bg-white text-ink"}`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-3">
        {view === "overview" && <Overview />}
        {view === "robustness" && <Robustness />}
        {view === "audit" && <AuditExplorer />}
      </main>
    </div>
  );
}
