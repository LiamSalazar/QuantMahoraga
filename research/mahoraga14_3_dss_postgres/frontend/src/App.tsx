import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  BarChart3,
  Boxes,
  Braces,
  CalendarDays,
  ChevronsUpDown,
  Database,
  GitBranch,
  Gauge,
  Layers3,
  LineChart as LineIcon,
  ListFilter,
  MonitorCog,
  Network,
  Play,
  RefreshCcw,
  ScanSearch,
  SlidersHorizontal,
  Table2,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8010";
const OFFICIAL = "B1.05_C1.10_L1.10_R1.05";
const UNIVERSE = "base_universe_12";

type Row = Record<string, any>;
type ViewKey = "overview" | "whatif" | "robustness" | "replay" | "slice" | "modules" | "tickers" | "regimes" | "performance";

type Options = {
  candidates: string[];
  universes: string[];
  folds: number[];
  tickers: string[];
  modules: string[];
  horizons: number[];
  regimes: string[];
  metrics: string[];
  benchmarks: string[];
  date_range: { start: string | null; end: string | null };
  slider_ranges: Record<string, { min: number; max: number; values: number[] }>;
  default_candidate: string;
  default_universe: string;
};

const emptyOptions: Options = {
  candidates: [OFFICIAL],
  universes: [UNIVERSE],
  folds: [1, 2, 3, 4, 5],
  tickers: [],
  modules: [],
  horizons: [1, 5, 20, 60],
  regimes: [],
  metrics: ["CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "robust_score"],
  benchmarks: ["QQQ", "SPY", "CONTROL"],
  date_range: { start: null, end: null },
  slider_ranges: {},
  default_candidate: OFFICIAL,
  default_universe: UNIVERSE,
};

const navItems: { key: ViewKey; label: string; icon: ReactNode }[] = [
  { key: "overview", label: "Overview", icon: <Gauge size={17} /> },
  { key: "whatif", label: "What-if Lab", icon: <SlidersHorizontal size={17} /> },
  { key: "robustness", label: "Robustness Surface", icon: <Activity size={17} /> },
  { key: "replay", label: "Decision Replay", icon: <Play size={17} /> },
  { key: "slice", label: "Slice & Dice", icon: <Braces size={17} /> },
  { key: "modules", label: "Module Lab", icon: <Layers3 size={17} /> },
  { key: "tickers", label: "Ticker Contribution", icon: <BarChart3 size={17} /> },
  { key: "regimes", label: "Regime Lab", icon: <Network size={17} /> },
  { key: "performance", label: "Query Performance", icon: <MonitorCog size={17} /> },
];

function valueLabel(value: any, digits = 3): string {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "boolean") return value ? "yes" : "no";
  const n = Number(value);
  if (Number.isFinite(n)) {
    if (Math.abs(n) > 100) return n.toFixed(1);
    if (Math.abs(n) > 10) return n.toFixed(2);
    return n.toFixed(digits);
  }
  return String(value);
}

function percent(value: any): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "n/a";
  return `${(n * 100).toFixed(1)}%`;
}

function compactId(id: string): string {
  if (!id) return "n/a";
  if (id === OFFICIAL) return "Official";
  return id.replace("B", "B ").replace("_C", " C").replace("_L", " L").replace("_R", " R").replace("EXTREME_", "Extreme ");
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function params(input: Record<string, any>): string {
  const q = new URLSearchParams();
  Object.entries(input).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "" || value === "all") return;
    if (Array.isArray(value)) value.forEach((item) => q.append(key, String(item)));
    else q.set(key, String(value));
  });
  const out = q.toString();
  return out ? `?${out}` : "";
}

function useApi<T>(path: string, fallback: T): [T, boolean, string | null] {
  const [data, setData] = useState<T>(fallback);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchJson<T>(path)
      .then((next) => {
        if (!cancelled) {
          setData(next);
          setError(null);
        }
      })
      .catch((exc) => {
        if (!cancelled) setError(String(exc));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path]);
  return [data, loading, error];
}

function Select({
  label,
  value,
  options,
  onChange,
  compact = false,
}: {
  label: string;
  value: string | number;
  options: (string | number)[];
  onChange: (value: string) => void;
  compact?: boolean;
}) {
  return (
    <label className={compact ? "control compact" : "control"}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option).length > 28 ? compactId(String(option)) : String(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function DateControl({ label, value, min, max, onChange }: { label: string; value: string; min?: string | null; max?: string | null; onChange: (value: string) => void }) {
  return (
    <label className="control compact">
      <span>{label}</span>
      <input type="date" value={value} min={min ?? undefined} max={max ?? undefined} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider-control">
      <span>
        {label}
        <b>{valueLabel(value, 2)}</b>
      </span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function Segments({ value, options, onChange }: { value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <div className="segments">
      {options.map((option) => (
        <button key={option} className={value === option ? "active" : ""} onClick={() => onChange(option)}>
          {option}
        </button>
      ))}
    </div>
  );
}

function StatStrip({ rows }: { rows: { label: string; value: any; tone?: string }[] }) {
  return (
    <div className="stat-strip">
      {rows.map((row) => (
        <div className="stat" key={row.label}>
          <span>{row.label}</span>
          <strong className={row.tone ?? ""}>{row.value}</strong>
        </div>
      ))}
    </div>
  );
}

function Panel({ title, icon, children, action }: { title: string; icon?: ReactNode; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>
          {icon}
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function DataTable({ rows, columns, pageSize = 12 }: { rows: Row[]; columns: string[]; pageSize?: number }) {
  const [sortKey, setSortKey] = useState(columns[0] ?? "");
  const [desc, setDesc] = useState(true);
  const [page, setPage] = useState(0);
  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const an = Number(av);
      const bn = Number(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) return desc ? bn - an : an - bn;
      return desc ? String(bv ?? "").localeCompare(String(av ?? "")) : String(av ?? "").localeCompare(String(bv ?? ""));
    });
    return copy;
  }, [rows, sortKey, desc]);
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageRows = sorted.slice(page * pageSize, page * pageSize + pageSize);
  useEffect(() => setPage(0), [rows.length, sortKey]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>
                <button
                  onClick={() => {
                    if (sortKey === column) setDesc(!desc);
                    else {
                      setSortKey(column);
                      setDesc(true);
                    }
                  }}
                >
                  {column}
                  <ChevronsUpDown size={13} />
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pageRows.map((row, idx) => (
            <tr key={`${idx}-${columns.map((c) => row[c]).join("|")}`}>
              {columns.map((column) => (
                <td key={column}>{valueLabel(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pager">
        <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>
          Prev
        </button>
        <span>
          {page + 1} / {totalPages}
        </span>
        <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}>
          Next
        </button>
      </div>
    </div>
  );
}

function Overview({ filters }: { filters: Filters }) {
  const query = params(filters);
  const [data] = useApi<any>(`/overview${query}`, { scorecard: [], equity_curve: [], exposure_turnover: [], decision_summary: {}, fold_performance: [] });
  const score = data.scorecard?.[0] ?? {};
  return (
    <div className="view-grid">
      <StatStrip
        rows={[
          { label: "CAGR", value: valueLabel(score.cagr, 2), tone: "green" },
          { label: "Sharpe", value: valueLabel(score.sharpe, 3), tone: "cyan" },
          { label: "MaxDD", value: valueLabel(score.maxdd, 2), tone: "amber" },
          { label: "Helped", value: percent(data.decision_summary?.helped_rate), tone: "violet" },
        ]}
      />
      <Panel title="Equity Curve" icon={<LineIcon size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <AreaChart data={data.equity_curve}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="date_value" minTickGap={42} stroke="#88909a" />
              <YAxis stroke="#88909a" domain={["auto", "auto"]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Area type="monotone" dataKey="equity" stroke="#7dd3a8" fill="#21382d" strokeWidth={2} legendType="none" isAnimationActive={false} />
              <Line type="monotone" dataKey="equity" dot={false} stroke="#7dd3a8" strokeWidth={2.6} isAnimationActive={false} />
              <Brush dataKey="date_value" height={18} stroke="#7dd3a8" travellerWidth={8} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Drawdown" icon={<ScanSearch size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <AreaChart data={data.equity_curve}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="date_value" minTickGap={42} stroke="#88909a" />
              <YAxis stroke="#88909a" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="drawdown" stroke="#f2b866" fill="#3a2e1f" strokeWidth={2} legendType="none" isAnimationActive={false} />
              <Line type="monotone" dataKey="drawdown" dot={false} stroke="#f2b866" strokeWidth={2.5} isAnimationActive={false} />
              <Brush dataKey="date_value" height={18} stroke="#f2b866" travellerWidth={8} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Exposure & Turnover" icon={<Activity size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <LineChart data={data.exposure_turnover}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="date_value" minTickGap={42} stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Line type="monotone" dataKey="expected_exposure" dot={false} stroke="#80d8ff" strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey="expected_turnover" dot={false} stroke="#c4a4ff" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Fold Performance" icon={<Table2 size={17} />}>
        <DataTable rows={data.fold_performance ?? []} columns={["fold", "avg_realized_return", "avg_alpha_vs_qqq", "helped_rate", "avg_exposure", "observations"]} />
      </Panel>
    </div>
  );
}

function WhatIfLab({ filters, options }: { filters: Filters; options: Options }) {
  const [budget, setBudget] = useState(1.05);
  const [conviction, setConviction] = useState(1.1);
  const [leader, setLeader] = useState(1.1);
  const [backoff, setBackoff] = useState(1.05);
  const [cost, setCost] = useState(5);
  const [slip, setSlip] = useState(2);
  const [horizon, setHorizon] = useState(20);
  const query = params({ candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id, horizon, cost_bps: cost, slippage_bps: slip, limit: 5000 });
  const [data] = useApi<any>(`/whatif/grid${query}`, { rows: [], pareto: [], demo_rows: 0 });
  const rows = data.rows ?? [];
  const selected = useMemo(() => {
    if (!rows.length) return null;
    return [...rows].sort((a, b) => {
      const da = Math.abs(a.budget_multiplier - budget) + Math.abs(a.conviction_multiplier - conviction) + Math.abs(a.leader_multiplier - leader) + Math.abs(a.backoff_strength - backoff);
      const db = Math.abs(b.budget_multiplier - budget) + Math.abs(b.conviction_multiplier - conviction) + Math.abs(b.leader_multiplier - leader) + Math.abs(b.backoff_strength - backoff);
      return da - db;
    })[0];
  }, [rows, budget, conviction, leader, backoff]);
  return (
    <div className="view-grid">
      <div className="lab-controls">
        <Slider label="Budget" value={budget} min={0.9} max={1.15} step={0.05} onChange={setBudget} />
        <Slider label="Conviction" value={conviction} min={0.9} max={1.3} step={0.1} onChange={setConviction} />
        <Slider label="Leader" value={leader} min={0.9} max={1.3} step={0.1} onChange={setLeader} />
        <Slider label="Backoff" value={backoff} min={0.9} max={1.2} step={0.05} onChange={setBackoff} />
        <Slider label="Cost bps" value={cost} min={0} max={20} step={5} onChange={setCost} />
        <Slider label="Slippage bps" value={slip} min={0} max={5} step={1} onChange={setSlip} />
        <Select label="Horizon" value={horizon} options={options.horizons} onChange={(v) => setHorizon(Number(v))} compact />
      </div>
      <StatStrip
        rows={[
          { label: "Nearest Sharpe", value: valueLabel(selected?.sharpe), tone: "cyan" },
          { label: "Nearest CAGR", value: valueLabel(selected?.cagr), tone: "green" },
          { label: "Nearest MaxDD", value: valueLabel(selected?.maxdd), tone: "amber" },
          { label: "Demo rows", value: data.demo_rows ?? 0, tone: "violet" },
        ]}
      />
      <Panel title="Sharpe Heatmap" icon={<Boxes size={17} />}>
        <Heatmap rows={rows} x="budget_multiplier" y="conviction_multiplier" z="sharpe" />
      </Panel>
      <Panel title="MaxDD Heatmap" icon={<Boxes size={17} />}>
        <Heatmap rows={rows} x="budget_multiplier" y="conviction_multiplier" z="maxdd" reverse />
      </Panel>
      <Panel title="Pareto Frontier" icon={<GitBranch size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid stroke="#262b31" />
              <XAxis dataKey="maxdd" name="MaxDD" stroke="#88909a" />
              <YAxis dataKey="cagr" name="CAGR" stroke="#88909a" />
              <ZAxis dataKey="robust_score" range={[60, 260]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={tooltipStyle} />
              <Scatter data={data.pareto ?? []} fill="#7dd3a8" isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Scenario Ranking" icon={<Table2 size={17} />}>
        <DataTable rows={rows} columns={["rank", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "cagr", "sharpe", "maxdd", "robust_score", "demo_mode"]} />
      </Panel>
    </div>
  );
}

function Robustness({ filters, options }: { filters: Filters; options: Options }) {
  const [metric, setMetric] = useState("Sharpe");
  const query = params({ metric, fold: filters.fold, universe_id: filters.universe_id, regime: filters.regime });
  const [data] = useApi<any>(`/robustness/surface${query}`, { rows: [] });
  const rows = data.rows ?? [];
  return (
    <div className="view-grid">
      <div className="local-controls">
        <Select label="Metric" value={metric} options={options.metrics} onChange={setMetric} />
        <Segments value={metric} options={["CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "robust_score"]} onChange={setMetric} />
      </div>
      <Panel title="Multiplier Surface" icon={<Activity size={17} />}>
        <Heatmap rows={rows} x="budget_multiplier" y="leader_multiplier" z="metric_value" reverse={metric === "MaxDD"} markOfficial />
      </Panel>
      <Panel title="Candidate Grid" icon={<Table2 size={17} />}>
        <DataTable rows={rows} columns={["candidate_id", "sweep_role", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "metric_name", "metric_value", "robust_score", "demo_mode"]} />
      </Panel>
    </div>
  );
}

function DecisionReplay({ filters, options }: { filters: Filters; options: Options }) {
  const [dateValue, setDateValue] = useState("");
  const [ticker, setTicker] = useState("");
  const query = params({ candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id, date: dateValue, ticker });
  const [data] = useApi<any>(`/decision/replay${query}`, { positions: [], modules: [], outcomes: [], market_context: [], timeline: [] });
  const decision = data.decision ?? {};
  return (
    <div className="view-grid">
      <div className="local-controls">
        <DateControl label="Decision date" value={dateValue} min={options.date_range.start} max={options.date_range.end} onChange={setDateValue} />
        <Select label="Ticker" value={ticker} options={["", ...options.tickers]} onChange={setTicker} compact />
      </div>
      <StatStrip
        rows={[
          { label: "Date", value: decision.date_value ?? "auto" },
          { label: "State", value: decision.participation_state ?? "n/a", tone: "violet" },
          { label: "Budget", value: valueLabel(decision.long_budget), tone: "green" },
          { label: "Backoff", value: valueLabel(decision.hard_backoff_flag), tone: "amber" },
        ]}
      />
      <Panel title="Decision Timeline" icon={<CalendarDays size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <LineChart data={data.timeline ?? []}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="date_value" minTickGap={40} stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Line dataKey="expected_exposure" dot={false} stroke="#80d8ff" strokeWidth={2} isAnimationActive={false} />
              <Line dataKey="drawdown" dot={false} stroke="#f2b866" strokeWidth={2} isAnimationActive={false} />
              <Brush dataKey="date_value" height={18} stroke="#80d8ff" />
            </LineChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Weights & Tickers" icon={<BarChart3 size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <BarChart data={(data.positions ?? []).slice(0, 16)}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="ticker" stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="final_weight" fill="#7dd3a8" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Module Trace" icon={<Layers3 size={17} />}>
        <DataTable rows={data.modules ?? []} columns={["module_name", "module_active", "intensity_score", "state_label", "raw_value"]} />
      </Panel>
      <Panel title="Outcomes" icon={<Table2 size={17} />}>
        <DataTable rows={data.outcomes ?? []} columns={["horizon", "realized_return", "alpha_vs_qqq", "alpha_vs_spy", "helped_flag", "realized_exposure"]} />
      </Panel>
    </div>
  );
}

function SliceDice({ filters, options }: { filters: Filters; options: Options }) {
  const [dims, setDims] = useState<string[]>(["candidate_id", "fold"]);
  const [measure, setMeasure] = useState("alpha");
  const [operation, setOperation] = useState("slice");
  const [module, setModule] = useState("");
  const [ticker, setTicker] = useState("");
  const [horizon, setHorizon] = useState(20);
  const query = params({ dimensions: dims, measure, operation, candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id, module, ticker, regime: filters.regime, horizon });
  const [data] = useApi<any>(`/slice${query}`, { rows: [] });
  const dimChoices = ["candidate_id", "fold", "universe_id", "ticker", "module_name", "regime", "horizon"];
  return (
    <div className="view-grid">
      <div className="dimension-grid">
        {dimChoices.map((dim) => (
          <label key={dim} className="check">
            <input
              type="checkbox"
              checked={dims.includes(dim)}
              onChange={(event) => setDims(event.target.checked ? [...dims, dim] : dims.filter((item) => item !== dim))}
            />
            <span>{dim}</span>
          </label>
        ))}
      </div>
      <div className="local-controls">
        <Select label="Measure" value={measure} options={["return", "alpha", "drawdown", "exposure", "turnover", "helped_rate"]} onChange={setMeasure} compact />
        <Select label="Operation" value={operation} options={["slice", "dice", "roll-up", "drill-down", "pivot"]} onChange={setOperation} compact />
        <Select label="Module" value={module} options={["", ...options.modules]} onChange={setModule} compact />
        <Select label="Ticker" value={ticker} options={["", ...options.tickers]} onChange={setTicker} compact />
        <Select label="Horizon" value={horizon} options={options.horizons} onChange={(v) => setHorizon(Number(v))} compact />
      </div>
      <Panel title="Dynamic Cube" icon={<Database size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <BarChart data={(data.rows ?? []).slice(0, 24)}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey={dims[0]} stroke="#88909a" tickFormatter={(v) => String(v).slice(0, 16)} />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey={measure} fill="#c4a4ff" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Pivot Table" icon={<Table2 size={17} />}>
        <DataTable rows={data.rows ?? []} columns={[...dims, measure, "observations"].filter(Boolean)} />
      </Panel>
    </div>
  );
}

function ModuleLab({ filters }: { filters: Filters }) {
  const query = params({ candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id });
  const [data] = useApi<any>(`/module/effectiveness${query}`, { activation: [], by_horizon: [], timeline: [] });
  return (
    <div className="view-grid">
      <Panel title="Activation Timeline" icon={<Activity size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <LineChart data={data.timeline ?? []}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="date_value" minTickGap={42} stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Line dataKey="activation_rate" stroke="#7dd3a8" dot={false} isAnimationActive={false} />
              <Brush dataKey="date_value" height={18} stroke="#7dd3a8" />
            </LineChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Module x Horizon" icon={<Layers3 size={17} />}>
        <DataTable rows={data.by_horizon ?? []} columns={["module_name", "horizon", "activation_rate", "helped_rate", "avg_alpha_vs_qqq", "observations"]} />
      </Panel>
      <Panel title="Activation Rate" icon={<BarChart3 size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <BarChart data={data.activation ?? []}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="module_name" stroke="#88909a" tickFormatter={(v) => String(v).replace("_model", "").slice(0, 16)} />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="activation_rate" fill="#80d8ff" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
    </div>
  );
}

function TickerContribution({ filters }: { filters: Filters }) {
  const query = params({ candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id });
  const [data] = useApi<any>(`/ticker/contribution${query}`, { rows: [] });
  const rows = data.rows ?? [];
  return (
    <div className="view-grid">
      <Panel title="Contribution" icon={<BarChart3 size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <BarChart data={rows.slice(0, 18)}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="ticker" stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="total_pnl_contribution" fill="#7dd3a8" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Selection x Leader" icon={<Activity size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid stroke="#262b31" />
              <XAxis dataKey="selection_rate" stroke="#88909a" />
              <YAxis dataKey="leader_flag_rate" stroke="#88909a" />
              <ZAxis dataKey="total_pnl_contribution" range={[80, 320]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Scatter data={rows} fill="#c4a4ff" isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Ticker Table" icon={<Table2 size={17} />}>
        <DataTable rows={rows} columns={["ticker", "total_pnl_contribution", "selection_rate", "leader_flag_rate", "avg_final_weight", "avg_score", "observations"]} />
      </Panel>
    </div>
  );
}

function RegimeLab({ filters }: { filters: Filters }) {
  const query = params({ candidate_id: filters.candidate_id, fold: filters.fold, universe_id: filters.universe_id });
  const [data] = useApi<any>(`/regime/behavior${query}`, { rows: [] });
  const rows = data.rows ?? [];
  return (
    <div className="view-grid">
      <Panel title="Regime Matrix" icon={<Network size={17} />}>
        <ChartBox>
          <ResponsiveContainer>
            <BarChart data={rows}>
              <CartesianGrid stroke="#262b31" vertical={false} />
              <XAxis dataKey="regime" stroke="#88909a" />
              <YAxis stroke="#88909a" />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Bar dataKey="avg_exposure" fill="#80d8ff" isAnimationActive={false} />
              <Bar dataKey="backoff_activation" fill="#f2b866" isAnimationActive={false} />
              <Bar dataKey="continuation_activation" fill="#7dd3a8" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </Panel>
      <Panel title="Regime Table" icon={<Table2 size={17} />}>
        <DataTable rows={rows} columns={["regime", "participation_state", "avg_net_return", "avg_benchmark_return", "avg_exposure", "avg_drawdown", "backoff_activation", "continuation_activation", "avg_leader_blend", "observations"]} />
      </Panel>
    </div>
  );
}

function QueryPerformance() {
  const [data] = useApi<any>("/query/performance", { rows: [] });
  return (
    <div className="view-grid">
      <Panel title="Query Performance" icon={<MonitorCog size={17} />}>
        <DataTable rows={data.rows ?? []} columns={["endpoint", "backend", "source_relation", "query_count", "avg_elapsed_ms", "p95_elapsed_ms", "avg_rows_returned", "used_materialized_view", "last_seen_at"]} />
      </Panel>
    </div>
  );
}

function Heatmap({ rows, x, y, z, reverse = false, markOfficial = false }: { rows: Row[]; x: string; y: string; z: string; reverse?: boolean; markOfficial?: boolean }) {
  const values = rows.map((row) => Number(row[z])).filter(Number.isFinite);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  return (
    <ChartBox>
      <ResponsiveContainer>
        <ScatterChart>
          <CartesianGrid stroke="#262b31" />
          <XAxis dataKey={x} type="number" stroke="#88909a" domain={["dataMin", "dataMax"]} />
          <YAxis dataKey={y} type="number" stroke="#88909a" domain={["dataMin", "dataMax"]} />
          <ZAxis dataKey={z} range={[120, 420]} />
          <Tooltip contentStyle={tooltipStyle} />
          <Scatter data={rows} isAnimationActive={false}>
            {rows.map((entry, index) => {
              const v = Number(entry[z]);
              const t = max === min ? 0.5 : (v - min) / (max - min);
              const score = reverse ? 1 - t : t;
              const color = score > 0.66 ? "#7dd3a8" : score > 0.33 ? "#f2b866" : "#c66f8a";
              const official = markOfficial && entry.candidate_id === OFFICIAL;
              return <Cell key={`cell-${index}`} fill={official ? "#80d8ff" : color} stroke={official ? "#f7fbff" : "transparent"} strokeWidth={official ? 2 : 0} />;
            })}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </ChartBox>
  );
}

function ChartBox({ children }: { children: ReactNode }) {
  return <div className="chart-box">{children}</div>;
}

type Filters = {
  candidate_id: string;
  fold?: number;
  universe_id: string;
  benchmark: string;
  start_date?: string;
  end_date?: string;
  regime?: string;
};

const tooltipStyle = {
  background: "#11161b",
  border: "1px solid #303740",
  borderRadius: 8,
  color: "#e8ecef",
};

export default function App() {
  const [options] = useApi<Options>("/metadata/options", emptyOptions);
  const [health] = useApi<any>("/health", { backend: "parquet", demo_mode: true, row_counts: {} });
  const [view, setView] = useState<ViewKey>("overview");
  const [candidate, setCandidate] = useState(OFFICIAL);
  const [fold, setFold] = useState<number | undefined>(undefined);
  const [universe, setUniverse] = useState(UNIVERSE);
  const [benchmark, setBenchmark] = useState("QQQ");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [regime, setRegime] = useState("");

  useEffect(() => {
    if (options.default_candidate) setCandidate(options.default_candidate);
    if (options.default_universe) setUniverse(options.default_universe);
  }, [options.default_candidate, options.default_universe]);

  const filters: Filters = {
    candidate_id: candidate,
    fold,
    universe_id: universe,
    benchmark,
    start_date: start || undefined,
    end_date: end || undefined,
    regime: regime || undefined,
  };

  const activeView = {
    overview: <Overview filters={filters} />,
    whatif: <WhatIfLab filters={filters} options={options} />,
    robustness: <Robustness filters={filters} options={options} />,
    replay: <DecisionReplay filters={filters} options={options} />,
    slice: <SliceDice filters={filters} options={options} />,
    modules: <ModuleLab filters={filters} />,
    tickers: <TickerContribution filters={filters} />,
    regimes: <RegimeLab filters={filters} />,
    performance: <QueryPerformance />,
  }[view];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Database size={21} />
          <div>
            <strong>Mahoraga Quant DSS</strong>
            <span>backend: {health.backend ?? "parquet"}{health.demo_mode ? "/demo" : ""}</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button key={item.key} className={view === item.key ? "active" : ""} onClick={() => setView(item.key)}>
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <span>{Number(health.row_counts?.fact_position_daily ?? 0).toLocaleString()} positions</span>
          <span>{Number(health.row_counts?.fact_outcome ?? 0).toLocaleString()} outcomes</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <h1>{navItems.find((item) => item.key === view)?.label}</h1>
            <span>{compactId(candidate)} / {universe}</span>
          </div>
          <button className="icon-button" onClick={() => window.location.reload()} title="Refresh">
            <RefreshCcw size={17} />
          </button>
        </header>
        <section className="filter-bar">
          <Select label="Candidate" value={candidate} options={options.candidates} onChange={setCandidate} />
          <Select label="Fold" value={fold ?? "all"} options={["all", ...options.folds]} onChange={(v) => setFold(v === "all" ? undefined : Number(v))} compact />
          <Select label="Universe" value={universe} options={options.universes} onChange={setUniverse} compact />
          <Select label="Benchmark" value={benchmark} options={options.benchmarks} onChange={setBenchmark} compact />
          <Select label="Regime" value={regime} options={["", ...options.regimes]} onChange={setRegime} compact />
          <DateControl label="Start" value={start} min={options.date_range.start} max={options.date_range.end} onChange={setStart} />
          <DateControl label="End" value={end} min={options.date_range.start} max={options.date_range.end} onChange={setEnd} />
        </section>
        {activeView}
      </main>
    </div>
  );
}
