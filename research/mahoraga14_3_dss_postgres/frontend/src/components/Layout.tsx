import { Activity, BarChart3, Database, Gauge, GitBranch, Layers3, MonitorCog, Network, Play, RefreshCcw, Search, ShieldCheck, SlidersHorizontal, Table2 } from "lucide-react";
import type { HealthSummary, NavItem, Options, ViewKey } from "../api/types";
import { OFFICIAL_LABEL } from "../utils/labels";

const navItems: NavItem[] = [
  { key: "command", label: "Command Center", icon: <Gauge size={17} /> },
  { key: "baseline", label: "Baseline Evidence", icon: <ShieldCheck size={17} /> },
  { key: "robustness", label: "Robustness Lab", icon: <Activity size={17} /> },
  { key: "whatif", label: "What-if & Stress", icon: <SlidersHorizontal size={17} /> },
  { key: "replay", label: "Decision Replay", icon: <Play size={17} /> },
  { key: "modules", label: "Module Attribution", icon: <Layers3 size={17} /> },
  { key: "tickers", label: "Ticker Contribution", icon: <BarChart3 size={17} /> },
  { key: "regimes", label: "Regime Analysis", icon: <Network size={17} /> },
  { key: "olap", label: "OLAP Explorer", icon: <Table2 size={17} /> },
  { key: "engineering", label: "Data Engineering", icon: <MonitorCog size={17} /> },
];

export function Layout({
  active,
  setActive,
  health,
  options,
  onRefresh,
  children,
}: {
  active: ViewKey;
  setActive: (view: ViewKey) => void;
  health: HealthSummary | null;
  options: Options | null;
  onRefresh: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <GitBranch />
          <div>
            <strong>Mahoraga Quant</strong>
            <span>Research Command Center</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button key={item.key} className={active === item.key ? "active" : ""} onClick={() => setActive(item.key)}>
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <span>Backend</span>
          <strong>{health?.backend ?? "connecting"}</strong>
          <small>{health?.contains_simulated_whatif ? "audited artifacts + flagged simulated what-if" : "audited artifacts"}</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <span>Official candidate</span>
            <h1>{OFFICIAL_LABEL}</h1>
            <small>B1.05_C1.10_L1.10_R1.05 · {options?.default_universe ?? "base_universe_12"}</small>
          </div>
          <button className="icon-button" onClick={onRefresh} title="Clear frontend cache and retry">
            <RefreshCcw size={18} />
          </button>
        </header>
        <Search className="watermark" />
        {children}
      </main>
    </div>
  );
}
