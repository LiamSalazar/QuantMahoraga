import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { clearApiCache } from "./api/client";
import type { HealthSummary, Options, ViewKey } from "./api/types";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Layout } from "./components/Layout";
import { LoadingState } from "./components/States";
import { useApiResource } from "./hooks/useApiResource";

const CommandCenter = lazy(() => import("./views/CommandCenter"));
const BaselineEvidence = lazy(() => import("./views/BaselineEvidence"));
const RobustnessLab = lazy(() => import("./views/RobustnessLab"));
const WhatIfLab = lazy(() => import("./views/WhatIfLab"));
const DecisionReplay = lazy(() => import("./views/DecisionReplay"));
const ModuleAttribution = lazy(() => import("./views/ModuleAttribution"));
const TickerContribution = lazy(() => import("./views/TickerContribution"));
const RegimeAnalysis = lazy(() => import("./views/RegimeAnalysis"));
const OLAPExplorer = lazy(() => import("./views/OLAPExplorer"));
const DataEngineering = lazy(() => import("./views/DataEngineering"));

function ActiveView({ active, options, setActive }: { active: ViewKey; options: Options | null; setActive: (view: ViewKey) => void }) {
  const props = { options };
  switch (active) {
    case "baseline":
      return <BaselineEvidence />;
    case "robustness":
      return <RobustnessLab options={options} />;
    case "whatif":
      return <WhatIfLab options={options} />;
    case "replay":
      return <DecisionReplay options={options} />;
    case "modules":
      return <ModuleAttribution options={options} />;
    case "tickers":
      return <TickerContribution options={options} onOpenView={setActive} />;
    case "regimes":
      return <RegimeAnalysis options={options} />;
    case "olap":
      return <OLAPExplorer options={options} onOpenView={setActive} />;
    case "engineering":
      return <DataEngineering />;
    case "command":
    default:
      return <CommandCenter {...props} onOpenView={setActive} />;
  }
}

export default function App() {
  const [active, setActiveState] = useState<ViewKey>(() => (location.hash.replace("#", "") as ViewKey) || "command");
  const [cacheNonce, setCacheNonce] = useState(0);
  const health = useApiResource<HealthSummary>("/data/health-summary", { cacheNonce }, true, false);
  const options = useApiResource<Options>("/metadata/options", undefined, true, true);
  const stableOptions = useMemo(() => options.data, [options.data]);
  const setActive = (view: ViewKey) => {
    setActiveState(view);
    history.replaceState(null, "", `#${view}`);
  };

  useEffect(() => {
    const onHash = () => {
      const next = (location.hash.replace("#", "") || "command") as ViewKey;
      setActiveState(next);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <Layout
      active={active}
      setActive={setActive}
      health={health.data}
      options={stableOptions}
      onRefresh={() => {
        clearApiCache();
        setCacheNonce((value) => value + 1);
      }}
    >
      <ErrorBoundary key={active}>
        <Suspense fallback={<LoadingState label="Loading view module" />}>
          <ActiveView active={active} options={stableOptions} setActive={setActive} />
        </Suspense>
      </ErrorBoundary>
    </Layout>
  );
}
