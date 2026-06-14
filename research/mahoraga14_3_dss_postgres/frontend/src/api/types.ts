import type { ReactNode } from "react";

export type Row = Record<string, unknown>;

export type ApiRows<T extends Row = Row> = {
  count?: number;
  rows: T[];
  [key: string]: unknown;
};

export type Options = {
  candidates: string[];
  universes: string[];
  folds: number[];
  tickers: string[];
  modules: string[];
  horizons: number[];
  regimes: string[];
  metrics: string[];
  benchmarks: string[];
  date_range?: { start?: string | null; end?: string | null } | null;
  slider_ranges?: Record<string, { min: number; max: number; values: number[] }>;
  default_candidate?: string;
  default_universe?: string;
};

export type HealthSummary = {
  ok: boolean;
  backend: string;
  profile?: string;
  row_counts: Record<string, number>;
  logical_counts?: Record<string, number>;
  real_rows?: number;
  simulated_rows?: number;
  contains_simulated_whatif?: boolean;
  latest_run_id?: string | null;
  marts_available?: string[];
  query_logs_active?: boolean;
  query_log_count?: number;
  validation_passed?: boolean | null;
  row_origin_note?: string;
};

export type ViewKey =
  | "command"
  | "baseline"
  | "robustness"
  | "whatif"
  | "replay"
  | "modules"
  | "tickers"
  | "regimes"
  | "olap"
  | "engineering";

export type NavItem = {
  key: ViewKey;
  label: string;
  icon: ReactNode;
};

export type ResourceState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  retry: () => void;
};
