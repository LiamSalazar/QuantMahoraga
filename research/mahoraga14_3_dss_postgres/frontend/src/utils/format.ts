export function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function formatNumber(value: unknown, digits = 2): string {
  const n = asNumber(value);
  if (n === null) return "—";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 10_000) return `${(n / 1_000).toFixed(1)}k`;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(digits);
}

export function formatPercent(value: unknown, digits = 1): string {
  const n = asNumber(value);
  if (n === null) return "—";
  const scaled = Math.abs(n) <= 1 ? n * 100 : n;
  return `${scaled.toFixed(digits)}%`;
}

export function formatDrawdown(value: unknown): string {
  return formatPercent(value, 1);
}

export function formatAlpha(value: unknown): string {
  return formatPercent(value, 1);
}

export function formatMetric(value: unknown, key = ""): string {
  const lower = key.toLowerCase();
  if (lower.includes("cagr") || lower.includes("alpha") || lower.includes("maxdd") || lower.includes("exposure") || lower.includes("turnover") || lower.includes("rate") || lower.includes("return")) {
    return formatPercent(value, lower.includes("turnover") ? 2 : 1);
  }
  return formatNumber(value, lower.includes("count") || lower.includes("rows") || lower.includes("observations") ? 0 : 3);
}

export function formatDate(value: unknown): string {
  return typeof value === "string" ? value.slice(0, 10) : "—";
}

export function formatText(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}
