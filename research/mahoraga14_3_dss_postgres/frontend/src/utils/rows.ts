import type { Row } from "../api/types";

export function pick(row: Row | null | undefined, keys: string[]): unknown {
  if (!row) return null;
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== null) return row[key];
  }
  return null;
}

export function rowsFrom(payload: unknown, key = "rows"): Row[] {
  if (!payload || typeof payload !== "object") return [];
  const value = (payload as Record<string, unknown>)[key];
  return Array.isArray(value) ? (value as Row[]) : [];
}

export function topRows(rows: Row[], key: string, count: number, descending = true): Row[] {
  return [...rows]
    .filter((row) => Number.isFinite(Number(row[key])))
    .sort((a, b) => (Number(a[key]) - Number(b[key])) * (descending ? -1 : 1))
    .slice(0, count);
}
