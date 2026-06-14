import type { Row } from "../api/types";

export function hasSeries(rows: Row[] | undefined, xKey: string, yKey: string, minPoints = 2): boolean {
  return (rows ?? []).filter((row) => row[xKey] !== null && row[xKey] !== undefined && row[yKey] !== null && row[yKey] !== undefined).length >= minPoints;
}

export function hasScatter(rows: Row[] | undefined, xKey: string, yKey: string, minPoints = 3): boolean {
  return hasSeries(rows, xKey, yKey, minPoints);
}

export function hasHeatmap(rows: Row[] | undefined, xKey: string, yKey: string, valueKey: string, minCells = 4): boolean {
  const cells = new Set((rows ?? []).filter((row) => row[xKey] !== undefined && row[yKey] !== undefined && row[valueKey] !== undefined).map((row) => `${row[xKey]}::${row[yKey]}`));
  return cells.size >= minCells;
}
