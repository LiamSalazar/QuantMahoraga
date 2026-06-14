import { formatAlpha, formatDrawdown, formatMetric, formatNumber, formatPercent } from "./format";

export const OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05";
export const OFFICIAL_LABEL = "Official Baseline - Mahoraga 14.3R ROBUST_MAIN";

export function isOfficialCandidate(candidateId?: unknown): boolean {
  return candidateId === OFFICIAL_CANDIDATE_ID;
}

function parseCandidate(candidateId: string): Record<string, string> | null {
  const match = candidateId.match(/^B(?<budget>[\d.]+)_C(?<conviction>[\d.]+)_L(?<leader>[\d.]+)_R(?<backoff>[\d.]+)$/);
  return match?.groups ?? null;
}

export function formatCandidateLabel(candidateId?: unknown): string {
  if (typeof candidateId !== "string" || !candidateId) return "No candidate";
  if (isOfficialCandidate(candidateId)) return OFFICIAL_LABEL;
  if (candidateId === "EXTREME_pro-risk") return "Extreme: pro-risk";
  if (candidateId === "EXTREME_pro-defense") return "Extreme: pro-defense";
  if (candidateId === "EXTREME_all-high") return "Extreme: all-high stress";
  if (candidateId === "EXTREME_all-low") return "Extreme: all-low stress";
  const parsed = parseCandidate(candidateId);
  if (!parsed) return candidateId;
  return `Budget ${parsed.budget} / Conviction ${parsed.conviction} / Leader ${parsed.leader} / Backoff ${parsed.backoff}`;
}

export function formatUniverseLabel(universeId?: unknown): string {
  if (universeId === "base_universe_12") return "base_universe_12 · official 12-name technology universe";
  return typeof universeId === "string" ? universeId.replaceAll("_", " ") : "No universe";
}

export function formatDemoMode(value?: unknown): string {
  return value ? "Simulated what-if · not official performance" : "Observed/audited scenario";
}

export { formatAlpha, formatDrawdown, formatMetric, formatNumber, formatPercent };
