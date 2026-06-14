export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8010";

const cache = new Map<string, unknown>();

export function queryString(params?: Record<string, unknown>): string {
  if (!params) return "";
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "" || value === "all") return;
    if (Array.isArray(value)) {
      value.forEach((item) => search.append(key, String(item)));
    } else {
      search.set(key, String(value));
    }
  });
  const out = search.toString();
  return out ? `?${out}` : "";
}

export function apiKey(path: string, params?: Record<string, unknown>): string {
  return `${path}${queryString(params)}`;
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, unknown>,
  signal?: AbortSignal,
  useCache = true,
): Promise<T> {
  const key = apiKey(path, params);
  if (useCache && cache.has(key)) return cache.get(key) as T;
  const response = await fetch(`${API_BASE}${key}`, { signal });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${detail ? ` · ${detail.slice(0, 180)}` : ""}`);
  }
  const payload = (await response.json()) as T;
  if (useCache) cache.set(key, payload);
  return payload;
}

export function clearApiCache(prefix?: string): void {
  if (!prefix) {
    cache.clear();
    return;
  }
  [...cache.keys()].forEach((key) => {
    if (key.startsWith(prefix)) cache.delete(key);
  });
}
