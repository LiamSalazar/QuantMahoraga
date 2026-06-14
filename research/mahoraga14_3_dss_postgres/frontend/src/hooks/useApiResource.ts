import { useEffect, useMemo, useReducer } from "react";
import { apiGet, apiKey } from "../api/client";
import type { ResourceState } from "../api/types";

type State<T> = { data: T | null; loading: boolean; error: string | null; nonce: number };

export function useApiResource<T>(
  path: string,
  params?: Record<string, unknown>,
  enabled = true,
  useCache = true,
): ResourceState<T> {
  const key = useMemo(() => apiKey(path, params), [path, JSON.stringify(params ?? {})]);
  const [state, dispatch] = useReducer((current: State<T>, patch: Partial<State<T>>) => ({ ...current, ...patch }), {
    data: null,
    loading: enabled,
    error: null,
    nonce: 0,
  });

  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    dispatch({ loading: true, error: null });
    apiGet<T>(path, params, controller.signal, useCache)
      .then((data) => dispatch({ data, loading: false, error: null }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        dispatch({ loading: false, error: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, [key, enabled, state.nonce, useCache]);

  return {
    data: state.data,
    loading: state.loading,
    error: state.error,
    retry: () => dispatch({ nonce: state.nonce + 1 }),
  };
}
