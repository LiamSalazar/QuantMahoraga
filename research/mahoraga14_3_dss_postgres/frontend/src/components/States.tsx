import { AlertTriangle, Loader2, RefreshCcw } from "lucide-react";

export function LoadingState({ label = "Loading research slice" }: { label?: string }) {
  return (
    <div className="state-card loading">
      <Loader2 size={18} />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ title = "No useful chart density", detail = "Showing the table or KPI evidence instead." }: { title?: string; detail?: string }) {
  return (
    <div className="state-card">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function ErrorState({ error, retry }: { error: string; retry: () => void }) {
  return (
    <div className="state-card error">
      <AlertTriangle size={18} />
      <strong>View failed safely</strong>
      <span>{error}</span>
      <button className="ghost-button" onClick={retry}>
        <RefreshCcw size={14} /> Retry
      </button>
    </div>
  );
}
