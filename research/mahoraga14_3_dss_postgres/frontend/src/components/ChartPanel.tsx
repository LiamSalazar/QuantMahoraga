import type { ReactNode } from "react";
import { EmptyState } from "./States";

export function ChartPanel({
  title,
  question,
  source,
  ready,
  emptyDetail,
  action,
  children,
}: {
  title: string;
  question: string;
  source: string;
  ready: boolean;
  emptyDetail?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel chart-panel">
      <header className="chart-header">
        <div>
          <div className="panel-kicker">{question}</div>
          <h3>{title}</h3>
          <small>Source: {source}</small>
        </div>
        {action ? <div className="section-action">{action}</div> : null}
      </header>
      <div className="chart-box">{ready ? children : <EmptyState detail={emptyDetail ?? "Not enough useful combinations for this chart."} />}</div>
    </section>
  );
}
