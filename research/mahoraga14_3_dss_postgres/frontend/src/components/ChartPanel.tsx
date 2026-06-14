import type { ReactNode } from "react";
import { EmptyState } from "./States";

export function ChartPanel({
  title,
  question,
  source,
  ready,
  emptyDetail,
  children,
}: {
  title: string;
  question: string;
  source: string;
  ready: boolean;
  emptyDetail?: string;
  children: ReactNode;
}) {
  return (
    <section className="panel chart-panel">
      <div className="panel-kicker">{question}</div>
      <h3>{title}</h3>
      <small>Source: {source}</small>
      <div className="chart-box">{ready ? children : <EmptyState detail={emptyDetail ?? "Not enough useful combinations for this chart."} />}</div>
    </section>
  );
}
