import type { ReactNode } from "react";

export function SectionHeader({ title, question, source, action }: { title: string; question?: string; source?: string; action?: ReactNode }) {
  return (
    <header className="section-header">
      <div>
        <h2>{title}</h2>
        {question ? <p>{question}</p> : null}
        {source ? <small>Source: {source}</small> : null}
      </div>
      {action ? <div className="section-action">{action}</div> : null}
    </header>
  );
}
