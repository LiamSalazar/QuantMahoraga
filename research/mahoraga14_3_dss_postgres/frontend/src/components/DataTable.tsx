import { useMemo, useState } from "react";
import type { Row } from "../api/types";
import type { ReactNode } from "react";
import { formatText } from "../utils/format";
import { formatCandidateLabel } from "../utils/labels";
import { EmptyState } from "./States";

function hasValue(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false;
  if (typeof value === "number" && !Number.isFinite(value)) return false;
  return true;
}

export function DataTable({
  rows,
  columns,
  pageSize = 12,
  rowAction,
  rowClassName,
}: {
  rows: Row[];
  columns?: string[];
  pageSize?: number;
  rowAction?: (row: Row) => ReactNode;
  rowClassName?: (row: Row) => string;
}) {
  const [page, setPage] = useState(0);
  const cleanRows = useMemo(() => rows.filter((row) => Object.values(row).some(hasValue)), [rows]);
  const visibleColumns = useMemo(() => {
    const requested = columns ?? Object.keys(cleanRows[0] ?? {}).slice(0, 10);
    return requested.filter((column) => cleanRows.some((row) => hasValue(row[column])));
  }, [cleanRows, columns]);
  if (!cleanRows.length || !visibleColumns.length) return <EmptyState title="No rows returned" detail="Reset filters or broaden the research slice." />;
  const pages = Math.max(1, Math.ceil(cleanRows.length / pageSize));
  const safePage = Math.min(page, pages - 1);
  const pageRows = cleanRows.slice(safePage * pageSize, safePage * pageSize + pageSize);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {visibleColumns.map((column) => (
              <th key={column}>{column.replaceAll("_", " ")}</th>
            ))}
            {rowAction ? <th>Action</th> : null}
          </tr>
        </thead>
        <tbody>
          {pageRows.map((row, index) => (
            <tr key={index} className={rowClassName ? rowClassName(row) : undefined}>
              {visibleColumns.map((column) => (
                <td key={column} title={String(row[column] ?? "")}>
                  {column.toLowerCase().includes("candidate") ? (
                    <>
                      <b>{formatCandidateLabel(row[column])}</b>
                      <small>{formatText(row[column])}</small>
                    </>
                  ) : (
                    formatText(row[column])
                  )}
                </td>
              ))}
              {rowAction ? <td className="table-action">{rowAction(row)}</td> : null}
            </tr>
          ))}
        </tbody>
      </table>
      {pages > 1 ? (
        <div className="pager">
          <button onClick={() => setPage(Math.max(0, safePage - 1))} disabled={safePage === 0}>
            Previous
          </button>
          <span>
            {safePage + 1} / {pages}
          </span>
          <button onClick={() => setPage(Math.min(pages - 1, safePage + 1))} disabled={safePage >= pages - 1}>
            Next
          </button>
        </div>
      ) : null}
    </div>
  );
}
