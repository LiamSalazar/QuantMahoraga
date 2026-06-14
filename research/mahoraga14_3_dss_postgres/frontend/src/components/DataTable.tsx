import { useMemo, useState } from "react";
import type { Row } from "../api/types";
import { formatText } from "../utils/format";
import { formatCandidateLabel } from "../utils/labels";
import { EmptyState } from "./States";

export function DataTable({ rows, columns, pageSize = 12 }: { rows: Row[]; columns?: string[]; pageSize?: number }) {
  const [page, setPage] = useState(0);
  const visibleColumns = useMemo(() => columns ?? Object.keys(rows[0] ?? {}).slice(0, 10), [rows, columns]);
  if (!rows.length) return <EmptyState title="No rows returned" detail="Reset filters or broaden the research slice." />;
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const safePage = Math.min(page, pages - 1);
  const pageRows = rows.slice(safePage * pageSize, safePage * pageSize + pageSize);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {visibleColumns.map((column) => (
              <th key={column}>{column.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pageRows.map((row, index) => (
            <tr key={index}>
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
