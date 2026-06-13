from __future__ import annotations

import polars as pl


def parse_date(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
