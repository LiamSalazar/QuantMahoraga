from __future__ import annotations

import argparse

from .config import make_config
from .load_postgres import execute_sql_file
from .paths import get_paths


def refresh(database_url: str | None = None) -> None:
    paths = get_paths()
    config = make_config(mode="postgres", database_url=database_url)
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is required to refresh materialized views")
    execute_sql_file(config.database_url, paths.sql_root / "008_refresh_materialized_views.sql")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Mahoraga DSS materialized views.")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    refresh(args.database_url)
    print("refreshed materialized views")


if __name__ == "__main__":
    main()
