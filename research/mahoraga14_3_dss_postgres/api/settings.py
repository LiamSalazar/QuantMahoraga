from __future__ import annotations

import os
from dataclasses import dataclass

from etl.config import load_dotenv
from etl.paths import get_paths


@dataclass(frozen=True)
class DssSettings:
    backend: str
    profile: str
    database_url: str | None
    api_host: str
    api_port: int


def load_settings() -> DssSettings:
    paths = get_paths()
    load_dotenv(paths.phase_root / ".env")
    backend = os.getenv("DSS_BACKEND", "parquet").lower()
    if backend == "demo":
        backend = "parquet"
    return DssSettings(
        backend=backend,
        profile=os.getenv("DSS_PROFILE", "small"),
        database_url=os.getenv("DATABASE_URL"),
        api_host=os.getenv("DSS_API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("DSS_API_PORT", "8010")),
    )
