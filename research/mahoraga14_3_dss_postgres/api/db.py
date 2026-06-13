from __future__ import annotations

from .demo_backend import ParquetBackend
from .postgres_backend import PostgresBackend
from .settings import DssSettings


def create_backend(settings: DssSettings):
    if settings.backend == "postgres":
        return PostgresBackend(settings.database_url or "")
    return ParquetBackend()
