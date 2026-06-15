from __future__ import annotations

from pydantic import BaseModel, Field


class ApiRows(BaseModel):
    count: int
    rows: list[dict]


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    profile: str
    demo_mode: bool
    row_counts: dict[str, int] = Field(default_factory=dict)
    schema_version: str | None = None
