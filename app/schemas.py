from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    debug: bool | None = None


class AgentStepOut(BaseModel):
    step: int
    kind: str
    payload: dict[str, Any]


class AskResponse(BaseModel):
    success: bool
    result: str
    error: str | None = None
    steps: list[AgentStepOut] | None = None
