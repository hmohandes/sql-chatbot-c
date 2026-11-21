from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agent import get_agent
from app.config import get_settings
from app.schemas import AgentStepOut, AskRequest, AskResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logger.info("Starting SQL chatbot (model=%s)", settings.model_id)
    yield


app = FastAPI(
    title="SQL Chatbot Agent",
    description="Ask natural-language questions; the Gemma 4 agent queries PostgreSQL.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    settings = get_settings()
    debug = settings.debug if payload.debug is None else payload.debug
    try:
        outcome = get_agent().run(payload.question, debug=debug)
    except Exception as exc:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    steps = None
    if debug:
        steps = [
            AgentStepOut(step=item.step, kind=item.kind, payload=item.payload)
            for item in outcome.steps
        ]

    return AskResponse(
        success=outcome.success,
        result=outcome.result,
        error=None if outcome.success else outcome.error,
        steps=steps,
    )
