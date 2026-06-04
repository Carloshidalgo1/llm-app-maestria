from contextlib import asynccontextmanager
from typing import Literal
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .qa_system import CarnicosQASystem, QAResponse

_qa_system: CarnicosQASystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qa_system
    load_dotenv()
    _qa_system = CarnicosQASystem(verbose=False)
    yield
    _qa_system = None


app = FastAPI(title="Carnicos KB API", lifespan=lifespan)


def get_qa_system() -> CarnicosQASystem:
    if _qa_system is None:
        raise HTTPException(status_code=503, detail="Agente no inicializado.")
    return _qa_system


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    phone_number: str = Field(
        ...,
        description="Número de teléfono del usuario. Se usa como thread_id para memoria de sesión.",
    )


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    tool_was_called: bool
    pending_approval: bool
    thread_id: str
    interrupt_payload: dict | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "edit", "reject"]
    message: str = ""
    edited_query: str | None = None


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


def _map_response(qa_resp: QAResponse, thread_id: str) -> ChatResponse:
    return ChatResponse(
        answer=qa_resp.answer,
        confidence=qa_resp.confidence,
        tool_was_called=qa_resp.tool_output != "",
        pending_approval=qa_resp.pending_approval,
        thread_id=thread_id,
        interrupt_payload=qa_resp.interrupt_payload,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    qa = get_qa_system()
    try:
        qa_resp = qa.answer_with_trace(req.message, thread_id=req.phone_number)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _map_response(qa_resp, req.phone_number)


@app.post("/chat/resume", response_model=ChatResponse)
def resume(req: ResumeRequest) -> ChatResponse:
    if req.decision == "edit" and req.edited_query is None:
        raise HTTPException(
            status_code=422,
            detail="edited_query es obligatorio cuando decision='edit'.",
        )
    qa = get_qa_system()
    try:
        qa_resp = qa.resume_with_decision(
            thread_id=req.thread_id,
            decision_type=req.decision,
            message=req.message,
            edited_query=req.edited_query,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _map_response(qa_resp, req.thread_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _qa_system is not None:
        return HealthResponse(status="ok", agent_ready=True)
    return HealthResponse(status="degraded", agent_ready=False)


def run() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("carnicos_kb.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
