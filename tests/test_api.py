import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

import carnicos_kb.api as api_module
from carnicos_kb.api import app
from carnicos_kb.qa_system import QAResponse


@pytest.fixture
def mock_qa():
    mock = MagicMock()
    mock.answer_with_trace.return_value = QAResponse(answer="El NIT es...", confidence="high")
    mock.resume_with_decision.return_value = QAResponse(
        answer="Respuesta aprobada", confidence="high"
    )
    return mock


@pytest.fixture
async def client(mock_qa):
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, mock_qa


async def test_health_returns_200_when_ready(client):
    ac, _ = client
    response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["agent_ready"] is True
    assert data["status"] == "ok"


async def test_health_returns_degraded_when_agent_not_ready(mock_qa):
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            api_module._qa_system = None
            response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["agent_ready"] is False
    assert data["status"] == "degraded"


async def test_chat_returns_answer(mock_qa):
    mock_qa.answer_with_trace.return_value = QAResponse(answer="El NIT es...", confidence="high")
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/chat",
                json={"message": "¿Cuál es el NIT?", "phone_number": "+573001234567"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""
    assert data["pending_approval"] is False
    assert data["thread_id"] == "+573001234567"


async def test_chat_returns_pending_when_hitl_active(mock_qa):
    mock_qa.answer_with_trace.return_value = QAResponse(
        answer="",
        confidence="unknown",
        pending_approval=True,
        interrupt_payload={
            "description": "Consulta crítica",
            "query": "precio",
            "thread_id": "+573001234567",
        },
    )
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/chat",
                json={"message": "precio", "phone_number": "+573001234567"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["pending_approval"] is True
    assert data["interrupt_payload"] is not None


async def test_resume_approve(mock_qa):
    mock_qa.resume_with_decision.return_value = QAResponse(
        answer="Respuesta aprobada", confidence="high"
    )
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/chat/resume",
                json={"thread_id": "+573001234567", "decision": "approve"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""


async def test_resume_edit_requires_edited_query(mock_qa):
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/chat/resume",
                json={"thread_id": "+573001234567", "decision": "edit", "edited_query": None},
            )
    assert response.status_code == 422


async def test_chat_empty_message_rejected(mock_qa):
    with patch("carnicos_kb.api.CarnicosQASystem") as MockClass:
        MockClass.return_value = mock_qa
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/chat",
                json={"message": "", "phone_number": "+573001234567"},
            )
    assert response.status_code == 422
