"""Pruebas de memoria conversacional por thread_id.

Verifica que el agente reciba el thread_id correcto, que conversaciones
distintas no compartan estado y que la respuesta se extraiga bien del
resultado del agente.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from carnicos_kb.qa_system import CarnicosQASystem, QAResponse


SAMPLE_KNOWLEDGE = """\
# Base de conocimiento segmentada

## C0001 | historia.md | Nuestra Historia

**Fuente:** `data/processed/dataset_carnicos/historia.md`

Alimentos Carnicos fue fundada en 1935 en Colombia.
"""


def _make_agent_result(answer: str, tool_content: str = "") -> dict:
    """Construye un resultado de agente simulado."""
    messages = [HumanMessage(content="pregunta")]
    if tool_content:
        messages.append(ToolMessage(content=tool_content, tool_call_id="call_1"))
    messages.append(AIMessage(content=answer))
    return {"messages": messages}


@pytest.fixture()
def qa_system() -> CarnicosQASystem:
    """CarnicosQASystem con init_chat_model, RAG tool y agente completamente mockeados."""
    fake_tool = MagicMock()
    fake_tool.name = "base_documental_carnicos"
    fake_retriever = MagicMock()
    fake_agent = MagicMock()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake-key"}):
        with patch("carnicos_kb.qa_system.init_chat_model"):
            with patch("carnicos_kb.qa_system.load_knowledge_base", return_value=SAMPLE_KNOWLEDGE):
                with patch(
                    "carnicos_kb.qa_system.get_knowledge_stats",
                    return_value={"total_characters": 100, "total_words": 20, "total_paragraphs": 3},
                ):
                    with patch.object(CarnicosQASystem, "_build_documents", return_value=[]):
                        with patch.object(
                            CarnicosQASystem,
                            "_build_rag_tool",
                            return_value=(fake_tool, fake_retriever),
                        ):
                            with patch.object(
                                CarnicosQASystem, "_build_hitl_middleware", return_value=MagicMock()
                            ):
                                with patch("carnicos_kb.qa_system.build_dynamic_rag_prompt", return_value=MagicMock()):
                                    with patch.object(
                                        CarnicosQASystem, "_build_agent", return_value=fake_agent
                                    ):
                                        system = CarnicosQASystem(
                                            model="openai:gpt-test",
                                            temperature=0.0,
                                            max_tokens=500,
                                            verbose=False,
                                        )
    return system


def test_agent_receives_correct_thread_id(qa_system: CarnicosQASystem) -> None:
    """El agente debe recibir el thread_id en la config del checkpointer."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta.")

    qa_system.answer_with_trace("Pregunta de prueba", thread_id="sesion-xyz")

    call_args = qa_system._agent.invoke.call_args
    config = call_args[1]["config"] if "config" in call_args[1] else call_args[0][1]
    assert config["configurable"]["thread_id"] == "sesion-xyz"


def test_different_thread_ids_make_separate_calls(qa_system: CarnicosQASystem) -> None:
    """Dos thread_ids distintos generan invocaciones independientes."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta.")

    qa_system.answer_with_trace("Pregunta A", thread_id="hilo-1")
    qa_system.answer_with_trace("Pregunta B", thread_id="hilo-2")

    assert qa_system._agent.invoke.call_count == 2
    configs = [call[1].get("config") or call[0][1] for call in qa_system._agent.invoke.call_args_list]
    thread_ids = [c["configurable"]["thread_id"] for c in configs]
    assert thread_ids == ["hilo-1", "hilo-2"]


def test_answer_extracts_last_ai_message(qa_system: CarnicosQASystem) -> None:
    """La respuesta final debe ser el ultimo AIMessage del resultado del agente."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta esperada.")

    response = qa_system.answer_with_trace("Pregunta", thread_id="t1")

    assert response.answer == "Respuesta esperada."


def test_tool_output_is_collected_from_tool_messages(qa_system: CarnicosQASystem) -> None:
    """El tool_output de QAResponse debe recopilar el contenido de los ToolMessages."""
    qa_system._agent.invoke.return_value = _make_agent_result(
        "Respuesta.",
        tool_content="## C0050 | contacto.md\nTelefono: 01 8000 527 300.",
    )

    response = qa_system.answer_with_trace("Cual es el telefono?", thread_id="t1")

    assert "01 8000 527 300" in response.tool_output


def test_empty_question_returns_validation_message(qa_system: CarnicosQASystem) -> None:
    """Una pregunta vacia debe devolver un mensaje de validacion sin llamar al agente."""
    response = qa_system.answer_with_trace("   ", thread_id="t1")

    qa_system._agent.invoke.assert_not_called()
    assert "pregunta valida" in response.answer.lower()


def test_agent_exception_returns_error_response(qa_system: CarnicosQASystem) -> None:
    """Una excepcion del agente debe devolver un QAResponse de error, no propagarse."""
    qa_system._agent.invoke.side_effect = RuntimeError("Fallo de red simulado.")

    response = qa_system.answer_with_trace("Pregunta", thread_id="t1")

    assert isinstance(response, QAResponse)
    assert "error" in response.answer.lower()
    assert response.tool_name == "error"
