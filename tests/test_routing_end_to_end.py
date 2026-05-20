"""Pruebas de flujo end-to-end del agente ReAct con LLM mockeado.

Verifica que el agente unico RAG recibe las preguntas correctamente, que
la trazabilidad funciona, y que QAResponse refleja el resultado real del
agente LangChain.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from carnicos_kb.document_retriever_tool import DOCUMENTAL_KNOWLEDGE_TOOL_NAME
from carnicos_kb.qa_system import CarnicosQASystem, QAResponse


SAMPLE_KNOWLEDGE = """\
# Base de conocimiento segmentada

## C0001 | historia.md | Nuestra Historia

**Fuente:** `data/processed/dataset_carnicos/historia.md`

Alimentos Carnicos fue fundada en 1935 en Colombia. Forma parte del Grupo
Nutresa y produce marcas reconocidas como Zenu, Rica y Cunit.

---

## C0002 | bienestar-animal.md | Compromiso con el bienestar animal

**Fuente:** `data/processed/dataset_carnicos/bienestar-animal.md`

El primer compromiso de la empresa es la meta 2027 sobre cerdas en gestacion
libres de jaulas. Tambien trabaja en abastecimiento responsable.
"""


def _make_agent_result(answer: str, tool_content: str = "") -> dict:
    messages: list = [HumanMessage(content="pregunta")]
    if tool_content:
        messages.append(ToolMessage(content=tool_content, tool_call_id="call_1"))
    messages.append(AIMessage(content=answer))
    return {"messages": messages}


@pytest.fixture()
def qa_system() -> CarnicosQASystem:
    """CarnicosQASystem con init_chat_model, herramienta RAG y agente completamente mockeados."""
    fake_tool = MagicMock()
    fake_tool.name = DOCUMENTAL_KNOWLEDGE_TOOL_NAME
    fake_retriever = MagicMock()
    fake_agent = MagicMock()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake-key-for-unit-tests"}):
        with patch("carnicos_kb.qa_system.init_chat_model"):
            with patch("carnicos_kb.qa_system.load_knowledge_base", return_value=SAMPLE_KNOWLEDGE):
                with patch(
                    "carnicos_kb.qa_system.get_knowledge_stats",
                    return_value={"total_characters": 200, "total_words": 40, "total_paragraphs": 4},
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


# -----------------------------------------------------------------------
# Flujo basico
# -----------------------------------------------------------------------


def test_agent_is_invoked_with_user_question(qa_system: CarnicosQASystem) -> None:
    """El agente debe recibir la pregunta del usuario como HumanMessage."""
    qa_system._agent.invoke.return_value = _make_agent_result("Historia documentada.")

    qa_system.answer_with_trace("Cuando fue fundada la empresa?", thread_id="t1")

    call_args = qa_system._agent.invoke.call_args
    input_messages = call_args[0][0]["messages"]
    assert len(input_messages) == 1
    assert isinstance(input_messages[0], HumanMessage)
    assert "fundada" in input_messages[0].content


def test_response_answer_comes_from_last_ai_message(qa_system: CarnicosQASystem) -> None:
    """El answer de QAResponse debe ser el contenido del ultimo AIMessage."""
    qa_system._agent.invoke.return_value = _make_agent_result(
        "Fue fundada en 1935.", "## C0001\nContenido."
    )

    response = qa_system.answer_with_trace("Cuando fue fundada?", thread_id="t1")

    assert response.answer == "Fue fundada en 1935."


def test_tool_output_contains_retrieved_fragments(qa_system: CarnicosQASystem) -> None:
    """Los fragmentos recuperados deben aparecer en tool_output de QAResponse."""
    tool_content = "## C0002 | bienestar-animal.md\nMeta 2027 sobre cerdas libres de jaulas."
    qa_system._agent.invoke.return_value = _make_agent_result(
        "El compromiso principal es la meta 2027.", tool_content
    )

    response = qa_system.answer_with_trace(
        "Que compromisos existen sobre bienestar animal?", thread_id="t1"
    )

    assert "C0002" in response.tool_output
    assert "meta 2027" in response.tool_output.lower()


def test_response_tool_name_is_rag_tool(qa_system: CarnicosQASystem) -> None:
    """El tool_name de QAResponse debe reflejar la herramienta RAG configurada."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta.")

    response = qa_system.answer_with_trace("Pregunta", thread_id="t1")

    assert response.tool_name == DOCUMENTAL_KNOWLEDGE_TOOL_NAME


# -----------------------------------------------------------------------
# Flujo de contactos/datos concretos (antes ruta estructurada)
# -----------------------------------------------------------------------


def test_phone_query_goes_through_rag_tool(qa_system: CarnicosQASystem) -> None:
    """Las preguntas sobre telefonos usan la herramienta RAG unica."""
    tool_content = (
        "## C0300 | contacto.md | Contacto\n"
        "Telefono Rica: 01 8000 527 300\n"
        "Telefono Cunit: 01 8000 526 782"
    )
    qa_system._agent.invoke.return_value = _make_agent_result(
        "El telefono de Rica es 01 8000 527 300.", tool_content
    )

    response = qa_system.answer_with_trace(
        "Cual es el telefono de servicio al cliente?", thread_id="t1"
    )

    assert "01 8000 527 300" in response.tool_output
    assert "01 8000 527 300" in response.answer


# -----------------------------------------------------------------------
# Thread_id y aislamiento de sesiones
# -----------------------------------------------------------------------


def test_thread_id_is_passed_in_checkpointer_config(qa_system: CarnicosQASystem) -> None:
    """El thread_id debe incluirse en el config del checkpointer."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta.")

    qa_system.answer_with_trace("Pregunta", thread_id="sesion-abc")

    call_args = qa_system._agent.invoke.call_args
    config = call_args[1].get("config") or call_args[0][1]
    assert config["configurable"]["thread_id"] == "sesion-abc"


def test_multiple_turns_use_same_thread_id(qa_system: CarnicosQASystem) -> None:
    """Turnos consecutivos de la misma sesion comparten thread_id."""
    qa_system._agent.invoke.return_value = _make_agent_result("Respuesta.")

    qa_system.answer_with_trace("Primera pregunta", thread_id="hilo-1")
    qa_system.answer_with_trace("Segunda pregunta", thread_id="hilo-1")

    configs = [
        (c[1].get("config") or c[0][1])
        for c in qa_system._agent.invoke.call_args_list
    ]
    assert all(c["configurable"]["thread_id"] == "hilo-1" for c in configs)


# -----------------------------------------------------------------------
# Manejo de errores
# -----------------------------------------------------------------------


def test_empty_question_returns_validation_response(qa_system: CarnicosQASystem) -> None:
    response = qa_system.answer_with_trace("   ", thread_id="t1")

    qa_system._agent.invoke.assert_not_called()
    assert "pregunta valida" in response.answer.lower()


def test_agent_exception_does_not_propagate(qa_system: CarnicosQASystem) -> None:
    qa_system._agent.invoke.side_effect = Exception("Error simulado.")

    response = qa_system.answer_with_trace("Pregunta", thread_id="t1")

    assert isinstance(response, QAResponse)
    assert response.tool_name == "error"
    assert "error" in response.answer.lower()
