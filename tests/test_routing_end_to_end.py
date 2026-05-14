"""Pruebas de enrutamiento end-to-end con LLM mockeado.

Estas pruebas verifican que el flujo completo del agente funciona:
router -> herramienta real -> cadena de respuesta, sin llamar a OpenAI.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from carnicos_kb.document_retriever_tool import DOCUMENTAL_KNOWLEDGE_TOOL_NAME
from carnicos_kb.qa_system import CarnicosQASystem, RouteDecision
from carnicos_kb.structured_data_tool import STRUCTURED_DATA_TOOL_NAME


SAMPLE_KNOWLEDGE = """\
# Base de conocimiento segmentada

## C0001 | historia.md | Nuestra Historia

**Fuente:** `data/processed/dataset_carnicos/historia.md`

Alimentos Carnicos fue fundada en 1935 en Colombia. Forma parte del Grupo
Nutresa y produce marcas reconocidas como Zenú, Rica y Cunit.

---

## C0002 | bienestar-animal.md | Compromiso con el bienestar animal

**Fuente:** `data/processed/dataset_carnicos/bienestar-animal.md`

El primer compromiso de la empresa es la meta 2027 sobre cerdas en gestacion
libres de jaulas. También trabaja en abastecimiento responsable.
"""


@pytest.fixture()
def qa_system() -> CarnicosQASystem:
    """Crea un CarnicosQASystem con LLM mockeado y herramientas reales."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake-key-for-unit-tests"}):
        with patch("carnicos_kb.qa_system.ChatOpenAI"):
            system = CarnicosQASystem(
                knowledge_dir=None,
                model="gpt-test",
                temperature=0.0,
                max_tokens=500,
                verbose=False,
            )
    system.knowledge_base = SAMPLE_KNOWLEDGE

    from carnicos_kb.document_retriever_tool import build_documental_knowledge_tool
    from carnicos_kb.structured_data_tool import build_structured_data_tool

    system.documental_tool = build_documental_knowledge_tool(SAMPLE_KNOWLEDGE)
    system.structured_tool = build_structured_data_tool()
    system.tools_by_name = {
        system.documental_tool.name: system.documental_tool,
        system.structured_tool.name: system.structured_tool,
    }
    return system


def _mock_route(qa_system: CarnicosQASystem, tool_name: str, reason: str) -> None:
    """Configura el router para devolver una ruta fija."""
    qa_system.router_chain = MagicMock()
    qa_system.router_chain.invoke.return_value = RouteDecision(
        tool_name=tool_name,
        reason=reason,
    )


def _mock_answer(qa_system: CarnicosQASystem, answer_text: str) -> None:
    """Configura la cadena de respuesta para devolver un texto fijo."""
    qa_system.answer_chain = MagicMock()
    qa_system.answer_chain.invoke.return_value = answer_text


# -----------------------------------------------------------------------
# Prueba de enrutamiento: pregunta estructurada
# -----------------------------------------------------------------------


def test_structured_question_routes_to_structured_tool(
    qa_system: CarnicosQASystem,
) -> None:
    """El agente debe usar datos_estructurados_carnicos para telefonos."""
    _mock_route(
        qa_system,
        tool_name=STRUCTURED_DATA_TOOL_NAME,
        reason="La pregunta solicita un telefono concreto.",
    )
    _mock_answer(qa_system, "El telefono nacional es 01 8000 519 368.")

    response = qa_system.answer_with_trace(
        "Cual es el telefono de servicio al cliente?"
    )

    assert response.tool_name == STRUCTURED_DATA_TOOL_NAME
    assert "01 8000 519 368" in response.tool_output
    assert response.answer == "El telefono nacional es 01 8000 519 368."


# -----------------------------------------------------------------------
# Prueba de enrutamiento: pregunta documental
# -----------------------------------------------------------------------


def test_documental_question_routes_to_documental_tool(
    qa_system: CarnicosQASystem,
) -> None:
    """El agente debe usar base_documental_carnicos para preguntas abiertas."""
    _mock_route(
        qa_system,
        tool_name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        reason="La pregunta requiere contexto documental.",
    )
    _mock_answer(qa_system, "El primer compromiso es la meta 2027 sobre cerdas.")

    response = qa_system.answer_with_trace(
        "Que compromisos existen sobre bienestar animal?"
    )

    assert response.tool_name == DOCUMENTAL_KNOWLEDGE_TOOL_NAME
    assert "bienestar animal" in response.tool_output.lower()
    assert "C0002" in response.tool_output


# -----------------------------------------------------------------------
# Prueba de enrutamiento: conversacion mixta
# -----------------------------------------------------------------------


def test_mixed_conversation_routes_correctly_per_turn(
    qa_system: CarnicosQASystem,
) -> None:
    """En una conversacion mixta, cada turno usa la herramienta correcta."""
    # Turno 1: pregunta documental
    _mock_route(
        qa_system,
        tool_name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        reason="Pregunta abierta sobre historia.",
    )
    _mock_answer(qa_system, "Fue fundada en 1935 en Colombia.")

    response_1 = qa_system.answer_with_trace(
        "Cuando fue fundada la empresa?",
        remember=True,
    )

    assert response_1.tool_name == DOCUMENTAL_KNOWLEDGE_TOOL_NAME

    # Turno 2: pregunta estructurada
    _mock_route(
        qa_system,
        tool_name=STRUCTURED_DATA_TOOL_NAME,
        reason="Dato concreto sobre NIT.",
    )
    _mock_answer(qa_system, "El NIT es 890.304.130 - 4.")

    response_2 = qa_system.answer_with_trace(
        "Cual es el NIT de la empresa?",
        remember=True,
    )

    assert response_2.tool_name == STRUCTURED_DATA_TOOL_NAME
    assert "890.304.130 - 4" in response_2.tool_output

    # Turno 3: vuelve a documental
    _mock_route(
        qa_system,
        tool_name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        reason="Pregunta abierta sobre bienestar.",
    )
    _mock_answer(qa_system, "Meta 2027 sobre cerdas libres de jaulas.")

    response_3 = qa_system.answer_with_trace(
        "Que dice la empresa sobre bienestar animal?",
        remember=True,
    )

    assert response_3.tool_name == DOCUMENTAL_KNOWLEDGE_TOOL_NAME
    assert "bienestar" in response_3.tool_output.lower()


# -----------------------------------------------------------------------
# Prueba de memoria: historial se preserva entre turnos
# -----------------------------------------------------------------------


def test_memory_accumulates_across_turns(qa_system: CarnicosQASystem) -> None:
    """La memoria interna conserva los turnos anteriores."""
    _mock_route(
        qa_system,
        tool_name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        reason="Contexto documental.",
    )
    _mock_answer(qa_system, "Respuesta del turno 1.")

    qa_system.answer_with_trace("Primera pregunta", remember=True)
    assert len(qa_system.get_memory_messages()) == 2  # user + assistant

    _mock_answer(qa_system, "Respuesta del turno 2.")
    qa_system.answer_with_trace("Segunda pregunta", remember=True)
    assert len(qa_system.get_memory_messages()) == 4  # 2 turnos completos


# -----------------------------------------------------------------------
# Prueba de memoria: historial externo se pasa al router
# -----------------------------------------------------------------------


def test_external_chat_history_reaches_router(qa_system: CarnicosQASystem) -> None:
    """El historial de Streamlit llega al router como mensajes normalizados."""
    _mock_route(
        qa_system,
        tool_name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        reason="Pregunta de seguimiento.",
    )
    _mock_answer(qa_system, "El primero fue la meta 2027.")

    external_history = [
        {"role": "user", "content": "Que compromisos de bienestar animal existen?"},
        {"role": "assistant", "content": "Hay varios compromisos documentados."},
    ]

    qa_system.answer_with_trace(
        "Cual fue el primero que mencionaste?",
        chat_history=external_history,
    )

    call_kwargs = qa_system.router_chain.invoke.call_args
    passed_history = call_kwargs[0][0]["chat_history"]
    assert len(passed_history) == 2
    assert isinstance(passed_history[0], HumanMessage)
    assert isinstance(passed_history[1], AIMessage)


# -----------------------------------------------------------------------
# Prueba de enrutamiento: herramienta real produce salida usable
# -----------------------------------------------------------------------


def test_real_structured_tool_output_feeds_answer_chain(
    qa_system: CarnicosQASystem,
) -> None:
    """La salida real de la herramienta estructurada se pasa a la cadena final."""
    _mock_route(
        qa_system,
        tool_name=STRUCTURED_DATA_TOOL_NAME,
        reason="Dato concreto sobre sedes.",
    )
    _mock_answer(qa_system, "Las sedes incluyen Barranquilla, Bogota y Cali.")

    qa_system.answer_with_trace("Listar las sedes comerciales")

    call_kwargs = qa_system.answer_chain.invoke.call_args[0][0]
    assert call_kwargs["tool_name"] == STRUCTURED_DATA_TOOL_NAME
    assert "Barranquilla" in call_kwargs["tool_output"]
    assert "Cali" in call_kwargs["tool_output"]
