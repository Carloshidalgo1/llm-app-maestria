from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from carnicos_kb.qa_system import build_conversation_messages, normalize_chat_history


def test_normalize_chat_history_from_streamlit_messages() -> None:
    history = [
        {"role": "user", "content": "Que marcas hacen parte del portafolio?"},
        {"role": "assistant", "content": "Las marcas documentadas incluyen Rica y Cunit."},
        {"role": "system", "content": "Este mensaje no debe reutilizarse como memoria."},
        {"role": "user", "content": "   "},
    ]

    messages = normalize_chat_history(history)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Que marcas hacen parte del portafolio?"
    assert messages[1].content == "Las marcas documentadas incluyen Rica y Cunit."


def test_build_conversation_messages_preserves_prompt_history_and_question_order() -> None:
    history = [
        HumanMessage(content="Habla de los compromisos de bienestar animal."),
        AIMessage(content="El primer compromiso mencionado fue la meta de cerdas libres de jaulas."),
    ]

    messages = build_conversation_messages(
        system_prompt="Prompt del sistema",
        question="Cual fue el primero que mencionaste?",
        chat_history=history,
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "Cual fue el primero que mencionaste?"
