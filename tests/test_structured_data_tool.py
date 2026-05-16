from carnicos_kb.paths import DEFAULT_STRUCTURED_DATA_FILE
from carnicos_kb.structured_data_tool import (
    STRUCTURED_DATA_TOOL_NAME,
    StructuredDataStore,
    build_structured_data_tool,
)


def test_structured_data_store_finds_customer_service_phone() -> None:
    store = StructuredDataStore.from_file(DEFAULT_STRUCTURED_DATA_FILE)

    result = store.search("Cual es el telefono de servicio al cliente?")

    assert "01 8000 519 368" in result
    assert "contacto.md" in result


def test_structured_data_store_returns_single_nit_record() -> None:
    store = StructuredDataStore.from_file(DEFAULT_STRUCTURED_DATA_FILE)

    result = store.search("Cual es el NIT de la empresa?")

    assert "890.304.130 - 4" in result
    assert "890.340.130 - 4" not in result
    assert "inconsistencia" not in result.lower()
    assert "Lineas gratuitas" not in result


def test_langchain_structured_tool_invokes_store() -> None:
    tool = build_structured_data_tool()

    result = tool.invoke({"query": "Listar puntos de venta propios"})

    assert tool.name == STRUCTURED_DATA_TOOL_NAME
    assert "Cali" in result
    assert "Barranquilla" in result
