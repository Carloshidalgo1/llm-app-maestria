from langchain_core.documents import Document

from carnicos_kb.vector_retriever_tool import render_langchain_document


def test_render_langchain_document_includes_title_and_source() -> None:
    document = Document(
        page_content="Contenido recuperado.",
        metadata={
            "title": "historia",
            "source": "data/processed/dataset_carnicos/historia.md",
        },
    )

    rendered = render_langchain_document(document)

    assert "historia" in rendered
    assert "data/processed/dataset_carnicos/historia.md" in rendered
    assert "Contenido recuperado." in rendered


def test_render_langchain_document_uses_fallback_when_no_metadata() -> None:
    document = Document(page_content="Texto sin metadatos.", metadata={})

    rendered = render_langchain_document(document)

    assert "Sin titulo" in rendered
    assert "Fuente no registrada" in rendered
    assert "Texto sin metadatos." in rendered


def test_render_langchain_document_structure() -> None:
    document = Document(
        page_content="Detalle del producto.",
        metadata={"title": "productos", "source": "productos.md"},
    )

    rendered = render_langchain_document(document)
    lines = rendered.splitlines()

    assert lines[0] == "## productos"
    assert lines[1].startswith("Fuente:")
    assert "Detalle del producto." in rendered
