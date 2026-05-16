from langchain_core.documents import Document

from carnicos_kb.chroma_retriever_tool import render_chroma_document


def test_render_chroma_document_includes_source_metadata() -> None:
    document = Document(
        page_content="Contenido recuperado.",
        metadata={
            "chunk_id": "C0001",
            "title": "historia.md | Nuestra Historia",
            "source": "historia.md",
        },
    )

    rendered = render_chroma_document(document)

    assert "C0001" in rendered
    assert "Nuestra Historia" in rendered
    assert "historia.md" in rendered
    assert "Contenido recuperado." in rendered


def test_render_chroma_document_does_not_duplicate_prepared_header() -> None:
    prepared_content = "\n".join(
        [
            "Chunk: C0001",
            "Titulo: historia.md | Nuestra Historia",
            "Fuente: historia.md",
            "",
            "Contenido recuperado.",
        ]
    )
    document = Document(
        page_content=prepared_content,
        metadata={
            "chunk_id": "C0001",
            "title": "historia.md | Nuestra Historia",
            "source": "historia.md",
        },
    )

    rendered = render_chroma_document(document)

    assert rendered == prepared_content
