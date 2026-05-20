from pathlib import Path

from langchain_core.documents import Document

from carnicos_kb.document_retriever_tool import KnowledgeChunk
from carnicos_kb.rag_index_builder import (
    build_pgvector_index,
    format_chunk_for_embedding,
    load_chunks_from_markdown,
)


SAMPLE_CHUNKS_MARKDOWN = """
# Base de conocimiento segmentada

## C0001 | historia.md | Nuestra Historia

**Fuente:** `data/processed/dataset_carnicos/historia.md`

Alimentos Carnicos tiene trayectoria empresarial.

---

## C0002 | bienestar-animal.md | Bienestar Animal

**Fuente:** `data/processed/dataset_carnicos/bienestar-animal.md`

La empresa trabaja por el bienestar animal.
"""


def test_load_chunks_from_markdown_detects_chunk_ids(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.md"
    chunks_file.write_text(SAMPLE_CHUNKS_MARKDOWN, encoding="utf-8")

    chunks = load_chunks_from_markdown(chunks_file)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "C0001"
    assert chunks[1].source.endswith("bienestar-animal.md")


def test_format_chunk_for_embedding_includes_metadata() -> None:
    chunk = KnowledgeChunk(
        chunk_id="C0001",
        title="historia.md | Nuestra Historia",
        source="data/processed/dataset_carnicos/historia.md",
        text="Texto del chunk.",
    )

    prepared_text = format_chunk_for_embedding(chunk)

    assert "Chunk: C0001" in prepared_text
    assert "Titulo: historia.md | Nuestra Historia" in prepared_text
    assert "Fuente: data/processed/dataset_carnicos/historia.md" in prepared_text
    assert "Texto del chunk." in prepared_text


def test_build_pgvector_index_adds_documents_with_expected_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {}

    class FakePGVectorRetriever:
        def __init__(self, database_url, collection_name, embedding_model):
            calls["database_url"] = database_url
            calls["collection_name"] = collection_name
            calls["embedding_model"] = embedding_model

        def index_documents(self, documents):
            calls["documents"] = documents
            return len(documents)

    documents = [
        Document(page_content="Historia", metadata={"source": "historia.md"}),
        Document(page_content="Bienestar", metadata={"source": "bienestar.md"}),
    ]
    monkeypatch.setattr(
        "carnicos_kb.rag_index_builder.require_openai_api_key",
        lambda: None,
    )
    monkeypatch.setattr(
        "carnicos_kb.rag_index_builder.build_chunks_with_splitter",
        lambda input_dir, chunk_size, chunk_overlap: documents,
    )
    monkeypatch.setattr(
        "carnicos_kb.rag_index_builder.PGVectorRetriever",
        FakePGVectorRetriever,
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    count = build_pgvector_index(
        dataset_dir=dataset_dir,
        embedding_model="text-embedding-3-small",
        collection_name="test_collection",
        chunk_size=900,
        chunk_overlap=100,
        database_url="postgresql://user:pass@localhost:5432/testdb",
    )

    assert count == 2
    assert calls["database_url"] == "postgresql://user:pass@localhost:5432/testdb"
    assert calls["collection_name"] == "test_collection"
    assert calls["embedding_model"] == "text-embedding-3-small"
    assert calls["documents"] == documents
