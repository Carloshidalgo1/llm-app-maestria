import sys
import types
from pathlib import Path

from carnicos_kb.document_retriever_tool import KnowledgeChunk
from carnicos_kb.rag_index_builder import (
    build_chroma_metadata,
    format_chunk_for_embedding,
    load_chunks_from_markdown,
    save_chroma_index,
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


def test_build_chroma_metadata_keeps_traceable_fields(tmp_path: Path) -> None:
    chunk = KnowledgeChunk(
        chunk_id="C0001",
        title="historia.md | Nuestra Historia",
        source="historia.md",
        text="Texto A",
    )

    metadata = build_chroma_metadata([chunk], tmp_path / "chunks.md")

    assert metadata == [
        {
            "chunk_id": "C0001",
            "title": "historia.md | Nuestra Historia",
            "source": "historia.md",
            "source_path": str(tmp_path / "chunks.md"),
        }
    ]


def test_save_chroma_index_adds_texts_with_ids_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {}

    class FakeChroma:
        def __init__(self, collection_name, embedding_function, persist_directory):
            calls["collection_name"] = collection_name
            calls["embedding_function"] = embedding_function
            calls["persist_directory"] = persist_directory

        def add_texts(self, texts, metadatas, ids):
            calls["texts"] = texts
            calls["metadatas"] = metadatas
            calls["ids"] = ids
            return ids

    fake_chroma_module = types.SimpleNamespace(Chroma=FakeChroma)
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_module)
    monkeypatch.setattr(
        "carnicos_kb.rag_index_builder.create_embedding_client",
        lambda embedding_model: f"embeddings:{embedding_model}",
    )

    chunk = KnowledgeChunk(
        chunk_id="C0001",
        title="historia.md | Nuestra Historia",
        source="historia.md",
        text="Texto A",
    )
    count = save_chroma_index(
        chunks=[chunk],
        texts=["Texto preparado"],
        embedding_model="text-embedding-3-small",
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        source_path=tmp_path / "chunks.md",
    )

    assert count == 1
    assert calls["collection_name"] == "test_collection"
    assert calls["embedding_function"] == "embeddings:text-embedding-3-small"
    assert calls["texts"] == ["Texto preparado"]
    assert calls["ids"] == ["C0001"]
    assert calls["metadatas"][0]["chunk_id"] == "C0001"
