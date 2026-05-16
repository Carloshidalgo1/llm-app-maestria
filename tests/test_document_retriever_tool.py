from carnicos_kb.document_retriever_tool import (
    DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
    parse_knowledge_chunks,
)


SAMPLE_KNOWLEDGE = """
# Base de conocimiento segmentada

## C0001 | historia.md | Nuestra Historia

**Fuente:** `data/processed/dataset_carnicos/historia.md`

La empresa tiene una historia de desarrollo empresarial.

---

## C0002 | bienestar-animal.md | Compromiso con el bienestar animal

**Fuente:** `data/processed/dataset_carnicos/bienestar-animal.md`

La organizacion trabaja por el bienestar animal y el abastecimiento responsable.
"""


def test_parse_knowledge_chunks_detects_metadata() -> None:
    chunks = parse_knowledge_chunks(SAMPLE_KNOWLEDGE)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "C0001"
    assert chunks[1].source.endswith("bienestar-animal.md")


def test_documental_knowledge_tool_name_is_defined() -> None:
    assert DOCUMENTAL_KNOWLEDGE_TOOL_NAME == "base_documental_carnicos"
