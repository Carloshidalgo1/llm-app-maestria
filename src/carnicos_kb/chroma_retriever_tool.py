"""Herramienta LangChain para recuperar contexto documental desde Chroma."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings

from .document_retriever_tool import DOCUMENTAL_KNOWLEDGE_TOOL_NAME
from .paths import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def resolve_project_path(path: Path) -> Path:
    """Convierte rutas relativas del proyecto en rutas absolutas."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_chroma_class():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "Chroma no esta instalado. Ejecuta `make sync` o instala "
            "`langchain-chroma` antes de activar el recuperador Chroma."
        ) from exc
    return Chroma


def render_chroma_document(document: Document) -> str:
    """Presenta un documento recuperado desde Chroma con fuente trazable."""
    content = document.page_content.strip()
    if content.startswith("Chunk:"):
        return content

    metadata = document.metadata or {}
    chunk_id = str(metadata.get("chunk_id", "chunk_sin_id"))
    title = str(metadata.get("title", "Sin titulo"))
    source = str(metadata.get("source", "Fuente no registrada"))

    return "\n".join(
        [
            f"## {chunk_id} | {title}",
            f"Fuente: {source}",
            "",
            content,
        ]
    )


class ChromaDocumentRetriever:
    """Recupera chunks por similitud semantica desde una coleccion Chroma."""

    def __init__(
        self,
        persist_directory: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        embedding_model: Optional[str] = None,
        max_chunks: int = 3,
    ):
        self.persist_directory = resolve_project_path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )
        self.max_chunks = max_chunks

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                "No existe el indice Chroma en "
                f"{self.persist_directory}. Ejecuta `make rag-index-chroma`."
            )

        Chroma = _load_chroma_class()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=OpenAIEmbeddings(model=self.embedding_model),
            persist_directory=str(self.persist_directory),
        )

    def search(self, query: str, max_chunks: Optional[int] = None) -> str:
        """Devuelve los chunks mas cercanos semanticamente a la pregunta."""
        query = query.strip()
        if not query:
            return "No se recibio una pregunta documental valida."

        retrieved_docs = self.vector_store.similarity_search(
            query,
            k=max_chunks or self.max_chunks,
        )
        if not retrieved_docs:
            return (
                "No encontre fragmentos documentales claramente relacionados. "
                "Responde que no hay informacion verificada suficiente."
            )

        return "\n\n".join(render_chroma_document(doc) for doc in retrieved_docs)


def build_chroma_documental_knowledge_tool(
    persist_directory: Optional[Path] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> StructuredTool:
    """Crea la herramienta documental respaldada por busqueda vectorial Chroma."""
    retriever = ChromaDocumentRetriever(
        persist_directory=persist_directory or DEFAULT_CHROMA_DIR,
        collection_name=collection_name or DEFAULT_CHROMA_COLLECTION,
        embedding_model=embedding_model,
    )

    def consultar_base_documental(query: str) -> str:
        """Devuelve fragmentos documentales relevantes para la pregunta."""
        return retriever.search(query)

    return StructuredTool.from_function(
        name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        func=consultar_base_documental,
        description=(
            "Usar para preguntas abiertas sobre historia, productos, marcas, "
            "sostenibilidad, procesos, bienestar animal, gobierno corporativo "
            "o explicaciones que requieren busqueda semantica documental."
        ),
    )
