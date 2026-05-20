"""Recuperador documental y middleware RAG usando LangChain nativo.

Soporta dos backends de vector store:
- PGVector (PostgreSQL): persistente, recomendado para produccion.
- InMemoryVectorStore: efimero, usado como fallback en desarrollo.

Ambos exponen la misma interfaz de busqueda y son compatibles con
el middleware dynamic_prompt.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from .document_retriever_tool import DOCUMENTAL_KNOWLEDGE_TOOL_NAME
from .paths import DEFAULT_PG_COLLECTION

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_RAG_CONTEXT_CHUNKS = 3
_RAG_TOOL_DESCRIPTION = (
    "Busca en la base de conocimiento de Alimentos Carnicos S.A.S. "
    "Usar para cualquier pregunta sobre la empresa: telefonos, lineas "
    "de atencion, sedes, puntos de venta, centros de distribucion, NIT, "
    "fecha de creacion, empleo, visitas a planta, horarios, marcas, "
    "productos, historia, sostenibilidad, bienestar animal, procesos "
    "o gobierno corporativo."
)


# ---------------------------------------------------------------------------
# Protocolo compartido
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentRetriever(Protocol):
    """Interfaz comun para todos los retrievers del sistema."""

    def search(self, query: str) -> str: ...


def render_langchain_document(doc: Document) -> str:
    """Presenta un Document de LangChain con fuente trazable."""
    content = doc.page_content.strip()
    metadata = doc.metadata or {}
    source = str(metadata.get("source", "Fuente no registrada"))
    title = str(metadata.get("title", "Sin titulo"))
    return "\n".join([f"## {title}", f"Fuente: {source}", "", content])


def _render_docs(docs: list[Document]) -> str:
    if not docs:
        return (
            "No encontre fragmentos documentales claramente relacionados. "
            "Responde que no hay informacion verificada suficiente."
        )
    return "\n\n".join(render_langchain_document(doc) for doc in docs)


def _pgvector_connection(database_url: str) -> str:
    """Convierte DATABASE_URL a formato SQLAlchemy requerido por PGVector."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


# ---------------------------------------------------------------------------
# Backend 1: PGVector (PostgreSQL — persistente, produccion)
# ---------------------------------------------------------------------------

class PGVectorRetriever:
    """Recupera chunks desde PostgreSQL usando PGVector. Persistente entre reinicios."""

    def __init__(
        self,
        database_url: str,
        collection_name: str = DEFAULT_PG_COLLECTION,
        embedding_model: str | None = None,
        max_chunks: int = _RAG_CONTEXT_CHUNKS,
    ):
        self.max_chunks = max_chunks
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=_pgvector_connection(database_url),
            use_jsonb=True,
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.max_chunks},
        )

    def index_documents(self, documents: list[Document]) -> int:
        """Indexa documentos en PostgreSQL. Llama solo desde carnicos-build-rag."""
        ids = self.vector_store.add_documents(documents)
        return len(ids)

    def search(self, query: str) -> str:
        """Devuelve los chunks mas cercanos semanticamente a la pregunta."""
        query = query.strip()
        if not query:
            return "No se recibio una pregunta documental valida."
        docs = self.retriever.invoke(query)
        return _render_docs(docs)


def build_pgvector_documental_knowledge_tool(
    database_url: str,
    collection_name: str = DEFAULT_PG_COLLECTION,
    embedding_model: str | None = None,
    max_chunks: int = _RAG_CONTEXT_CHUNKS,
) -> tuple[StructuredTool, PGVectorRetriever]:
    """Crea la herramienta documental respaldada por PGVector (PostgreSQL).

    Carga el indice existente desde PostgreSQL. Requiere que carnicos-build-rag
    haya sido ejecutado al menos una vez para poblar la coleccion.

    Returns:
        Tupla (tool, retriever) donde retriever puede usarse para dynamic_prompt.
    """
    retriever = PGVectorRetriever(
        database_url=database_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        max_chunks=max_chunks,
    )

    def consultar_base_documental(query: str) -> str:
        """Devuelve fragmentos documentales relevantes para la pregunta."""
        return retriever.search(query)

    tool = StructuredTool.from_function(
        name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        func=consultar_base_documental,
        description=_RAG_TOOL_DESCRIPTION,
    )
    return tool, retriever


# ---------------------------------------------------------------------------
# Backend 2: InMemoryVectorStore (fallback desarrollo)
# ---------------------------------------------------------------------------

class LangChainRetriever:
    """Recupera chunks por similitud semantica usando InMemoryVectorStore nativo."""

    def __init__(
        self,
        documents: list[Document],
        embedding_model: str | None = None,
        max_chunks: int = _RAG_CONTEXT_CHUNKS,
    ):
        self.max_chunks = max_chunks
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = InMemoryVectorStore(embedding=embeddings)
        self.vector_store.add_documents(documents)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.max_chunks}
        )

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "No se recibio una pregunta documental valida."
        docs = self.retriever.invoke(query)
        return _render_docs(docs)


def build_langchain_documental_knowledge_tool(
    documents: list[Document],
    embedding_model: str | None = None,
    max_chunks: int = _RAG_CONTEXT_CHUNKS,
) -> tuple[StructuredTool, LangChainRetriever]:
    """Crea herramienta documental con InMemoryVectorStore (sin persistencia).

    Usar solo en desarrollo. En produccion preferir build_pgvector_documental_knowledge_tool.
    """
    retriever = LangChainRetriever(
        documents=documents,
        embedding_model=embedding_model,
        max_chunks=max_chunks,
    )

    def consultar_base_documental(query: str) -> str:
        """Devuelve fragmentos documentales relevantes para la pregunta."""
        return retriever.search(query)

    tool = StructuredTool.from_function(
        name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
        func=consultar_base_documental,
        description=_RAG_TOOL_DESCRIPTION,
    )
    return tool, retriever


# ---------------------------------------------------------------------------
# Middleware dynamic_prompt (compatible con ambos backends)
# ---------------------------------------------------------------------------

def build_dynamic_rag_prompt(retriever: DocumentRetriever, base_system_prompt: str):
    """Crea middleware dynamic_prompt que inyecta contexto RAG en cada llamada al modelo.

    Compatible con PGVectorRetriever y LangChainRetriever gracias al protocolo
    DocumentRetriever. Antes de cada llamada al LLM recupera los chunks mas
    relevantes para el ultimo mensaje del usuario y los inyecta en el system prompt.
    """

    @dynamic_prompt
    def rag_context_prompt(request: ModelRequest) -> str:
        last_human = next(
            (m for m in reversed(request.messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human:
            context = retriever.search(last_human.content)
            return (
                f"{base_system_prompt}\n\n"
                f"## Contexto recuperado de la base documental\n\n{context}"
            )
        return base_system_prompt

    return rag_context_prompt
