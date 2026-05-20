"""Construccion del indice RAG vectorial con OpenAI embeddings.

Backend por defecto: PGVector (PostgreSQL). Requiere DATABASE_URL en .env.
Fallback: InMemoryVectorStore (solo validacion, sin persistencia).

Comandos:

    carnicos-build-rag                   # indexa en PostgreSQL (DATABASE_URL)
    carnicos-build-rag --dry-run         # valida sin llamar a OpenAI
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from .vector_retriever_tool import PGVectorRetriever
from .chunking import build_chunks_with_splitter
from .document_retriever_tool import KnowledgeChunk, parse_knowledge_chunks
from .paths import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PG_COLLECTION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# 1. Entrada: leer chunks desde Markdown
# ---------------------------------------------------------------------------

def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_chunks_from_markdown(chunks_path: Path) -> list[KnowledgeChunk]:
    """Carga y parsea `base_conocimiento_chunks.md`."""
    absolute_path = resolve_project_path(chunks_path)
    if not absolute_path.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {absolute_path}")

    markdown_content = absolute_path.read_text(encoding="utf-8")
    chunks = parse_knowledge_chunks(markdown_content)
    if not chunks:
        raise ValueError(f"No se detectaron chunks en: {absolute_path}")
    return chunks


def format_chunk_for_embedding(chunk: KnowledgeChunk) -> str:
    """Prepara el texto enviado al modelo de embeddings.

    Incluir titulo y fuente mejora la recuperacion cuando el usuario pregunta
    por una seccion concreta (ej. "bienestar animal").
    """
    return "\n".join(
        [
            f"Chunk: {chunk.chunk_id}",
            f"Titulo: {chunk.title}",
            f"Fuente: {chunk.source}",
            "",
            chunk.text.strip(),
        ]
    )


# ---------------------------------------------------------------------------
# 2. Preparacion de embeddings
# ---------------------------------------------------------------------------

def require_openai_api_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk_test"):
        raise ValueError(
            "OPENAI_API_KEY no esta configurada con una clave real. "
            "Agregala en .env antes de construir el indice RAG."
        )


def build_langchain_index(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> InMemoryVectorStore:
    """Construye un indice vectorial nativo LangChain con RecursiveCharacterTextSplitter.

    Usa InMemoryVectorStore de langchain_core como vector store nativo. Llama a
    build_chunks_with_splitter para dividir los documentos Markdown y los indexa
    con OpenAIEmbeddings.

    Returns:
        InMemoryVectorStore listo para consultas via as_retriever().
    """
    require_openai_api_key()
    absolute_dir = resolve_project_path(dataset_dir)
    if not absolute_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de dataset: {absolute_dir}")

    documents: list[Document] = build_chunks_with_splitter(
        input_dir=absolute_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not documents:
        raise ValueError(f"No se obtuvieron documentos del directorio: {absolute_dir}")

    print(f"Documentos generados con RecursiveCharacterTextSplitter: {len(documents)}")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents)
    print(f"InMemoryVectorStore construido con {len(documents)} chunks.")
    return vector_store


def build_pgvector_index(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_PG_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    database_url: str | None = None,
) -> int:
    """Indexa documentos en PostgreSQL usando PGVector.

    Divide los documentos Markdown con RecursiveCharacterTextSplitter y los
    almacena en PostgreSQL con OpenAIEmbeddings. Los embeddings se calculan
    una sola vez y persisten entre reinicios de la aplicacion.

    Returns:
        Numero de documentos indexados.
    """
    require_openai_api_key()

    db_url = database_url or os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError(
            "DATABASE_URL no esta configurada. Agrega la variable en .env o "
            "pasa --database-url al comando."
        )

    absolute_dir = resolve_project_path(dataset_dir)
    if not absolute_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de dataset: {absolute_dir}")

    print("\nConstruyendo indice PGVector")
    print("-" * 72)
    print(f"Fuente: {absolute_dir}")
    print(f"Modelo embeddings: {embedding_model}")
    print(f"Coleccion PostgreSQL: {collection_name}")
    print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    documents: list[Document] = build_chunks_with_splitter(
        input_dir=absolute_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not documents:
        raise ValueError(f"No se obtuvieron documentos del directorio: {absolute_dir}")

    print(f"Chunks generados: {len(documents)}")
    print("Indexando en PostgreSQL...")

    retriever = PGVectorRetriever(
        database_url=db_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    count = retriever.index_documents(documents)
    print(f"Documentos indexados en PGVector: {count}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye el indice RAG en PostgreSQL (PGVector) usando "
            "RecursiveCharacterTextSplitter y OpenAI embeddings."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Carpeta con los archivos Markdown fuente (default: data/processed/dataset_carnicos).",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Modelo de embeddings de OpenAI.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("PG_COLLECTION_NAME", DEFAULT_PG_COLLECTION),
        help="Nombre de la coleccion PGVector en PostgreSQL.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        help="Tamano maximo de chunk en caracteres (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        help="Solapamiento entre chunks en caracteres (default: 200).",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="URL de conexion PostgreSQL (default: DATABASE_URL del entorno).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida lectura y chunking sin llamar a OpenAI ni escribir en PostgreSQL.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    if args.dry_run:
        absolute_dir = resolve_project_path(args.dataset_dir)
        documents = build_chunks_with_splitter(
            input_dir=absolute_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"\nDry-run: {len(documents)} chunks generados desde {absolute_dir}")
        print("No se llamo a OpenAI ni se escribio en PostgreSQL.")
        return

    build_pgvector_index(
        dataset_dir=args.dataset_dir,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        database_url=args.database_url or None,
    )


if __name__ == "__main__":
    main()
