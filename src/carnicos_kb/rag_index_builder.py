"""Construccion del indice RAG vectorial en Chroma con OpenAI embeddings.

Comando:

    uv run carnicos-build-rag

Para revision sin llamar a OpenAI:

    uv run carnicos-build-rag --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from .document_retriever_tool import KnowledgeChunk, parse_knowledge_chunks
from .paths import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNKS_FILE,
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
# 2. Proceso central: embeddings y almacenamiento en Chroma
# ---------------------------------------------------------------------------

def require_openai_api_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk_test"):
        raise ValueError(
            "OPENAI_API_KEY no esta configurada con una clave real. "
            "Agregala en .env antes de construir el indice RAG."
        )


def create_embedding_client(embedding_model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=embedding_model)


def build_chroma_metadata(
    chunks: list[KnowledgeChunk],
    source_path: Path,
) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source": chunk.source,
            "source_path": str(source_path),
        }
        for chunk in chunks
    ]


def save_chroma_index(
    chunks: list[KnowledgeChunk],
    texts: list[str],
    embedding_model: str,
    persist_directory: Path,
    collection_name: str,
    source_path: Path,
) -> int:
    """Guarda chunks y embeddings en una coleccion local de Chroma."""
    if len(chunks) != len(texts):
        raise ValueError(
            f"La cantidad de chunks no coincide con los textos: "
            f"{len(chunks)} chunks vs {len(texts)} textos."
        )

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "Chroma no esta instalado. Ejecuta `make sync` o instala "
            "`langchain-chroma` antes de usar este comando."
        ) from exc

    absolute_persist_directory = resolve_project_path(persist_directory)
    absolute_persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=create_embedding_client(embedding_model),
        persist_directory=str(absolute_persist_directory),
    )
    document_ids = vector_store.add_texts(
        texts=texts,
        metadatas=build_chroma_metadata(chunks, source_path),
        ids=[chunk.chunk_id for chunk in chunks],
    )
    return len(document_ids)


def print_chroma_summary(
    persist_directory: Path,
    collection_name: str,
    chunk_count: int,
    embedding_model: str,
) -> None:
    print()
    print("Indice Chroma construido")
    print("-" * 72)
    print(f"Directorio: {resolve_project_path(persist_directory)}")
    print(f"Coleccion: {collection_name}")
    print(f"Modelo embeddings: {embedding_model}")
    print(f"Chunks indexados: {chunk_count}")


# ---------------------------------------------------------------------------
# 3. Orquestacion
# ---------------------------------------------------------------------------

def build_rag_index(
    chunks_path: Path,
    embedding_model: str,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION,
    dry_run: bool = False,
) -> int | None:
    """Construye el indice RAG en Chroma o valida los chunks en modo dry-run."""
    absolute_chunks_path = resolve_project_path(chunks_path)
    chunks = load_chunks_from_markdown(absolute_chunks_path)
    texts = [format_chunk_for_embedding(chunk) for chunk in chunks]

    print("Preparacion del indice RAG")
    print("-" * 72)
    print(f"Fuente: {absolute_chunks_path}")
    print(f"Chunks detectados: {len(chunks)}")
    print(f"Modelo de embeddings: {embedding_model}")

    if dry_run:
        first_chunk = chunks[0]
        print()
        print("Modo dry-run: no se llamo a OpenAI ni se escribio el indice.")
        print(f"Primer chunk: {first_chunk.chunk_id} | {first_chunk.title}")
        return None

    require_openai_api_key()
    count = save_chroma_index(
        chunks=chunks,
        texts=texts,
        embedding_model=embedding_model,
        persist_directory=chroma_dir,
        collection_name=chroma_collection,
        source_path=absolute_chunks_path,
    )
    print_chroma_summary(
        persist_directory=chroma_dir,
        collection_name=chroma_collection,
        chunk_count=count,
        embedding_model=embedding_model,
    )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un indice RAG vectorial en Chroma usando "
            "text-embedding-3-small y los chunks Markdown del proyecto."
        )
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path(os.getenv("CARNICOS_KNOWLEDGE_PATH", DEFAULT_CHUNKS_FILE)),
        help="Archivo Markdown segmentado que se va a indexar.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Modelo de embeddings de OpenAI.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path(os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_DIR)),
        help="Directorio local donde Chroma persiste la coleccion.",
    )
    parser.add_argument(
        "--chroma-collection",
        default=os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_CHROMA_COLLECTION),
        help="Nombre de la coleccion Chroma.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida lectura y parseo sin llamar a OpenAI ni escribir indice.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    build_rag_index(
        chunks_path=args.chunks_path,
        embedding_model=args.embedding_model,
        chroma_dir=args.chroma_dir,
        chroma_collection=args.chroma_collection,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
