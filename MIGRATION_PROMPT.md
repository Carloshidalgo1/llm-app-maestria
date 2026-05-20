# Prompt de migración: carnicos-kb → LangChain nativo (Módulo 3)

## Contexto del proyecto

Este es el proyecto **carnicos-kb**, un sistema Q&A documental sobre Alimentos
Carnicos S.A.S. construido con Python 3.10+, administrado con `uv` y empaquetado
como `src/carnicos_kb/`.

### Stack actual (a reemplazar)

| Componente | Implementación actual | Reemplazar con |
|---|---|---|
| LLM | `ChatOpenAI(model=..., temperature=..., api_key=...)` en `qa_system.py` | `init_chat_model` |
| Agente | `create_react_agent` de `langgraph.prebuilt` en `qa_system.py` | `create_agent` de `langchain.agents` |
| Control humano | No existe | `HumanInTheLoopMiddleware` |
| Chunking | Lógica manual en `chunking.py` (`split_section`, `split_long_block`, `split_by_headings`) | `RecursiveCharacterTextSplitter` |
| Vector store | PGVector en `vector_retriever_tool.py` y `rag_index_builder.py` | `langchain_core.vectorstores` + embeddings nativos de LangChain |
| RAG prompt | Sin chain explícita; `consultar_base_documental()` devuelve texto plano directamente | `dynamic_prompt` (`ChatPromptTemplate` dinámica) |
| Checkpointer | `InMemorySaver` en desarrollo y `PostgresSaver` en produccion | `PostgresSaver` |

### Archivos principales afectados

```
src/carnicos_kb/
├── qa_system.py             ← LLM, agente, checkpointer, middleware
├── vector_retriever_tool.py ← vector store + retriever + RAG prompt
├── rag_index_builder.py     ← construcción del índice vectorial
├── chunking.py              ← chunker (reemplazar lógica principal)
└── paths.py                 ← agregar constante POSTGRES_CONNECTION_STRING
```

---

## Objetivo

Migrar el proyecto para que use **obligatoriamente** los siete componentes
listados a continuación. No deben quedar rastros activos de las implementaciones
anteriores para los componentes migrados. Los archivos que no se listan arriba
(`scraper.py`, `pdf_extractor.py`, `streamlit_app.py`, etc.) no deben modificarse
salvo que sea necesario para que el sistema funcione correctamente tras la migración.

---

## 1. `init_chat_model` — Inicialización del LLM

**Import:** `from langchain.chat_models import init_chat_model`

**Dónde:** `src/carnicos_kb/qa_system.py`, método `__init__` de `CarnicosQASystem`.

**Qué cambiar:**

Eliminar la instanciación directa de `ChatOpenAI`:

```python
# ELIMINAR
from langchain_openai import ChatOpenAI
self.llm = ChatOpenAI(
    model=self.model,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    api_key=api_key,
)
```

Reemplazar con:

```python
from langchain.chat_models import init_chat_model

self.llm = init_chat_model(
    model=f"openai:{self.model}",   # formato "provider:model"
    temperature=self.temperature,
    max_tokens=self.max_tokens,
)
```

`init_chat_model` lee `OPENAI_API_KEY` automáticamente del entorno; no se pasa
`api_key` explícitamente. Mantener la validación de `OPENAI_API_KEY` antes de
llamar a `init_chat_model`.

---

## 2. `create_agent` — Orquestación del agente

**Import:** `from langchain.agents import create_agent`

**Dónde:** `src/carnicos_kb/qa_system.py`, métodos `_build_agent` y `clear_memory`.

**Qué cambiar:**

Eliminar:

```python
from langgraph.prebuilt import create_react_agent
```

Reemplazar `_build_agent` con:

```python
def _build_agent(self):
    return create_agent(
        model=self.llm,
        tools=[self._rag_tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=self._checkpointer,
        middleware=[self._hitl_middleware],  # ver sección 3
        name="carnicos_qa_agent",
    )
```

El parámetro `prompt=` de `create_react_agent` se llama `system_prompt=` en
`create_agent`. El resto del comportamiento (loop ReAct, invocación con
`{"messages": [...]}`) es idéntico y no requiere cambios en `answer_with_trace`.

---

## 3. `HumanInTheLoopMiddleware` — Control de flujos críticos

**Import:** `from langchain.agents import HumanInTheLoopMiddleware`
(verificar import exacto; puede estar en `langchain.agents.middleware`)

**Dónde:** `src/carnicos_kb/qa_system.py`.

**Qué agregar:**

Instanciar el middleware en `__init__` antes de `_build_agent`:

```python
self._hitl_middleware = HumanInTheLoopMiddleware(
    # El middleware intercepta la llamada a la herramienta RAG
    # antes de ejecutarla, permitiendo inspección o aprobación.
    # Configurar para que solo interrumpa en herramientas críticas
    # o cuando el agente vaya a invocar más de N veces seguidas.
)
```

Pasar `middleware=[self._hitl_middleware]` a `create_agent` (ver sección 2).

El middleware debe activarse al menos en una ruta del flujo del agente. Documentar
en el docstring de `_build_agent` qué condición lo activa y por qué se considera
una operación crítica en el contexto de un sistema documental.

---

## 4. `RecursiveCharacterTextSplitter` — Chunking de la base de conocimiento

**Import:** `from langchain_text_splitters import RecursiveCharacterTextSplitter`

**Dónde:** `src/carnicos_kb/chunking.py` y `src/carnicos_kb/rag_index_builder.py`.

**Qué cambiar en `chunking.py`:**

La función `build_chunks` usa lógica manual (`split_by_headings`, `split_section`,
`split_long_block`). Reemplazar el núcleo del chunking con:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def build_chunks_with_splitter(
    input_dir: Path,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    documents: list[Document] = []
    for md_file in sorted(input_dir.glob("*.md")):
        text = clean_text(md_file.read_text(encoding="utf-8"))
        if not text:
            continue
        docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": md_file.as_posix(), "title": md_file.stem}],
        )
        documents.extend(docs)
    return documents
```

La función `build_chunks` original puede mantenerse para compatibilidad con el
comando `carnicos-chunk`, pero `rag_index_builder.py` debe usar
`build_chunks_with_splitter` para construir el índice vectorial.

**Qué cambiar en `rag_index_builder.py`:**

`load_chunks_from_markdown` y `format_chunk_for_embedding` se eliminan.
La entrada al índice pasa a ser la lista de `Document` devuelta por
`build_chunks_with_splitter`.

---

## 5. `langchain_core.vectorstores` y embeddings nativos de LangChain

**Imports clave:**

```python
from langchain_core.vectorstores import InMemoryVectorStore   # o FAISS si se prefiere persistencia
from langchain_core.embeddings import Embeddings               # clase base
from langchain_openai import OpenAIEmbeddings                  # implementación concreta
```

**Dónde:** `src/carnicos_kb/vector_retriever_tool.py` y
`src/carnicos_kb/rag_index_builder.py`.

**Qué cambiar:**

Eliminar toda dependencia de backends vectoriales obsoletos. Reemplazar
retrievers legacy con una clase `LangChainRetriever` que use
`InMemoryVectorStore` (para desarrollo) o `FAISS` (para producción con
persistencia):

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

class LangChainRetriever:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, max_chunks: int = 3):
        self.max_chunks = max_chunks
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def load_from_documents(self, documents: list[Document]) -> None:
        self.vector_store.add_documents(documents)

    def as_retriever(self, k: int | None = None):
        return self.vector_store.as_retriever(search_kwargs={"k": k or self.max_chunks})
```

En `rag_index_builder.py`, cargar los `Document` en PGVector a traves de
`PGVectorRetriever.index_documents`.

`paths.py` conserva solo las constantes activas de datos, PostgreSQL y coleccion.

---

## 6. `dynamic_prompt` — Alimentar el RAG Chain

**Import:** `from langchain_core.prompts import ChatPromptTemplate`
(el "dynamic prompt" es una `ChatPromptTemplate` con variables de contexto
inyectadas dinámicamente via retriever; no es una clase separada sino un patrón
de uso de `ChatPromptTemplate` + `RunnablePassthrough`).

**Dónde:** `src/carnicos_kb/vector_retriever_tool.py`.

**Qué construir:**

Crear una función `build_rag_chain` que arme la cadena RAG con prompt dinámico:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el asistente documental de Alimentos Carnicos S.A.S.\n\n"
     "Contexto recuperado de la base documental:\n{context}\n\n"
     "Responde SOLO con base en el contexto anterior. "
     "Si no hay evidencia suficiente, indícalo."),
    ("human", "{question}"),
])

def build_rag_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n---\n\n".join(
            render_langchain_document(doc) for doc in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
```

La herramienta documental (`StructuredTool`) debe invocar esta cadena en lugar
de llamar directamente a `similarity_search`:

```python
def consultar_base_documental(query: str) -> str:
    return rag_chain.invoke(query)
```

---

## 7. `PostgresSaver` — Checkpointer de memoria persistente

**Import:** `from langgraph.checkpoint.postgres import PostgresSaver`

**Dónde:** `src/carnicos_kb/qa_system.py`, método `_create_checkpointer`.

**Qué cambiar:**

Eliminar la logica de checkpointers locales como opciones primarias.
`PostgresSaver` es el checkpointer obligatorio. `InMemorySaver` puede mantenerse
únicamente como fallback de desarrollo (cuando `DATABASE_URL` no está configurada).

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

def _create_checkpointer(self):
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        try:
            conn = psycopg.connect(database_url, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()   # crea las tablas si no existen
            self._log(f"  Checkpointer: PostgreSQL ({database_url[:30]}...)")
            return saver
        except Exception as exc:
            self._log(f"  PostgreSQL no disponible ({exc}); usando InMemorySaver.")
    self._log("  Checkpointer: InMemorySaver (solo desarrollo)")
    return InMemorySaver()
```

Agregar en `paths.py`:

```python
DEFAULT_POSTGRES_URL = "postgresql://user:password@localhost:5432/carnicos_kb"
```

La variable de entorno que controla la conexión es `DATABASE_URL` (estándar de
la industria). Documentarla en el `README.md` y en `.env.example`.

---

## Dependencias: cambios en `pyproject.toml`

### Agregar

```toml
"langchain-text-splitters>=0.3",
"langgraph-checkpoint-postgres>=2.0",
"psycopg[binary]>=3.1",
```

No agregar dependencias de backends vectoriales locales no usados.

---

## Variables de entorno nuevas (`.env`)

```dotenv
# Checkpointer PostgreSQL (obligatorio en producción)
DATABASE_URL=postgresql://user:password@localhost:5432/carnicos_kb

PG_COLLECTION_NAME=carnicos_rag
```

---

## Tests: qué actualizar

| Archivo de test | Cambio requerido |
|---|---|
| `test_vector_retriever_tool.py` | Usar fixtures de `InMemoryVectorStore` y PGVector mockeado |
| `test_rag_index_builder.py` | Adaptar al nuevo flujo con `RecursiveCharacterTextSplitter` + vector store nativo |
| `test_agent_contract.py` | Cambiar `create_react_agent` → `create_agent`; verificar que el contrato de mensajes sigue siendo `{"messages": [...]}` |
| `test_conversation_memory.py` | Usar `PostgresSaver` con una BD de test o un `InMemorySaver` mockeado |
| `test_routing_end_to_end.py` | Verificar que `HumanInTheLoopMiddleware` no bloquea el flujo en tests automáticos (configurar con `auto_approve=True` en test fixtures) |
| `test_chunking.py` | Agregar tests para `build_chunks_with_splitter` junto a los tests existentes |

---

## Restricciones de implementación

1. Los siete componentes obligatorios deben aparecer en el código fuente del
   repositorio con sus imports exactos tal como se listan en cada sección.
2. No usar `create_react_agent` de `langgraph.prebuilt` como agente principal.
3. Usar PGVector como vector store de produccion.
4. No usar checkpointers locales de archivo en produccion.
5. `ChatOpenAI(...)` no debe aparecer como método de inicialización del LLM
   principal; debe reemplazarse por `init_chat_model`.
6. El `AGENT_SYSTEM_PROMPT` existente debe conservarse íntegro; solo cambia
   cómo se pasa al agente (`system_prompt=` en lugar de `prompt=`).
7. La interfaz pública de `CarnicosQASystem` (`answer`, `answer_with_trace`,
   `interactive_chat`, `clear_memory`) debe mantenerse sin cambios de firma.
