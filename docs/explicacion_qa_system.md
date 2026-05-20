# Explicacion de `src/carnicos_kb/qa_system.py`

Este documento explica en detalle el script `src/carnicos_kb/qa_system.py`, que es
el punto central del sistema de preguntas y respuestas sobre Alimentos Carnicos
S.A.S.

El archivo construye un agente conversacional con LangChain, conecta ese agente
a una herramienta RAG documental, maneja memoria por conversacion y expone una
interfaz publica simple para consola, Streamlit y tests.

## Responsabilidad general

`qa_system.py` hace cinco cosas principales:

1. Carga la configuracion desde `.env`.
2. Inicializa el modelo de lenguaje con `init_chat_model`.
3. Construye el recuperador documental RAG.
4. Construye un agente LangChain con memoria/checkpointer.
5. Expone metodos para responder preguntas y devolver trazabilidad.

El flujo resumido es:

```text
Pregunta del usuario
  |
  v
CarnicosQASystem.answer_with_trace()
  |
  v
create_agent(...)
  |
  v
dynamic_prompt recupera contexto RAG
  |
  v
LLM decide/responde usando la herramienta documental
  |
  v
QAResponse(answer, tool_name, tool_output)
```

## Dependencias principales

El script importa componentes de varias capas:

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
```

Tambien intenta importar soporte PostgreSQL:

```python
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver as _PostgresSaver
```

Si esos imports fallan, el sistema marca `_HAS_POSTGRES = False` y usa
`InMemorySaver` como fallback de desarrollo.

Desde el proyecto local importa:

```python
from .vector_retriever_tool import (
    build_dynamic_rag_prompt,
    build_langchain_documental_knowledge_tool,
    build_pgvector_documental_knowledge_tool,
)
from .knowledge_loader import get_knowledge_stats, load_knowledge_base
from .paths import DEFAULT_CHUNKS_FILE, DEFAULT_DATASET_DIR, DEFAULT_PG_COLLECTION
```

Eso significa que `qa_system.py` no implementa directamente el vector store. Lo
delegado a `vector_retriever_tool.py` es:

- crear la herramienta documental;
- crear el retriever PGVector o InMemory;
- renderizar los documentos recuperados;
- construir el middleware `dynamic_prompt` que inyecta contexto RAG.

## Variables y constantes

Al inicio se ejecuta:

```python
load_dotenv()
```

Con eso, el script lee variables de `.env` como:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_TOKENS`
- `OPENAI_EMBEDDING_MODEL`
- `DATABASE_URL`
- `PG_COLLECTION_NAME`
- `CARNICOS_KNOWLEDGE_PATH`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`

Las constantes por defecto son:

```python
DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
```

Puntos importantes:

- `DEFAULT_MODEL` ya usa el formato `provider:model`, requerido por
  `init_chat_model`.
- `DEFAULT_EMBEDDING_MODEL` se usa para recuperar documentos por similitud.
- `DEFAULT_CHUNK_SIZE` y `DEFAULT_CHUNK_OVERLAP` aplican solo cuando se usa el
  fallback `InMemoryVectorStore`.

## Prompt del agente

`AGENT_SYSTEM_PROMPT` es el prompt de sistema principal. Es una pieza critica
porque define las reglas de seguridad, alcance y precision factual.

Sus responsabilidades son:

- limitar el dominio a Alimentos Carnicos S.A.S.;
- tratar mensajes del usuario, historial y RAG como datos no confiables;
- exigir evidencia recuperada para datos factuales;
- prohibir inventar telefonos, NIT, sedes, horarios, cargos, URLs, etc.;
- definir como manejar inconsistencias documentales;
- proteger contra inyeccion de prompt directa e indirecta;
- definir el estilo de respuesta.

La jerarquia declarada en el prompt es:

```text
[SISTEMA]   Prompt del sistema
[EVIDENCIA] Fragmentos recuperados por RAG
[CONTEXTO]  Historial de conversacion
[ENTRADA]   Mensajes del usuario
```

La idea central es que el modelo no debe obedecer instrucciones escondidas en
el usuario, el historial o los documentos recuperados. Esos textos son datos,
no instrucciones de sistema.

## `QAResponse`

La clase:

```python
@dataclass(frozen=True)
class QAResponse:
    answer: str
    tool_name: str = "base_documental_carnicos"
    tool_reason: str = ""
    tool_output: str = ""
```

representa la respuesta publica del sistema.

Campos:

- `answer`: respuesta final para el usuario.
- `tool_name`: nombre de la herramienta usada. Por defecto es
  `base_documental_carnicos`.
- `tool_reason`: campo reservado para explicar la seleccion de herramienta.
  Actualmente queda vacio.
- `tool_output`: contenido devuelto por la herramienta RAG, recopilado desde
  los `ToolMessage`.

Es `frozen=True`, por lo que sus campos no se modifican despues de creada.

## Lectura robusta de variables numericas

Hay dos helpers:

```python
def _env_float(name: str, default: float) -> float
def _env_int(name: str, default: int) -> int
```

Ambos leen una variable de entorno y la convierten al tipo esperado. Si la
variable no existe o tiene un valor invalido, devuelven el valor por defecto.

Esto evita que una configuracion mal escrita como:

```env
OPENAI_TEMPERATURE=abc
```

rompa la inicializacion completa del sistema.

## Clase `CarnicosQASystem`

`CarnicosQASystem` es la clase principal del archivo. Encapsula el modelo, la
base de conocimiento, el retriever, el agente y el checkpointer.

La firma del constructor es:

```python
def __init__(
    self,
    knowledge_dir: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    verbose: bool = True,
)
```

Permite inyectar configuracion manualmente desde tests o scripts, pero si no se
pasa nada toma valores desde `.env` o desde las constantes por defecto.

### Paso 1: configuracion inicial

El constructor guarda:

```python
self.verbose
self.model
self.temperature
self.max_tokens
```

La prioridad es:

1. argumento explicito del constructor;
2. variable de entorno;
3. constante por defecto.

Por ejemplo, para el modelo:

```python
self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
```

### Paso 2: validacion de `OPENAI_API_KEY`

El sistema exige una clave de OpenAI:

```python
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(...)
```

Aunque `init_chat_model` lee la clave automaticamente del entorno, esta
validacion permite fallar temprano con un mensaje claro.

### Paso 3: ubicacion de la base de conocimiento

Si `knowledge_dir` no fue pasado, llama a:

```python
self._find_knowledge_path()
```

El metodo revisa estas rutas en orden:

```text
CARNICOS_KNOWLEDGE_PATH
data/processed/base_conocimiento_chunks.md
data/processed/dataset_carnicos
base_conocimiento_chunks.md
dataset_carnicos
```

La primera ruta existente se usa como fuente. Si ninguna existe, lanza
`FileNotFoundError`.

### Paso 4: carga de conocimiento y estadisticas

Luego carga la base:

```python
self.knowledge_base = load_knowledge_base(knowledge_path, verbose=False)
self.stats = get_knowledge_stats(self.knowledge_base)
```

Esto se hace incluso cuando el vector store real esta en PostgreSQL. Sirve para
mantener estadisticas y tener una fuente local disponible para fallback.

### Paso 5: inicializacion del LLM

El modelo se crea con:

```python
self.llm = init_chat_model(
    model=self.model,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
)
```

No se instancia `ChatOpenAI` directamente. `init_chat_model` permite usar el
formato `openai:gpt-4o-mini` y delega en LangChain la seleccion del proveedor.

### Paso 6: seleccion de backend RAG

El constructor lee:

```python
self._database_url = os.getenv("DATABASE_URL", "").strip()
```

Si `DATABASE_URL` existe:

- no construye documentos locales;
- asume que la coleccion PGVector ya fue poblada;
- el RAG recupera desde PostgreSQL.

Si `DATABASE_URL` no existe:

- construye documentos locales con `_build_documents`;
- crea un `InMemoryVectorStore`;
- usa este backend como fallback de desarrollo.

## `_build_documents`

Este metodo solo se usa cuando no hay `DATABASE_URL`.

Toma la base de conocimiento local y la divide con:

```python
RecursiveCharacterTextSplitter(
    chunk_size=_env_int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
    chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Si `knowledge_path` es archivo:

- crea documentos desde `self.knowledge_base`;
- usa metadata `source` y `title`.

Si `knowledge_path` es directorio:

- recorre los `.md`;
- ignora archivos vacios;
- crea documentos por archivo con metadata de fuente.

El resultado es:

```python
list[Document]
```

que luego se indexa en memoria.

## `_build_rag_tool`

Este metodo decide que herramienta RAG construir.

Primero lee:

```python
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
collection = os.getenv("PG_COLLECTION_NAME", DEFAULT_PG_COLLECTION)
```

Si hay `DATABASE_URL`, usa:

```python
build_pgvector_documental_knowledge_tool(...)
```

Ese camino crea:

- un `PGVectorRetriever`;
- una `StructuredTool`;
- busqueda semantica persistente en PostgreSQL.

Si no hay `DATABASE_URL`, usa:

```python
build_langchain_documental_knowledge_tool(...)
```

Ese camino crea:

- un `LangChainRetriever`;
- un `InMemoryVectorStore`;
- una `StructuredTool`;
- busqueda semantica efimera para desarrollo.

Ambos caminos devuelven:

```python
(tool, retriever)
```

El `tool` se entrega al agente. El `retriever` se usa tambien para construir el
middleware de prompt dinamico.

## `HumanInTheLoopMiddleware`

El metodo:

```python
def _build_hitl_middleware(self) -> HumanInTheLoopMiddleware
```

crea un middleware que interrumpiria antes de ejecutar la herramienta RAG:

```python
return HumanInTheLoopMiddleware(
    interrupt_on={self._rag_tool.name: True},
    description_prefix="Aprobacion requerida para consulta documental",
)
```

Sin embargo, el middleware no se pasa actualmente al agente.

Esto es intencional. En `_build_agent` el docstring explica que, sin un flujo de
aprobacion en la UI, activar este middleware puede dejar el estado del agente con
un `AIMessage` que contiene `tool_calls`, pero sin su `ToolMessage`
correspondiente. En el siguiente turno, OpenAI puede rechazar ese historial con
un error 400.

Por eso el agente usa solo:

```python
middleware=[self._rag_prompt_middleware]
```

El HITL queda construido como capacidad preparada, pero desactivada en el flujo
real.

## `_create_checkpointer`

Este metodo define como se guarda la memoria conversacional.

Si `DATABASE_URL` existe y `langgraph-checkpoint-postgres` esta disponible:

```python
conn = psycopg.connect(database_url, autocommit=True)
saver = _PostgresSaver(conn)
saver.setup()
return saver
```

Eso crea o valida las tablas necesarias en PostgreSQL y permite memoria
persistente por `thread_id`.

Si PostgreSQL no esta disponible, cae a:

```python
return InMemorySaver()
```

Ese fallback sirve para desarrollo y tests. No persiste entre reinicios.

## `_build_agent`

El agente se crea con:

```python
return create_agent(
    model=self.llm,
    tools=[self._rag_tool],
    middleware=[self._rag_prompt_middleware],
    checkpointer=self._checkpointer,
    name="carnicos_qa_agent",
)
```

Elementos clave:

- `model`: LLM inicializado por `init_chat_model`.
- `tools`: solo incluye la herramienta RAG documental.
- `middleware`: incluye el prompt dinamico RAG.
- `checkpointer`: mantiene memoria por `thread_id`.
- `name`: identifica el agente en trazas y estados.

El prompt de sistema no se pasa como parametro directo. Se inyecta mediante
`build_dynamic_rag_prompt`, que combina:

```text
AGENT_SYSTEM_PROMPT
+
contexto recuperado de la base documental
```

antes de cada llamada al modelo.

## `answer`

Metodo simple:

```python
def answer(self, question: str, thread_id: str = "default") -> str
```

Internamente llama:

```python
self.answer_with_trace(...).answer
```

Sirve cuando solo se quiere el texto final, sin trazabilidad.

## `_repair_thread_state`

Este metodo es defensivo. Su objetivo es reparar conversaciones que quedaron
con tool calls pendientes.

Problema que resuelve:

```text
AIMessage(tool_calls=[...])
pero no existe ToolMessage correspondiente
```

Ese estado puede ocurrir si un flujo HITL interrumpe antes de ejecutar la
herramienta. OpenAI exige que cada tool call tenga una respuesta de herramienta.

El metodo:

1. obtiene el estado del agente para un `thread_id`;
2. busca `AIMessage` con tool calls;
3. revisa cuales tool call ids ya tienen `ToolMessage`;
4. para los pendientes, crea `ToolMessage` sinteticos;
5. actualiza el estado del agente.

El contenido sintetico es:

```text
[Herramienta no ejecutada - interrupcion del sistema. Reintentando.]
```

Esto permite que el siguiente turno no falle por historial inconsistente.

## `answer_with_trace`

Este es el metodo principal de inferencia.

Firma:

```python
def answer_with_trace(
    self,
    question: str,
    thread_id: str = "default",
) -> QAResponse
```

Flujo:

1. Valida que la pregunta no este vacia.
2. Repara el estado del hilo con `_repair_thread_state`.
3. Construye la config del checkpointer:

```python
config = {"configurable": {"thread_id": thread_id}}
```

4. Invoca el agente:

```python
result = self._agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config,
)
```

5. Extrae todos los mensajes del resultado.
6. Busca el ultimo `AIMessage` como respuesta final.
7. Recopila los `ToolMessage` como `tool_output`.
8. Construye un `QAResponse`.
9. Guarda esa respuesta en `self.last_response`.

Si ocurre una excepcion, no la propaga. Devuelve:

```python
QAResponse(
    answer=f"Error al procesar la pregunta: {exc}",
    tool_name="error",
)
```

Esto hace que la UI o la consola reciban siempre un objeto de respuesta
controlado.

## Memoria por `thread_id`

La memoria no se maneja con una lista manual de mensajes. La maneja el
checkpointer de LangGraph/LangChain.

Cada conversacion se identifica con:

```python
thread_id
```

Si dos usuarios usan distintos `thread_id`, sus conversaciones quedan separadas.

Ejemplo:

```python
qa.answer_with_trace("Cual es el NIT?", thread_id="usuario-1")
qa.answer_with_trace("Y la sede principal?", thread_id="usuario-1")

qa.answer_with_trace("Cual es el telefono?", thread_id="usuario-2")
```

`usuario-1` y `usuario-2` tienen memorias independientes.

## `clear_memory`

```python
def clear_memory(self) -> None:
    self._checkpointer = InMemorySaver()
    self._agent = self._build_agent()
```

Reinicia la memoria en RAM y reconstruye el agente. Es util para limpiar una
sesion local de desarrollo.

Punto de cuidado: aunque el sistema estuviera usando PostgreSQL, este metodo
cambia el checkpointer a `InMemorySaver`. Si se requiere limpiar memoria
persistente en PostgreSQL, habria que implementar un borrado explicito por
thread o tabla.

## `interactive_chat`

Este metodo permite usar el sistema desde consola.

Flujo:

1. imprime encabezado;
2. crea un `thread_id` aleatorio con `uuid.uuid4()`;
3. entra en un bucle de preguntas;
4. sale si el usuario escribe `salir`, `exit` o `quit`;
5. ignora preguntas vacias;
6. llama `answer_with_trace`;
7. imprime la respuesta final.

Como el `thread_id` se crea una sola vez antes del bucle, toda la sesion de
consola comparte memoria conversacional.

## `main`

`main()` es el punto de entrada CLI.

Si el usuario ejecuta:

```powershell
python -m carnicos_kb.qa_system "Cual es el NIT?"
```

el script toma los argumentos como una sola pregunta y responde una vez.

Si se ejecuta sin argumentos:

```powershell
python -m carnicos_kb.qa_system
```

abre el chat interactivo.

El `pyproject.toml` tambien expone este entrypoint como:

```text
carnicos-qa = "carnicos_kb.qa_system:main"
```

## Relacion con `vector_retriever_tool.py`

`qa_system.py` no sabe los detalles internos de PGVector o InMemoryVectorStore.
Solo llama builders.

La separacion es:

```text
qa_system.py
  - orquesta agente, memoria, LLM y flujo conversacional

vector_retriever_tool.py
  - crea retrievers
  - crea herramientas LangChain
  - renderiza documentos recuperados
  - construye dynamic_prompt
```

Esto permite cambiar el backend de recuperacion sin reescribir la clase
`CarnicosQASystem`.

## Relacion con `rag_index_builder.py`

`rag_index_builder.py` se usa antes de ejecutar el agente en modo produccion.

Su trabajo es:

1. leer `data/processed/dataset_carnicos`;
2. dividir Markdown en documentos;
3. calcular embeddings;
4. guardar documentos en PGVector.

Luego `qa_system.py`, cuando detecta `DATABASE_URL`, consulta esa coleccion con
`PGVectorRetriever`.

## Escenarios de ejecucion

### Desarrollo sin PostgreSQL

```env
OPENAI_API_KEY=...
# DATABASE_URL no configurada
```

Resultado:

- carga Markdown local;
- divide en chunks con `RecursiveCharacterTextSplitter`;
- crea `InMemoryVectorStore`;
- usa `InMemorySaver`;
- no persiste indice ni memoria entre reinicios.

### Produccion con PostgreSQL

```env
OPENAI_API_KEY=...
DATABASE_URL=postgresql://user:password@localhost:5432/carnicos_kb
PG_COLLECTION_NAME=carnicos_rag
```

Resultado:

- usa PGVector para recuperar documentos;
- usa PostgresSaver para memoria persistente;
- requiere haber ejecutado antes `carnicos-build-rag`;
- no construye documentos locales para el vector store.

## Puntos de cuidado

- `HumanInTheLoopMiddleware` se crea pero no se activa en `create_agent`.
- `tool_reason` existe en `QAResponse`, pero actualmente no se llena.
- Si `DATABASE_URL` esta configurada pero PostgreSQL falla, el checkpointer cae
  a memoria, pero el retriever PGVector puede seguir fallando si no hay conexion.
- `clear_memory()` siempre cambia el checkpointer a `InMemorySaver`.
- El sistema captura excepciones en `answer_with_trace`, por lo que errores de
  backend se devuelven como texto de error en `QAResponse`.
- Si se usa el fallback InMemory, cada reinicio reconstruye el vector store.

## Resumen final

`qa_system.py` es el orquestador del agente. No es el indexador, no es el scraper
y no es el backend vectorial. Su responsabilidad es conectar:

- configuracion de entorno;
- base de conocimiento;
- LLM;
- herramienta RAG;
- prompt dinamico;
- memoria por thread;
- respuesta publica.

La clase principal, `CarnicosQASystem`, es la API que deben consumir la consola,
la UI Streamlit y cualquier script que necesite hacer preguntas al sistema.
