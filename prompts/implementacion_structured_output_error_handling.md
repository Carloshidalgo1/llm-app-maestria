# Prompt de implementación: Structured Output y Gestión de Errores

## Contexto

Sistema Q&A de Alimentos Carnicos S.A.S. basado en LangChain + LangGraph.
Archivos principales:
- `src/carnicos_kb/vector_retriever_tool.py` — define los `StructuredTool` RAG
- `src/carnicos_kb/qa_system.py` — define `CarnicosQASystem` y el agente

---

## Tarea

Implementa las siguientes tres mejoras:

---

### 1. Esquema Pydantic estricto para la herramienta RAG (`args_schema`)

**Archivo:** `src/carnicos_kb/vector_retriever_tool.py`

Actualmente los dos `StructuredTool.from_function` (en `build_pgvector_documental_knowledge_tool`
y `build_langchain_documental_knowledge_tool`) usan solo `query: str` sin `args_schema`.
El LLM no tiene un JSON Schema explícito que valide su entrada antes de ejecutar la herramienta.

Define el siguiente modelo Pydantic una sola vez, antes de las dos funciones `build_*`,
y pásalo como `args_schema` en ambas llamadas a `StructuredTool.from_function`:

```python
from pydantic import BaseModel, Field

class DocumentalQueryInput(BaseModel):
    """Parámetros de búsqueda en la base documental de Alimentos Carnicos S.A.S."""

    query: str = Field(
        description=(
            "Consulta en español, concisa y con términos específicos del dominio "
            "(marca, sede, NIT, teléfono, horario, proceso, producto). "
            "No incluir texto de instrucciones ni comandos al sistema."
        ),
        min_length=3,
        max_length=300,
    )
```

Ejemplo de uso en `StructuredTool.from_function`:

```python
tool = StructuredTool.from_function(
    name=DOCUMENTAL_KNOWLEDGE_TOOL_NAME,
    func=consultar_base_documental,
    description=_RAG_TOOL_DESCRIPTION,
    args_schema=DocumentalQueryInput,   # <-- agregar
)
```

---

### 2. Esquema Pydantic para la respuesta del agente (`response_format`)

**Archivo:** `src/carnicos_kb/qa_system.py`

Define el siguiente modelo antes de la clase `CarnicosQASystem`:

```python
from pydantic import BaseModel, Field

class AgentResponseSchema(BaseModel):
    """Respuesta estructurada del agente Q&A."""

    answer: str = Field(
        description=(
            "Respuesta final en español profesional sobre Alimentos Carnicos S.A.S. "
            "Si la herramienta falló o no devolvió información suficiente, responder "
            "con cortesía: 'En este momento no pude verificar [dato], pero puedo "
            "ayudarte con preguntas sobre [tema alternativo relacionado].'"
        )
    )
    tool_was_called: bool = Field(
        default=False,
        description="True si se invocó la herramienta documental para responder.",
    )
    confidence: str = Field(
        default="unknown",
        description=(
            "Nivel de confianza en la respuesta: 'high' (evidencia documental directa), "
            "'medium' (inferencia razonable), 'low' (sin evidencia suficiente)."
        ),
    )
```

En `_build_agent`, pasa `response_format=AgentResponseSchema` a `create_agent`:

```python
def _build_agent(self):
    return create_agent(
        model=self.llm,
        tools=[self._rag_tool],
        middleware=[self._rag_prompt_middleware],
        checkpointer=self._checkpointer,
        name="carnicos_qa_agent",
        response_format=AgentResponseSchema,   # <-- agregar
    )
```

En `answer_with_trace`, extrae la respuesta desde `result["structured_response"]`
cuando esté disponible; si no, haz fallback al último `AIMessage.content` como antes.
Agrega `confidence: str = "unknown"` como campo opcional al dataclass `QAResponse`.

---

### 3. Manejo de errores resiliente en la herramienta RAG

**Archivo:** `src/carnicos_kb/vector_retriever_tool.py`

En ambas funciones internas `consultar_base_documental` (dentro de `build_pgvector_*`
y `build_langchain_*`), envuelve la llamada a `retriever.search` en un `try/except`
para que un fallo (timeout, error de conexión PGVector, error de embeddings) no
propague una excepción al agente, sino que devuelva un mensaje que el agente pueda
usar para responder cortésmente:

```python
def consultar_base_documental(query: str) -> str:
    try:
        return retriever.search(query)
    except Exception as exc:
        return (
            f"[HERRAMIENTA_ERROR] La búsqueda documental no pudo completarse "
            f"({type(exc).__name__}). El agente debe informar al usuario que "
            f"no pudo verificar el dato en este momento y ofrecer ayuda alternativa."
        )
```

Aplica este patrón en las dos implementaciones (PGVector e InMemoryVectorStore).

---

## Restricciones

- No cambies la firma pública de ninguna función ni el contrato de `QAResponse`
  (solo añade el campo `confidence: str = "unknown"` si implementas el paso 2).
- No agregues imports innecesarios; reutiliza los ya existentes en cada archivo.
- No modifiques `AGENT_SYSTEM_PROMPT`.
- Si `create_agent` no acepta `response_format` en la versión instalada del paquete,
  omite el paso 2 y documenta el motivo en un comentario inline de una sola línea.
- Al finalizar, corre `ruff check src/carnicos_kb/` y corrige cualquier error reportado.

---

## Resumen de cambios esperados

| Paso | Archivo | Cambio |
|---|---|---|
| `DocumentalQueryInput` + `args_schema` | `vector_retriever_tool.py` | JSON Schema estricto que el LLM debe respetar al invocar la herramienta |
| `AgentResponseSchema` + `response_format` | `qa_system.py` | Fuerza al agente a emitir JSON validado con `answer`, `tool_was_called` y `confidence` |
| `try/except` en `consultar_base_documental` | `vector_retriever_tool.py` | Aísla fallos de red/embedding y devuelve un mensaje que guía la respuesta cortés del agente |
