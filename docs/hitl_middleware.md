# Human-in-the-Loop Middleware — Documentación Técnica

**Proyecto:** Asistente Q&A Alimentos Cárnicos S.A.S.  
**Módulo:** Control de flujos críticos con `HumanInTheLoopMiddleware`  
**Stack:** LangChain · LangGraph · Streamlit · PostgreSQL

---

## 1. ¿Qué es Human-in-the-Loop (HITL)?

Human-in-the-Loop es un patrón de diseño para sistemas de inteligencia artificial en el que **un operador humano interviene en el flujo de ejecución del agente** antes de que una acción de alto impacto sea ejecutada. El humano puede aprobar, modificar o rechazar la acción.

En sistemas de IA autónomos, el agente toma decisiones y ejecuta herramientas sin supervisión. HITL introduce un punto de control explícito:

```
Sin HITL:   Usuario → Agente → Herramienta → Respuesta
Con HITL:   Usuario → Agente → [PAUSA] → Operador decide → Herramienta → Respuesta
```

El concepto proviene de la ingeniería de control (sistemas críticos, aviación, medicina) y se ha adoptado en LLM Ops como mecanismo de **supervisión, auditoría y prevención de errores** en agentes conversacionales.

---

## 2. Motivación en este proyecto

El asistente Q&A consulta una base documental de Alimentos Cárnicos S.A.S. mediante RAG (Retrieval-Augmented Generation). Antes de buscar en esa base, el agente formula una **query al vector store** — una búsqueda semántica que puede recuperar información sensible.

Sin HITL, el flujo es completamente automático:

```
Pregunta del usuario
    → LLM genera query RAG
    → Vector store devuelve fragmentos documentales
    → LLM genera respuesta con esos fragmentos
    → Respuesta entregada al usuario
```

Con información sensible (precios, contratos, datos de personal, identificadores fiscales), **ningún humano ha revisado qué se recuperó ni qué se respondió**. HITL resuelve esto.

---

## 3. Arquitectura del sistema con HITL

### 3.1 Componentes involucrados

| Componente | Rol |
|---|---|
| `HumanInTheLoopMiddleware` | Intercepta el flujo después del modelo y antes de ejecutar la herramienta |
| `InterruptOnConfig` | Define las acciones permitidas y la descripción dinámica para el operador |
| `interrupt()` de LangGraph | Pausa el grafo y guarda el estado en el checkpointer |
| `Command(resume=...)` | Reanuda el grafo con la decisión del operador |
| `PostgresSaver` / `InMemorySaver` | Persiste el estado del grafo mientras espera la decisión |
| `_describe_rag_interrupt()` | Clasifica la consulta y genera el texto informativo para el operador |
| `HITL_CRITICAL_PATTERNS` | Lista de patrones regex que califican una consulta como crítica |
| `CarnicosQASystem.resume_with_decision()` | Método que construye el `Command` correcto según la decisión |
| Panel Streamlit | Interfaz visual con botones Aprobar / Editar / Rechazar |

### 3.2 Posición del middleware en el grafo LangGraph

`create_agent` construye internamente un `StateGraph` con dos nodos principales:

```
  ┌─────────────┐     tool_calls?     ┌─────────────┐
  │  model node │ ──────────────────▶ │  tools node │
  └─────────────┘                     └─────────────┘
         ▲                                   │
         └───────────────────────────────────┘
                    siempre

  Middleware HITL actúa aquí:
  ┌─────────────┐   after_model()   ┌──────────────────┐
  │  model node │ ─────────────────▶│ interrupt(HITL)  │  ← grafo se pausa
  └─────────────┘                   └──────────────────┘
```

El middleware se conecta en el hook `after_model`: se ejecuta **después** de que el LLM genera el `AIMessage` con `tool_calls`, pero **antes** de que el nodo de herramientas ejecute la búsqueda en el vector store.

---

## 4. Flujo completo con HITL activo

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Usuario envía pregunta                                          │
│     "¿Cuál es el precio de los productos Rica?"                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. LLM (GPT-4o-mini) genera AIMessage con tool_call               │
│     tool: base_documental_carnicos                                  │
│     args: {"query": "precio productos Rica"}                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. after_model() del middleware se activa                          │
│     - Detecta que la herramienta está en interrupt_on               │
│     - Llama a _describe_rag_interrupt():                            │
│       · query: "precio productos Rica"                              │
│       · patrón "precio" → clasificación: CRITICA                    │
│       · pregunta original del usuario incluida                      │
│     - Crea HITLRequest con ActionRequest y ReviewConfig             │
│     - Llama a interrupt(HITLRequest) → GRAFO SE PAUSA              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Estado guardado en checkpointer (PostgreSQL / InMemory)         │
│     - AIMessage con tool_calls pendiente                            │
│     - Interrupt con payload de la consulta                          │
│     - agent.get_state(config).interrupts → no vacío                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Streamlit muestra el panel de aprobación al operador            │
│     ┌────────────────────────────────────────────────┐              │
│     │ 🔒 El agente requiere aprobación               │              │
│     │ Consulta CRITICA al RAG documental             │              │
│     │ Query: "precio productos Rica"                 │              │
│     │ Pregunta original: ¿Cuál es el precio...?      │              │
│     │ [Aprobar] [Editar query ▼] [Rechazar ▼]        │              │
│     └────────────────────────────────────────────────┘              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         APROBAR          EDITAR       RECHAZAR
              │             │             │
              ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. resume_with_decision() construye Command(resume=...)            │
│     Aprobar:  {"type": "approve"}                                   │
│     Editar:   {"type": "edit", "edited_action": {"args": {query}}}  │
│     Rechazar: → respuesta forzada, no se reanuda con datos reales   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  (solo Aprobar y Editar)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. Grafo reanuda → tools node ejecuta herramienta RAG              │
│     → vector store retorna fragmentos documentales                  │
│     → LLM genera respuesta final                                    │
│     → Respuesta entregada al usuario                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Las tres decisiones posibles

### 5.1 Aprobar

El operador confirma que la consulta puede ejecutarse tal como el agente la formuló.

```python
decision = {"type": "approve"}
```

**Resultado:** el grafo reanuda, el vector store recibe la query original, el LLM genera la respuesta con los fragmentos recuperados.

**Indicador visual:** `✅ Operador: consulta aprobada — se ejecutará la búsqueda documental.`

### 5.2 Editar

El operador modifica la query antes de que llegue al vector store. Útil cuando la formulación del LLM es demasiado amplia o contiene términos que podrían recuperar información no deseada.

```python
decision = {
    "type": "edit",
    "edited_action": {
        "name": "base_documental_carnicos",
        "args": {"query": "nueva query ajustada por el operador"}
    }
}
```

**Resultado:** el grafo reanuda con la query modificada. El vector store nunca recibe la query original.

**Indicador visual:** `✏️ Operador: query editada antes de ejecutar. Query enviada: ...`

### 5.3 Rechazar

El operador decide que la consulta no debe ejecutarse. El usuario recibe una respuesta que informa el rechazo.

```python
decision = {
    "type": "reject",
    "message": "Información sensible — no autorizado"
}
```

**Comportamiento especial:** el middleware libera el estado del interrupt para que el grafo no quede bloqueado, pero el resultado generado por el agente (que podría incluir datos del RAG ejecutado de todas formas por el `ToolNode`) se **descarta completamente**. Se retorna una respuesta forzada:

```
"La consulta documental fue rechazada por el operador. Información sensible — no autorizado"
```

**Razón técnica del descarte:** `_process_decision` del middleware, para la decisión "reject", devuelve el `tool_call` original junto al `ToolMessage` sintético de rechazo. Esto hace que `revised_tool_calls` no quede vacío y el `ToolNode` ejecuta la herramienta de todas formas. Para evitar que datos reales del RAG lleguen al usuario tras un rechazo, `resume_with_decision()` fuerza la respuesta independientemente de lo que el agente generó.

**Indicador visual:** `❌ Operador: consulta rechazada. Motivo: ...`

---

## 6. Clasificación de criticidad

Antes de que el operador vea el panel, el sistema clasifica automáticamente la consulta como **CRÍTICA** o **RUTINARIA** usando expresiones regulares sobre la query que el LLM formuló:

```python
HITL_CRITICAL_PATTERNS = [
    r"precio[s]?\b",           # precios de productos
    r"tarifa[s]?\b",           # tarifas comerciales
    r"costo[s]?\b",            # costos internos
    r"contrat[oa]\b",          # contratos
    r"licitaci[oó]n\b",        # licitaciones
    r"n[uú]mero[s]?\s+de\s+(empleado|trabajador|personal)",  # datos de nómina
    r"n[oó]mina\b",            # nómina
    r"\bNIT\b",                # identificadores fiscales
    r"\bRUT\b",
    r"c[eé]dula\b",
    r"proveedor[es]?\b",       # información de proveedores
    r"cliente[s]?\s+nominales?\b",
]
```

La clasificación es **informativa** para el operador — le permite priorizar la atención. El panel de aprobación se muestra siempre que HITL esté activo, independientemente del nivel.

La función `_describe_rag_interrupt` recibe el `tool_call` (TypedDict dict en runtime) y el `AgentState` (también TypedDict dict en runtime), y genera el texto que verá el operador:

```
Consulta CRITICA al RAG documental

Query enviada al vector store:
  precio productos Rica porción

Pregunta original del usuario:
  ¿Cuál es el precio de los productos Rica?
```

---

## 7. Implementación técnica detallada

### 7.1 Configuración del middleware

```python
# qa_system.py — _build_hitl_middleware()

HumanInTheLoopMiddleware(
    interrupt_on={
        self._rag_tool.name: InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=_describe_rag_interrupt,  # callable dinámico
        )
    }
)
```

`interrupt_on` es un dict que mapea nombre de herramienta → configuración. El valor `True` activa las tres decisiones con descripción genérica. `InterruptOnConfig` permite personalizar qué decisiones están disponibles y qué texto ve el operador.

### 7.2 Activación condicional en el agente

```python
# qa_system.py — _build_agent()

middleware = [self._rag_prompt_middleware]      # siempre activo
if self.hitl_enabled:
    middleware.append(self._hitl_middleware)    # solo si HITL está activado

create_agent(
    model=self.llm,
    tools=[self._rag_tool],
    middleware=middleware,
    checkpointer=self._checkpointer,
    ...
)
```

Cuando `hitl_enabled=False`, el agente opera sin interrupciones. Cuando `hitl_enabled=True`, el middleware se añade al pipeline de `create_agent`.

**Nota importante:** cambiar el valor de `hitl_enabled` reconstruye el agente por completo. Esto se logra porque `@st.cache_resource` en Streamlit usa todos los parámetros de `get_qa_system()` como clave de caché — incluido `hitl_enabled`.

### 7.3 Detección del estado interrumpido

```python
# qa_system.py — answer_with_trace()

result = self._agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config,
)

agent_state = self._agent.get_state(config)
if agent_state.interrupts:
    intr = agent_state.interrupts[0]
    intr_value = intr.value  # HITLRequest (TypedDict → dict en runtime)

    action_requests = intr_value.get("action_requests", [])
    first_ar = action_requests[0] if action_requests else {}
    query = first_ar.get("args", {}).get("query", "")
    description = first_ar.get("description", str(intr_value))

    return QAResponse(
        answer="",
        pending_approval=True,
        interrupt_payload={"description": description, "query": query, ...},
    )
```

`agent.get_state(config).interrupts` es una tupla de objetos `Interrupt`. Cada uno tiene un `value` que es el objeto pasado a `interrupt()` — en este caso un `HITLRequest`. Como `HITLRequest` es un `TypedDict`, en runtime es un dict Python puro y se accede con `["clave"]`, no con `.atributo`.

### 7.4 Reanudación con decisión

```python
# qa_system.py — resume_with_decision()

result = self._agent.invoke(
    Command(resume={"decisions": [decision]}),
    config=config,
)
```

`Command(resume=...)` es el mecanismo de LangGraph para responder a un `interrupt()`. El valor pasado en `resume` llega como retorno de la llamada `interrupt(hitl_request)` dentro de `after_model()`. La clave `"decisions"` es la que espera el middleware para procesar las decisiones.

### 7.5 TypedDict: el detalle técnico crítico

Un error frecuente al trabajar con LangGraph y LangChain es suponer que los tipos anotados como `TypedDict` son objetos con atributos. En Python, `TypedDict` es solo una anotación de tipos — en runtime, las instancias son **dicts ordinarios**.

| Tipo | Acceso correcto | Acceso incorrecto |
|---|---|---|
| `AgentState` (TypedDict) | `state["messages"]` o `state.get("messages", [])` | `state.messages` |
| `HITLRequest` (TypedDict) | `intr_value["action_requests"]` | `intr_value.action_requests` |
| `ActionRequest` (TypedDict) | `ar["args"]` | `ar.args` |

El error concreto que se produjo: `state.values` devuelve el método `.values()` del dict (un `builtin_function_or_method`) porque `AgentState` es un dict y los dicts tienen `.values()`. Llamar `.get()` sobre ese método lanza `AttributeError: 'builtin_function_or_method' object has no attribute 'get'`.

---

## 8. Configuración

### 8.1 Variable de entorno

```bash
# .env
HITL_ENABLED=false   # true activa HITL globalmente
```

- `false` (default): el agente responde automáticamente sin interrupciones.
- `true`: cada consulta al vector store requiere aprobación del operador.

### 8.2 Toggle en Streamlit

La interfaz expone un toggle en la barra lateral que sobreescribe el valor de la variable de entorno en tiempo de ejecución:

```
Configuración
├── [toggle] Control humano (HITL)
│   ├── ON  → 🔒 Aprobación requerida en cada consulta RAG
│   └── OFF → ⚡ HITL desactivado — respuestas automáticas
```

**Prioridad:** toggle de UI > variable de entorno `HITL_ENABLED` > default `False`.

---

## 9. Persistencia del estado durante la espera

Mientras el operador decide, el hilo del agente está suspendido. El estado (mensajes, tool_calls pendientes, checkpoint) persiste en:

| Entorno | Mecanismo | Característica |
|---|---|---|
| Producción (`DATABASE_URL` configurada) | `PostgresSaver` | Persistente entre reinicios del servidor |
| Desarrollo (sin `DATABASE_URL`) | `InMemorySaver` | Solo en memoria, se pierde al reiniciar |

El `thread_id` (identificador único de sesión) es la clave que vincula el estado del grafo con la decisión del operador. Streamlit lo mantiene en `st.session_state.thread_id`.

---

## 10. Gestión del estado en Streamlit

Un desafío específico de Streamlit es que **cualquier interacción en la UI (expandir un expander, abrir un popover) provoca un rerun completo del script**. Si el panel HITL solo se mostrara como resultado directo de una respuesta del agente, desaparecería en el primer rerun.

La solución implementada: persistir el payload del interrupt en `st.session_state["hitl_pending_payload"]`.

```
Ciclo de vida del panel HITL en Streamlit:

1. answer_with_trace() devuelve pending_approval=True
   → session_state["hitl_pending_payload"] = payload
   → Panel mostrado

2. Usuario interactúa (expander, popover) → rerun
   → "hitl_pending_payload" existe → panel re-mostrado
   → input de chat BLOQUEADO (no se aceptan nuevas preguntas)

3. Usuario hace clic en decisión → session_state["hitl_decision"] = {...} → rerun
   → "hitl_decision" existe → resume_with_decision() llamado
   → session_state.pop("hitl_pending_payload")
   → badge de decisión mostrado y guardado en historial
   → Respuesta del agente mostrada
   → input de chat VISIBLE (el código no hace return antes del chat_input)
```

Los mensajes de tipo `hitl_decision` se guardan en `st.session_state.messages` con `role="hitl_decision"` para que los badges de decisión persistan al scrollear el historial.

---

## 11. Diagrama de clases simplificado

```
CarnicosQASystem
├── hitl_enabled: bool
├── _hitl_middleware: HumanInTheLoopMiddleware
│   └── interrupt_on: {tool_name: InterruptOnConfig}
│       ├── allowed_decisions: ["approve", "edit", "reject"]
│       └── description: _describe_rag_interrupt (callable)
├── _agent: CompiledStateGraph
│   └── middleware: [rag_prompt_middleware, (hitl_middleware?)]
├── answer_with_trace(question, thread_id) → QAResponse
│   ├── si interrupts → QAResponse(pending_approval=True, interrupt_payload=...)
│   └── si no → QAResponse(answer=..., confidence=...)
└── resume_with_decision(thread_id, decision_type, ...) → QAResponse
    ├── "approve" → Command(resume={"decisions": [{"type": "approve"}]})
    ├── "edit"    → Command(resume={"decisions": [{"type": "edit", "edited_action": ...}]})
    └── "reject"  → Command(resume=...) + respuesta forzada (descarta output real del agente)

QAResponse (dataclass frozen)
├── answer: str
├── tool_name: str
├── tool_output: str
├── confidence: str
├── pending_approval: bool          ← nuevo campo HITL
└── interrupt_payload: dict | None  ← nuevo campo HITL
```

---

## 12. Limitaciones conocidas y decisiones de diseño

### Rechazo no previene ejecución del tool (limitación del middleware)

`_process_decision` del middleware devuelve `(tool_call, tool_message)` para "reject" — donde `tool_call` no es `None`. Esto implica que el `ToolNode` ejecuta la herramienta de todas formas. La solución implementada descarta el resultado y fuerza una respuesta de rechazo explícita en `resume_with_decision()`.

### Sin memoria persistente de decisiones pasadas

Las decisiones del operador no se almacenan fuera de la sesión Streamlit actual. Un sistema en producción debería guardar cada decisión en una tabla de auditoría (PostgreSQL) para cumplimiento normativo y para alimentar un policy store que aprenda a auto-aprobar patrones recurrentes.

### Un solo operador simultáneo

La arquitectura actual asume un único operador por sesión. Para múltiples operadores concurrentes se requeriría un sistema de colas (uno por `thread_id`) y una interfaz de revisión separada del chat del usuario.

### HITL interrumpe solo la herramienta RAG

El middleware está configurado con `interrupt_on = {rag_tool.name: ...}`. Si en el futuro se añaden más herramientas (búsqueda web, acceso a API externa), cada una requeriría su propia entrada en `interrupt_on` y su propia lógica de criticidad.

---

## 13. Referencias

| Concepto | Fuente |
|---|---|
| `HumanInTheLoopMiddleware` | `langchain.agents.middleware.human_in_the_loop` |
| `InterruptOnConfig` | `langchain.agents.middleware.human_in_the_loop` |
| `interrupt()` / `Command` | `langgraph.types` |
| `StateSnapshot.interrupts` | `langgraph.types.StateSnapshot` |
| `create_agent` | `langchain.agents.create_agent` |
| Patrón HITL en LLM Ops | [LangGraph How-to: Human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/) |
