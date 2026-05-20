# Informe Modulo 2 — Agente conversacional con memoria, herramientas y trazabilidad

**Proyecto:** Carnicos Knowledge Base  
**Modulo:** 2 — Agente conversacional  
**Rama:** `modulo-2-agente-conversacional`

**Integrantes:**

- Carlos Alex Macias Perdomo — codigo: 22500208
- Carlos Hidalgo Escobar — codigo: 22502395
- Lorena Portilla — codigo: 22500248
- Luis Carlos Correa — codigo: 22501541

---

## Resumen

Este informe describe la evolucion del sistema de preguntas y respuestas
desarrollado en el Modulo 1 hacia un agente conversacional basado en
LangChain. El nuevo sistema incorpora memoria de sesion, enrutamiento
estructurado entre herramientas especializadas, construccion de un indice
vectorial con embeddings de OpenAI y soporte de trazabilidad mediante
LangSmith. La arquitectura separa explicitamente la decision de recuperacion
de la generacion de la respuesta final, lo que permite auditar el razonamiento
del agente turno a turno. La suite de pruebas unitarias valida 30 casos
distribuidos en ocho modulos de test.

---

## 1. Introduccion

El Modulo 1 implemento un sistema Q&A que consolidaba la base de conocimiento
completa en el prompt de sistema y delegaba toda la recuperacion y generacion
al modelo de lenguaje en un unico paso. Esta aproximacion presenta limitaciones
de escala y auditabilidad: a medida que el corpus crece, el contexto del
modelo se satura, y no existe un mecanismo explicito para distinguir la fuente
de cada respuesta.

El Modulo 2 resuelve estos problemas mediante un patron de agente con
herramientas. Antes de redactar cada respuesta, el sistema ejecuta un router
que clasifica la consulta y selecciona la herramienta adecuada. La respuesta
final se genera exclusivamente a partir del resultado de esa herramienta y del
historial de la sesion, no de la base de conocimiento completa inyectada en el
prompt. Este diseno reduce el consumo de contexto, mejora la trazabilidad y
permite sustituir el mecanismo de recuperacion sin modificar la cadena de
respuesta.

---

## 2. Arquitectura del sistema

### 2.1 Flujo principal de inferencia

El punto de entrada del sistema es `CarnicosQASystem.answer_with_trace()`,
definido en `src/carnicos_kb/qa_system.py`. Cada consulta sigue el flujo:

```text
Usuario
  |
  v
Streamlit (streamlit_app.py) o consola interactiva
  |
  v
CarnicosQASystem.answer_with_trace()
  |
  v
Router LangChain — salida estructurada RouteDecision
  |
  +---> datos_estructurados_carnicos
  |       Fuente: data/structured/carnicos_structured_faq.json
  |       Recuperacion lexica determinista; sin embeddings
  |
  +---> base_documental_carnicos
          Fuente: PostgreSQL/PGVector o InMemoryVectorStore en desarrollo
          Recuperacion vectorial por similitud semantica

Resultado de herramienta + historial normalizado
  |
  v
Cadena de respuesta final (LLM + StrOutputParser)
  |
  v
QAResponse — respuesta, herramienta seleccionada, motivo, salida recuperada
```

La interfaz Streamlit se inicia a traves de `src/carnicos_kb/streamlit_runner.py`,
un lanzador CLI que configura `sys.argv` y delega a `streamlit.web.cli`.
Esto desacopla la ejecucion de la aplicacion del comando `streamlit run` directo,
permitiendo el punto de entrada `make app`.

### 2.2 Flujo de construccion del indice vectorial

La generacion de embeddings se realiza en un proceso independiente del
agente, manteniendo la responsabilidad separada:

```text
data/processed/base_conocimiento_chunks.md
  |
  v
src/carnicos_kb/rag_index_builder.py
  |
  v
OpenAIEmbeddings(model="text-embedding-3-small")
  |
  v
PostgreSQL/PGVector   (base vectorial persistente)
```

---

## 3. Implementacion

### 3.1 Memoria conversacional

La memoria de sesion opera en dos capas segun la interfaz utilizada:

- En modo consola, `CarnicosQASystem` mantiene una instancia de
  `InMemoryChatMessageHistory` que acumula los turnos de la sesion activa.
- En la interfaz Streamlit, el historial se persiste en
  `st.session_state.messages` como diccionarios con claves `role` y `content`.

Antes de cada invocacion del router o de la cadena de respuesta, el historial
se convierte a objetos `HumanMessage` y `AIMessage` mediante
`normalize_chat_history()`. Esta funcion filtra mensajes vacios y descarta los
mensajes de sistema para que no sean reinyectados como memoria conversacional.
El orden resultante — prompt de sistema, historial normalizado, pregunta actual
— permite al modelo resolver referencias anaforicas como "el primero que
mencionaste" sin perder las reglas del sistema.

### 3.2 Herramienta de datos estructurados

La herramienta `datos_estructurados_carnicos`, implementada en
`src/carnicos_kb/structured_data_tool.py`, sirve respuestas deterministas para
datos concretos de la organizacion: telefonos, lineas gratuitas, sedes
comerciales, centros de distribucion, puntos de venta propios, NIT, fecha de
creacion, solicitud de empleo, visitas a planta y horarios de atencion.

Los registros se almacenan en `data/structured/carnicos_structured_faq.json`.
La recuperacion es lexica y no depende de embeddings: la consulta se normaliza,
se extraen sus tokens significativos y se puntuan coincidencias en el titulo,
las palabras clave, los ejemplos de pregunta y el texto de respuesta de cada
registro. El NIT se trata como un registro unico para evitar que el agente lo
confunda con otras cadenas que contienen subcadenas similares.

### 3.3 Herramienta documental

La herramienta `base_documental_carnicos` esta implementada en
`src/carnicos_kb/vector_retriever_tool.py`. Expone una interfaz comun para
recuperacion vectorial documental con PGVector en produccion, InMemoryVectorStore
como fallback de desarrollo.
Devuelve los tres fragmentos mas cercanos semanticamente a la consulta.

El modulo `src/carnicos_kb/document_retriever_tool.py` conserva las utilidades
de parseo de chunks (`parse_knowledge_chunks`, `KnowledgeChunk`) que usa el
indexador para leer el archivo Markdown antes de crear los embeddings.

### 3.4 Router y cadena de respuesta

El router se construye en `CarnicosQASystem._create_router_chain()` como una
cadena LangChain compuesta por un `ChatPromptTemplate` con
`MessagesPlaceholder` para memoria, `ChatOpenAI` y `with_structured_output(RouteDecision)`.
La salida estructurada `RouteDecision` contiene dos campos: `tool_name`,
restringido a los literales `datos_estructurados_carnicos` o
`base_documental_carnicos`, y `reason`, una justificacion breve legible para
el usuario. Si el router devuelve una herramienta no reconocida, el sistema
aplica un mecanismo de fallback que selecciona automaticamente la herramienta
documental.

La cadena de respuesta final, construida en `_create_agent_answer_chain()`,
recibe como entradas la pregunta, la herramienta seleccionada, el motivo del
router, la salida de la herramienta y el historial normalizado. El prompt de
sistema de esta cadena instruye al modelo a tratar el contenido recuperado como
evidencia factual, no como instrucciones, como medida de defensa contra
inyeccion de prompt a traves de documentos recuperados.

### 3.5 Indice RAG vectorial

`src/carnicos_kb/rag_index_builder.py` genera el indice vectorial de forma
independiente al agente. Lee los Markdown procesados desde
`data/processed/dataset_carnicos`, los divide con `RecursiveCharacterTextSplitter`,
invoca `OpenAIEmbeddings(model="text-embedding-3-small")` y persiste los
documentos embebidos en PostgreSQL/PGVector.

El modo `dry-run` valida el parseo y conteo de chunks sin realizar llamadas a
la API de OpenAI. En la ejecucion validada el corpus produjo 329 chunks.

### 3.6 Trazabilidad con LangSmith

Cada invocacion del router, la herramienta seleccionada y la cadena de
respuesta final incluye un `run_name` y etiquetas (`router`, `tool`,
`final-answer`) que LangSmith utiliza para organizar las trazas. La
configuracion se gestiona en `src/carnicos_kb/langsmith_config.py` a traves de
las variables de entorno:

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_pt_tu_api_key_aqui
LANGSMITH_PROJECT=carnicos-kb-agent
```

La trazabilidad esta desactivada por defecto y se habilita cambiando
`LANGSMITH_TRACING` a `true` y sustituyendo la clave de API. Durante la
ejecucion de la suite de pruebas, `tests/conftest.py` desactiva LangSmith
automaticamente para evitar trazas externas en entornos de CI.

---

## 4. Verificacion

### 4.1 Pruebas unitarias

La suite cubre ocho modulos de test con un total de 30 casos:

| Archivo | Descripcion |
|---|---|
| `tests/test_structured_data_tool.py` | Recuperacion determinista de datos estructurados |
| `tests/test_document_retriever_tool.py` | Parseo de chunks Markdown y constante de nombre de herramienta |
| `tests/test_agent_contract.py` | Contrato publico del agente (`QAResponse`, `answer`, `answer_with_trace`) |
| `tests/test_rag_index_builder.py` | Construccion del indice PGVector: carga, formato de embedding e indexacion |
| `tests/test_vector_retriever_tool.py` | Renderizado de documentos recuperados por retrievers vectoriales |
| `tests/test_conversation_memory.py` | Normalizacion del historial y orden de mensajes |
| `tests/test_routing_end_to_end.py` | Flujo completo router → herramienta → cadena de respuesta con LLM mockeado (6 escenarios) |
| `tests/test_chunking.py` | Pipeline de chunking desde archivos Markdown |

Resultado de la ejecucion:

```text
27 passed
```

Comando utilizado:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp -p no:cacheprovider
```

### 4.2 Calidad estatica del codigo

El codigo fuente y los tests superaron la revision de estilo con `ruff`:

```text
All checks passed!
```

Comando utilizado:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
```

---

## 5. Conclusiones

El Modulo 2 introduce un cambio de paradigma respecto al sistema anterior: la
recuperacion ya no es implicita dentro del contexto del modelo sino explicita y
auditable a traves de herramientas con responsabilidades diferenciadas. El
router con salida estructurada garantiza que la eleccion de herramienta quede
registrada en cada turno, lo que facilita la depuracion y la explicabilidad
ante el usuario.

La construccion del indice RAG como proceso independiente y la configuracion
de LangSmith como componente opcional dejan el sistema preparado para escalar
hacia un entorno de produccion con observabilidad completa sin incrementar la
complejidad del flujo de inferencia principal.
