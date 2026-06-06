# Carnicos KB

> Asistente conversacional para **Alimentos Cárnicos S.A.S.** (Grupo Nutresa) desplegado en **WhatsApp**. Responde preguntas de primer contacto usando una base de conocimiento construida con web scraping y documentos públicos de la empresa, indexada con embeddings vectoriales (RAG) y enrutada mediante un agente LangGraph ReAct.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange)
![WhatsApp](https://img.shields.io/badge/Canal-WhatsApp-25D366)

---

## Descripción general

El sistema recibe mensajes de WhatsApp a través de **Twilio**, los enruta con **N8N** hacia una **API FastAPI** que ejecuta un agente **LangGraph ReAct** con acceso a una base vectorial. Cada respuesta incluye un campo `confidence` que determina si el agente responde directamente al cliente o escala la consulta a un asesor humano.

| Componente | Tecnología |
|---|---|
| Canal de mensajería | WhatsApp vía Twilio |
| Orquestación del flujo | N8N |
| Túnel HTTPS local | ngrok |
| API REST | FastAPI |
| Agente de razonamiento | LangGraph ReAct + GPT-4o-mini |
| Base vectorial | PGVector (prod) / InMemoryVectorStore (dev) |
| Memoria conversacional | PostgresSaver (prod) / InMemorySaver (dev) |
| Supervisión humana | HITL Middleware (opcional) |

---

## Arquitectura

```
 Usuario
 (WhatsApp)
     |
     | mensaje entrante
     v
 +--------+     +------------+     +--------+     +---------+
 | Twilio | --> |    N8N     | --> | ngrok  | --> | FastAPI |
 +--------+     | (6 nodos)  |     +--------+     +---------+
                +------------+                         |
                      ^                                | POST /chat
                      |                                v
                      |                    +---------------------+
                      |                    |  Agente LangGraph   |
                      |                    |  ReAct (GPT-4o-mini)|
                      |                    +---------------------+
                      |                                |
                      |                    +-----------+-----------+
                      |                    |                       |
                      |              InMemoryVectorStore      PostgreSQL
                      |              / PGVector (prod)        (historial)
                      |
               IF confidence
               "low" + tool_was_called
                  /           \
              TRUE             FALSE
                |                 |
        Escala asesor     Responde cliente
        + notifica
          cliente
```

---

## Flujo N8N

```
Webhook --> Edit Fields --> HTTP Request --> IF --> Mensaje al asesor --> Mensaje al cliente
                                              |
                                              --> Respuesta del agente
```

| Nodo | Función |
|---|---|
| **Webhook** | Recibe el mensaje entrante de Twilio |
| **Edit Fields** | Extrae y renombra `message` y `phone_number` |
| **HTTP Request** | Llama a `POST /chat` en la API FastAPI |
| **IF** | Evalúa `confidence == "low"` AND `tool_was_called == true` |
| **Mensaje al asesor** | Alerta al asesor con contexto de la consulta (rama TRUE) |
| **Mensaje al cliente** | Notifica al usuario que será atendido (rama TRUE) |
| **Respuesta del agente** | Envía la respuesta directamente al cliente (rama FALSE) |

> Ver [`docs/definicion_alcance.md`](docs/definicion_alcance.md) para el conjunto de preguntas de validación organizadas por grupo (alta confianza / baja confianza / fuera de dominio).

---

## Requisitos

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) — gestor de dependencias
- `make`
- Cuenta Twilio con número WhatsApp habilitado
- Instancia N8N (local o en la nube)
- ngrok instalado

**Instalación de herramientas en Windows (Chocolatey):**

```powershell
choco install -y python uv make git ngrok
```

---

## Instalación

```powershell
make sync
```

---

## Puesta en marcha

### 1. Configurar variables de entorno

```powershell
copy .env.example .env
# Editar .env con los valores reales
```

### 2. Construir la base de conocimiento (primera vez)

```powershell
make scrape       # Extrae páginas del sitio web
make pdf-fast     # Convierte PDFs a Markdown
make chunk        # Genera chunks semánticos
make rag-index    # Construye el índice vectorial en PGVector
```

### 3. Levantar el sistema

```powershell
# Terminal 1 — API del agente
make api

# Terminal 2 — túnel HTTPS público
make ngrok
```

ngrok imprime la URL pública. Configurar esa URL en el nodo **HTTP Request** de N8N.

**Header obligatorio en N8N** (plan gratuito de ngrok):

```
ngrok-skip-browser-warning: 1
```

---

## Comandos disponibles

| Comando | Descripción |
|---|---|
| `make scrape` | Extrae páginas del sitio web de la empresa |
| `make pdf-fast` | Convierte PDFs a Markdown con PyMuPDF |
| `make chunk` | Genera chunks semánticos del corpus |
| `make rag-index` | Construye el índice vectorial en PGVector |
| `make api` | Levanta la API FastAPI en `localhost:8000` |
| `make ngrok` | Abre túnel HTTPS público hacia la API |
| `make app` | Abre la interfaz Streamlit (panel HITL) |
| `make test` | Ejecuta la suite de pruebas |
| `make lint` | Revisa estilo con ruff |

---

## API REST

### `POST /chat`

Envía un mensaje al agente y obtiene la respuesta.

```json
// Request
{
  "message": "¿Qué marcas tiene Alimentos Cárnicos?",
  "phone_number": "573001234567"
}

// Response
{
  "answer": "...",
  "confidence": "high",
  "tool_was_called": true,
  "pending_approval": false,
  "thread_id": "573001234567"
}
```

### `POST /chat/resume`

Reanuda un flujo pausado por el middleware HITL.

```json
// Request
{
  "thread_id": "573001234567",
  "decision": "approve",
  "edited_query": null
}
```

### `GET /health`

```json
{ "status": "ok", "agent_ready": true }
```

**Campo `confidence`:**

| Valor | Significado | Acción en N8N |
|---|---|---|
| `"high"` | Evidencia documental directa o interacción social | Responde al cliente |
| `"medium"` | Inferencia razonable | Responde al cliente |
| `"low"` | Sin evidencia suficiente (pregunta factual) | Escala al asesor |

---

## Variables de entorno

```bash
# LLM
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=openai:gpt-4o-mini
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=1500
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Base vectorial — opcional, usa InMemory si no se configura
DATABASE_URL=postgresql://user:pass@localhost:5432/carnicos_kb
PG_COLLECTION_NAME=carnicos_rag
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=200

# Control humano en el loop
HITL_ENABLED=false

# API
API_HOST=0.0.0.0
API_PORT=8000

# Trazabilidad — opcional
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=carnicos-kb-agent

# Scraping
SITEMAP_URL=https://alimentoscarnicos.com.co/wp-sitemap-posts-page-1.xml
REQUEST_DELAY=2
```

---

## Pruebas

```powershell
make test
```

La suite cubre: chunking semántico, memoria conversacional, herramienta estructurada, recuperador documental, contrato del agente, construcción del índice RAG y enrutamiento end-to-end con LLM mockeado.

---

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/pdfs/                              # PDFs fuente
│   ├── processed/
│   │   ├── dataset_carnicos/                  # Markdown extraído por scraper
│   │   └── base_conocimiento_chunks.md        # Corpus consolidado con chunks
│   └── structured/
│       └── carnicos_structured_faq.json       # FAQ estructurado
├── docs/
│   ├── informe_tecnico_final.md               # Informe unificado Módulos 1-2-3
│   ├── definicion_alcance.md                  # Alcance y preguntas de validación N8N
│   ├── GUIA_RAPIDA.md                         # Inicio rápido y referencia de comandos
│   ├── hitl_middleware.md                     # Documentación HITL
│   ├── memoria_conversacional_modulo_2.md
│   ├── rag_vectorial.md
│   ├── informe_modulo_1.md
│   └── informe_modulo_2.md
├── src/
│   └── carnicos_kb/
│       ├── api.py                             # FastAPI: /chat, /chat/resume, /health
│       ├── qa_system.py                       # Agente LangGraph ReAct + HITL
│       ├── vector_retriever_tool.py           # Herramienta RAG (PGVector / InMemory)
│       ├── document_retriever_tool.py         # Parseo de chunks Markdown
│       ├── streamlit_app.py                   # Interfaz web con panel HITL
│       ├── streamlit_runner.py                # Punto de entrada Streamlit
│       ├── structured_data_tool.py            # Herramienta LangChain para datos JSON
│       ├── rag_index_builder.py               # Construcción del índice PGVector
│       ├── chunking.py                        # Limpieza y chunking semántico
│       ├── knowledge_loader.py                # Carga de la base de conocimiento
│       ├── scraper.py                         # Scraping web desde sitemap
│       ├── pdf_extractor.py                   # Extracción PDF con Docling
│       ├── pdf_text_extractor.py              # Extracción PDF con PyMuPDF
│       ├── langsmith_config.py                # Configuración LangSmith
│       ├── paths.py                           # Rutas por defecto del proyecto
│       └── text_matching.py                   # Utilidades de búsqueda léxica
├── tests/
│   ├── conftest.py
│   ├── test_agent_contract.py
│   ├── test_chunking.py
│   ├── test_conversation_memory.py
│   ├── test_document_retriever_tool.py
│   ├── test_rag_index_builder.py
│   ├── test_routing_end_to_end.py
│   └── test_structured_data_tool.py
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/informe_tecnico_final.md`](docs/informe_tecnico_final.md) | Informe unificado Módulos 1-2-3, arquitectura completa |
| [`docs/definicion_alcance.md`](docs/definicion_alcance.md) | Alcance del sistema y preguntas de validación para N8N |
| [`docs/GUIA_RAPIDA.md`](docs/GUIA_RAPIDA.md) | Inicio rápido, endpoints y referencia de comandos |
| [`docs/hitl_middleware.md`](docs/hitl_middleware.md) | Documentación técnica del middleware HITL |
| [`docs/rag_vectorial.md`](docs/rag_vectorial.md) | Construcción del índice RAG vectorial |
| [`docs/informe_modulo_2.md`](docs/informe_modulo_2.md) | Informe de sustentación Módulo 2 |
