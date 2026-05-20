# Carnicos KB

Proyecto Python para construir una base de conocimiento en Markdown sobre **Alimentos Carnicos S.A.S.** a partir de scraping web, extraccion de PDFs y segmentacion en chunks para alimentar un LLM.

## Estructura

```text
.
├── data/
│   ├── raw/
│   │   └── pdfs/                          # PDFs fuente
│   ├── processed/
│   │   ├── dataset_carnicos/              # Markdown extraido
│   │   └── base_conocimiento_chunks.md
│   └── structured/
│       └── carnicos_structured_faq.json   # Datos concretos (herramienta estructurada)
├── docs/
│   ├── Requisitos.txt
│   ├── definicion_alcance.md
│   ├── informe_modulo_1.md
│   ├── informe_modulo_2.md                # Informe de sustentacion Modulo 2
│   ├── memoria_conversacional_modulo_2.md
│   ├── rag_vectorial.md
│   ├── GUIA_RAPIDA.md
│   └── QA_SYSTEM.md
├── src/
│   └── carnicos_kb/
│       ├── chunking.py                    # Limpieza y chunking semantico
│       ├── vector_retriever_tool.py       # Recuperador documental vectorial
│       ├── document_retriever_tool.py     # Utilidades de parseo de chunks Markdown
│       ├── knowledge_loader.py            # Carga de la base de conocimiento
│       ├── langsmith_config.py            # Estado de configuracion LangSmith
│       ├── paths.py                       # Rutas por defecto del proyecto
│       ├── pdf_extractor.py               # Extraccion PDF con Docling
│       ├── pdf_text_extractor.py          # Extraccion PDF con PyMuPDF
│       ├── qa_system.py                   # Agente Q&A con router y memoria
│       ├── rag_index_builder.py           # Construccion del indice RAG vectorial
│       ├── scraper.py                     # Scraping web desde sitemap
│       ├── streamlit_app.py               # Interfaz web Streamlit
│       ├── streamlit_runner.py            # Punto de entrada Streamlit
│       ├── structured_data_tool.py        # Herramienta LangChain estructurada
│       ├── text_matching.py               # Utilidades de busqueda lexica
│       └── validate_module1.py            # Validacion base Modulo 1
├── tests/
│   ├── conftest.py                        # Desactiva LangSmith en pruebas
│   ├── test_agent_contract.py             # Contrato del router y QAResponse
│   ├── test_chunking.py                   # Limpieza y segmentacion
│   ├── test_conversation_memory.py        # Memoria conversacional
│   ├── test_document_retriever_tool.py    # Herramienta documental
│   ├── test_rag_index_builder.py          # Indice RAG vectorial
│   ├── test_routing_end_to_end.py         # Enrutamiento end-to-end con mock
│   └── test_structured_data_tool.py       # Herramienta estructurada
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Requisitos

- Python 3.10+
- `uv`
- `make`
- Opcional en Windows: Chocolatey para instalar herramientas base

Instalacion con Chocolatey:

```powershell
choco install -y python uv make git
```

Tambien puedes usar el target:

```powershell
make bootstrap-choco
```

## Instalacion

Con `uv`:

```powershell
make sync
```

Con `pip`, si no quieres usar `uv`:

```powershell
make install-pip
```

Las dependencias se declaran en `pyproject.toml`. `uv.lock` conserva las versiones resueltas para instalaciones reproducibles.

## Comandos principales

```powershell
make scrape
make pdf
make pdf-fast
make chunk
make rag-index
make test
make lint
make app
```

Equivalentes directos con `uv`:

```powershell
uv run carnicos-scrape --output-dir data/processed/dataset_carnicos
uv run carnicos-pdf --input-dir data/raw/pdfs --output-dir data/processed/dataset_carnicos
uv run carnicos-chunk --input-dir data/processed/dataset_carnicos --output data/processed/base_conocimiento_chunks.md
uv run carnicos-build-rag --dataset-dir data/processed/dataset_carnicos
uv run carnicos-app --server.address localhost --server.port 8501
uv run --extra dev pytest --basetemp .pytest_tmp
```

## Flujo de trabajo

1. Guardar PDFs fuente en `data/raw/pdfs/`.
2. Ejecutar `make scrape` para extraer paginas del sitio web.
3. Ejecutar `make pdf` para extraer PDFs con Docling, o `make pdf-fast` para extraccion rapida con PyMuPDF.
4. Ejecutar `make chunk` para generar `data/processed/base_conocimiento_chunks.md`.
5. Ejecutar `make rag-index` para construir el indice vectorial en PostgreSQL/PGVector.
6. Ejecutar `make app` para abrir la interfaz Streamlit del asistente Q&A.

## Variables de entorno

El proyecto puede leer `.env`. Usa `.env.example` como plantilla:

```text
SITEMAP_URL=https://alimentoscarnicos.com.co/wp-sitemap-posts-page-1.xml
OUTPUT_DIR=data/processed/dataset_carnicos
REQUEST_TIMEOUT=10
REQUEST_DELAY=2
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
OPENAI_API_KEY=sk_test_tu_api_key_aqui
OPENAI_MODEL=gpt-5.4-nano
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=1500
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CARNICOS_KNOWLEDGE_PATH=data/processed/base_conocimiento_chunks.md
DATABASE_URL=postgresql://user:password@localhost:5432/carnicos_kb
PG_COLLECTION_NAME=carnicos_rag
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_pt_tu_api_key_aqui
LANGSMITH_PROJECT=carnicos-kb-agent
```

## Modulos

- `carnicos_kb.scraper`: scraping web desde sitemap XML con `requests`, `BeautifulSoup` y `trafilatura`.
- `carnicos_kb.pdf_extractor`: conversion estructurada de PDFs a Markdown con Docling, por lotes.
- `carnicos_kb.pdf_text_extractor`: conversion rapida de PDFs a Markdown con PyMuPDF.
- `carnicos_kb.chunking`: limpieza conservadora y chunking semantico de Markdown.
- `carnicos_kb.rag_index_builder`: construccion del indice RAG vectorial en PostgreSQL/PGVector con `text-embedding-3-small`.
- `carnicos_kb.vector_retriever_tool`: herramientas LangChain para recuperacion semantica con PGVector e InMemoryVectorStore. Incluye `DocumentalQueryInput` (schema Pydantic con `args_schema` para validar entradas del LLM) y manejo de errores resiliente en `consultar_base_documental` (fallos de red o embeddings devuelven un mensaje guia en lugar de propagar la excepcion).
- `carnicos_kb.document_retriever_tool`: utilidades para parsear chunks Markdown (IDs, titulo, fuente, texto).
- `carnicos_kb.structured_data_tool`: herramienta LangChain para datos concretos en JSON.
- `carnicos_kb.qa_system`: agente Q&A con memoria, router LangChain y trazas LangSmith. El agente emite respuestas validadas via `AgentResponseSchema` (`answer`, `tool_was_called`, `confidence`); `QAResponse` expone el campo `confidence` para indicar el nivel de evidencia documental de cada respuesta.
- `carnicos_kb.streamlit_app`: interfaz web Streamlit con chat, ruta del agente, alcance, guia rapida y estado de la base.

## Pruebas

```powershell
make test
```

Las pruebas validan limpieza, division por encabezados, construccion de chunks,
memoria conversacional, herramienta estructurada, recuperador documental,
contrato del router, indice RAG vectorial y enrutamiento end-to-end del agente
con LLM mockeado.

## Modulo 2

La documentacion de sustentacion esta en `docs/informe_modulo_2.md`. Incluye el
flujo Usuario -> Router -> Herramienta seleccionada -> LLM -> Respuesta, ademas
de un guion de demo con preguntas para memoria, datos estructurados y base
documental.

La construccion del indice RAG vectorial con embeddings se documenta en
`docs/rag_vectorial.md`.

## Modulo 3 — Structured Output y Gestion de Errores

Mejoras de robustez y observabilidad sobre el agente Q&A:

| Cambio | Archivo | Efecto |
|---|---|---|
| `DocumentalQueryInput` + `args_schema` | `vector_retriever_tool.py` | El LLM recibe un JSON Schema estricto al invocar la herramienta RAG; Pydantic valida `min_length=3` y `max_length=300` antes de ejecutar la busqueda |
| `try/except` en `consultar_base_documental` | `vector_retriever_tool.py` | Fallos de red, timeout o error de embeddings devuelven `[HERRAMIENTA_ERROR]` legible; el agente responde cortesmente en lugar de propagar la excepcion |
| `AgentResponseSchema` + `response_format` | `qa_system.py` | El agente emite JSON validado con `answer`, `tool_was_called` y `confidence` en lugar de texto libre |
| Campo `confidence` en `QAResponse` | `qa_system.py` | Cada respuesta incluye el nivel de confianza (`high` / `medium` / `low` / `unknown`) trazable por la UI y los tests |
