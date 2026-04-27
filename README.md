# Carnicos KB

Proyecto Python para construir una base de conocimiento en Markdown sobre **Alimentos Carnicos S.A.S.** a partir de scraping web, extraccion de PDFs y segmentacion en chunks para alimentar un LLM.

## Estructura

```text
.
├── data/
│   ├── raw/
│   │   └── pdfs/                  # PDFs fuente
│   └── processed/
│       ├── dataset_carnicos/       # Markdown extraido
│       └── base_conocimiento_chunks.md
├── docs/
│   ├── Requisitos.txt
│   └── definicion_alcance.md
├── src/
│   └── carnicos_kb/
│       ├── chunking.py
│       ├── pdf_extractor.py
│       ├── pdf_text_extractor.py
│       └── scraper.py
├── tests/
├── Makefile
├── pyproject.toml
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
make test
make lint
make app
```

Equivalentes directos con `uv`:

```powershell
uv run carnicos-scrape --output-dir data/processed/dataset_carnicos
uv run carnicos-pdf --input-dir data/raw/pdfs --output-dir data/processed/dataset_carnicos
uv run carnicos-chunk --input-dir data/processed/dataset_carnicos --output data/processed/base_conocimiento_chunks.md
uv run carnicos-app --server.address localhost --server.port 8501
uv run --extra dev pytest --basetemp .pytest_tmp
```

## Flujo de trabajo

1. Guardar PDFs fuente en `data/raw/pdfs/`.
2. Ejecutar `make scrape` para extraer paginas del sitio web.
3. Ejecutar `make pdf` para extraer PDFs con Docling, o `make pdf-fast` para extraccion rapida con PyMuPDF.
4. Ejecutar `make chunk` para generar `data/processed/base_conocimiento_chunks.md`.
5. Usar `data/processed/base_conocimiento_chunks.md` como contexto del prompt de sistema del LLM.
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
CARNICOS_KNOWLEDGE_PATH=data/processed/base_conocimiento_chunks.md
```

Si tu `.env` anterior apunta a `dataset_carnicos`, actualizalo a `data/processed/dataset_carnicos` para seguir la nueva estructura.

## Modulos

- `carnicos_kb.scraper`: scraping web desde sitemap XML con `requests`, `BeautifulSoup` y `trafilatura`.
- `carnicos_kb.pdf_extractor`: conversion estructurada de PDFs a Markdown con Docling, por lotes.
- `carnicos_kb.pdf_text_extractor`: conversion rapida de PDFs a Markdown con PyMuPDF.
- `carnicos_kb.chunking`: limpieza conservadora y chunking semantico de Markdown.
- `carnicos_kb.qa_system`: asistente Q&A sobre la base de conocimiento.
- `carnicos_kb.streamlit_app`: interfaz web Streamlit con chat, alcance, guia rapida y estado de la base.

## Pruebas

```powershell
make test
```

Las pruebas unitarias actuales validan limpieza, division por encabezados, construccion de chunks y renderizado del Markdown final.
