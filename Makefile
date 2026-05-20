SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.DEFAULT_GOAL := help

PYTHON ?= python
UV ?= uv
UV_CACHE_DIR ?= .uv_cache
PRE_COMMIT_HOME ?= .pre_commit_cache

export UV_CACHE_DIR
export PRE_COMMIT_HOME

unexport VIRTUAL_ENV

PDF_DIR ?= data/raw/pdfs
DATASET_DIR ?= data/processed/dataset_carnicos
CHUNKS_FILE ?= data/processed/base_conocimiento_chunks.md
PG_COLLECTION ?= carnicos_rag
EMBEDDING_MODEL ?= text-embedding-3-small
MAX_CHARS ?= 3200
BATCH_SIZE ?= 5
STREAMLIT_HOST ?= localhost
STREAMLIT_PORT ?= 8501

.PHONY: help bootstrap-choco sync install install-pip precommit-install precommit test test-pip lint scrape pdf pdf-fast chunk rag-index rag-index-dry-run pipeline qa validate-mod1 app streamlit clean clean-data

help:
	$(info Comandos disponibles:)
	$(info   make bootstrap-choco - Instalar python, uv, make y git con Chocolatey)
	$(info   make sync            - Crear/actualizar entorno con uv)
	$(info   make install         - Alias de make sync)
	$(info   make install-pip     - Instalar con pip en .venv sin uv)
	$(info   make precommit-install - Instalar hooks de pre-commit)
	$(info   make precommit       - Ejecutar pre-commit en todos los archivos)
	$(info   make test            - Ejecutar pytest con uv)
	$(info   make lint            - Ejecutar ruff)
	$(info   make scrape          - Ejecutar scraping web)
	$(info   make pdf             - Extraer PDFs con Docling)
	$(info   make pdf-fast        - Extraer PDFs con PyMuPDF)
	$(info   make chunk           - Generar base de conocimiento segmentada)
	$(info   make rag-index       - Construir indice RAG vectorial en PostgreSQL/PGVector)
	$(info   make rag-index-dry-run - Validar chunks sin llamar a OpenAI)
	$(info   make pipeline        - Ejecutar scraping, PDF y chunking)
	$(info   make qa              - Iniciar sistema Q&A interactivo)
	$(info   make validate-mod1   - Validar base Q&A del Modulo 1)
	$(info   make app             - Iniciar interfaz web Streamlit)
	$(info   make clean           - Limpiar caches y chunks generados)
	@Write-Host "" -NoNewline

bootstrap-choco:
	choco install -y python uv make git

sync:
	$(UV) sync --extra dev

install: sync

install-pip:
	$(PYTHON) -m venv .venv
	.\.venv\Scripts\python.exe -m pip install --upgrade pip
	.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

precommit-install:
	$(UV) run --extra dev pre-commit install

precommit:
	$(UV) run --extra dev pre-commit run --all-files

test:
	$(UV) run --extra dev pytest --basetemp .pytest_tmp

test-pip:
	.\.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp

lint:
	$(UV) run --extra dev ruff check src tests

scrape:
	$(UV) run carnicos-scrape --output-dir "$(DATASET_DIR)"

pdf:
	$(UV) run carnicos-pdf --input-dir "$(PDF_DIR)" --output-dir "$(DATASET_DIR)" --batch-size $(BATCH_SIZE)

pdf-fast:
	$(UV) run carnicos-pdf-fast --input-dir "$(PDF_DIR)" --output-dir "$(DATASET_DIR)"

chunk:
	$(UV) run carnicos-chunk --input-dir "$(DATASET_DIR)" --output "$(CHUNKS_FILE)" --max-chars $(MAX_CHARS)

rag-index:
	$(UV) run carnicos-build-rag --dataset-dir "$(DATASET_DIR)" --embedding-model "$(EMBEDDING_MODEL)" --collection "$(PG_COLLECTION)"

rag-index-dry-run:
	$(UV) run carnicos-build-rag --dataset-dir "$(DATASET_DIR)" --embedding-model "$(EMBEDDING_MODEL)" --dry-run

pipeline: scrape pdf chunk

qa:
	$(UV) run carnicos-qa

validate-mod1:
	$(UV) run carnicos-validate-mod1

app:
	$(UV) run carnicos-app --server.address "$(STREAMLIT_HOST)" --server.port $(STREAMLIT_PORT)

streamlit: app

clean:
	if (Test-Path "$(CHUNKS_FILE)") { Remove-Item -LiteralPath "$(CHUNKS_FILE)" -Force }
	if (Test-Path ".pytest_cache") { Remove-Item -LiteralPath ".pytest_cache" -Recurse -Force }
	if (Test-Path ".pytest_tmp") { Remove-Item -LiteralPath ".pytest_tmp" -Recurse -Force }
	if (Test-Path ".ruff_cache") { Remove-Item -LiteralPath ".ruff_cache" -Recurse -Force }
	Get-ChildItem -Path src,tests -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

clean-data:
	if (Test-Path "$(DATASET_DIR)") { Remove-Item -LiteralPath "$(DATASET_DIR)" -Recurse -Force }
	if (Test-Path "$(CHUNKS_FILE)") { Remove-Item -LiteralPath "$(CHUNKS_FILE)" -Force }
