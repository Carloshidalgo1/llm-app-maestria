.PHONY: help install run clean

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/Scripts/python
PIP = $(VENV_DIR)/Scripts/pip
ACTIVATE = $(VENV_DIR)/Scripts/activate

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias en el entorno virtual"
	@echo "  make run        - Ejecutar el bulk scraper (scraping masivo)"
	@echo "  make pdf        - Ejecutar el extractor de PDFs"
	@echo "  make clean      - Limpiar archivos generados"
	@echo "  make venv       - Crear entorno virtual"

venv:
	python -m venv $(VENV_DIR)

install: venv
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) bulk_scraper.py

pdf: install
	$(PYTHON) pdf_pro_extractor.py

clean:
	rm -rf dataset_carnicos/*.md
	rm -rf __pycache__