# Webscraping + Extracción de PDFs (corpus cárnicos)

Pipeline para construir un corpus en Markdown a partir de páginas web y PDFs del sector cárnico/ganadero. Los documentos resultantes se guardan en `dataset_carnicos/`.

## Instalación y Ejecución

Este proyecto utiliza un entorno virtual Python y un Makefile para facilitar la ejecución.

### Requisitos previos
- Python 3.8+
- Make (en Windows, instalar con Chocolatey o similar)

### Configuración inicial
1. Clona o descarga el proyecto.
2. Configura el entorno virtual y instala dependencias:
   ```
   make install
   ```
   Esto crea el entorno virtual en `venv/` e instala las dependencias de `requirements.txt`.

### Variables de entorno
El proyecto utiliza un archivo `.env` para configurar variables como URLs, timeouts y directorios de salida. Edita `.env` según tus necesidades:

- `BASE_URL`: URL base para scraping individual.
- `SITEMAP_URL`: URL del sitemap para bulk scraping.
- `OUTPUT_DIR`: Directorio donde guardar los archivos generados.
- `REQUEST_TIMEOUT`: Timeout en segundos para requests.
- `USER_AGENT`: User-Agent para las requests.

### Comandos disponibles
- `make run`: Ejecuta el bulk scraper (scraping masivo de URLs desde un sitemap).
- `make pdf`: Ejecuta el extractor de PDFs (`pdf_pro_extractor.py`).
- `make clean`: Limpia archivos generados en `dataset_carnicos/`.

## Scripts principales

- `bulk_scraper.py` — scraping web masivo desde un sitemap XML.
- `pdf_to_md.py` — extracción rápida de texto plano con PyMuPDF (sin layout ni tablas).
- `pdf_pro_extractor.py` — extracción estructurada con [Docling](https://github.com/DS4SD/docling) (layout, tablas, OCR) **con procesamiento por lotes**.

## Extracción por lotes con Docling (`pdf_pro_extractor.py`)

### Por qué
Al correr Docling sobre PDFs largos en CPU, el pipeline de preprocesado (RapidOCR + análisis de layout) carga todas las páginas en memoria a la vez. En documentos de 40+ páginas esto provocaba errores del tipo:

```
Stage preprocess failed for run 1, pages [34]: std::bad_alloc
Stage preprocess failed for run 1, pages [35]: std::bad_alloc
...
```

Con la consecuencia de perder decenas de páginas por documento.

### Estrategia
En lugar de convertir el PDF completo en una sola llamada, el script:

1. **Cuenta las páginas** del PDF con `pypdfium2` (`get_page_count`).
2. **Itera en ventanas de `BATCH_SIZE` páginas** (por defecto 5) e invoca a Docling con el parámetro `page_range=(start, end)`, de modo que cada llamada reserva memoria únicamente para ese subconjunto.
3. **Exporta cada lote a Markdown** y lo acumula en una lista.
4. **Libera el resultado intermedio** (`del result` + `gc.collect()`) antes de pedir el siguiente lote, evitando que los tensores de OCR, imágenes renderizadas y estructuras de layout se acumulen entre iteraciones.
5. **Aísla fallos por lote**: si un rango concreto falla, se inserta un marcador `<!-- ERROR extracting pages X-Y: ... -->` en el Markdown y se continúa con el siguiente, en vez de abortar todo el documento.
6. **Concatena** los fragmentos y los persiste en `dataset_carnicos/<nombre>.md`.

### Configuración
`BATCH_SIZE` está definido como constante al inicio del archivo:

```python path=pdf_pro_extractor.py start=11
BATCH_SIZE = 5
```

Recomendaciones:

- **Máquinas con poca RAM o PDFs con muchas imágenes/OCR pesado:** baja a `3` o incluso `2`.
- **Máquinas con RAM holgada y PDFs “limpios” (texto vectorial):** puedes subir a `10` para reducir el overhead de reinicializar el pipeline.
- El `DocumentConverter` se instancia **una sola vez** y se reutiliza entre lotes y entre documentos, por lo que los modelos de Docling/RapidOCR se cargan una única vez.

### Resultado de la validación

Ejecutado sobre el corpus de prueba (PDF de 90 páginas que previamente fallaba desde la página 34):

- **0** errores `std::bad_alloc` (antes: 57 páginas perdidas en ese único documento).
- Pico de memoria trazado por Python estable en ~150–180 MiB entre lotes.
- Markdown final con tablas y encabezados reconstruidos correctamente (~99 headings y ~64 tablas en el documento de referencia).
- `~3.5 s/página` en CPU.

## Uso

```powershell path=null start=null
# 1. Dejar los PDFs fuente en ./pdf/
# 2. Ejecutar el extractor
python pdf_pro_extractor.py
# 3. Revisar los .md generados en ./dataset_carnicos/
```

Si aparece `std::bad_alloc` con algún PDF especialmente pesado, reduce `BATCH_SIZE` en `pdf_pro_extractor.py` y vuelve a ejecutar.

## Dependencias clave

- `docling` (>= 2.90)
- `pypdfium2` (viene como dependencia transitiva de Docling; se usa explícitamente para contar páginas)
- `rapidocr` (dependencia transitiva de Docling para OCR)
