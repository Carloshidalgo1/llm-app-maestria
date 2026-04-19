import gc
import os

import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter


# Tamaño del lote de páginas. Valores pequeños reducen la presión de memoria
# (std::bad_alloc en RapidOCR / pipeline de preprocesado) a costa de un
# pequeño overhead por cada invocación de Docling.
BATCH_SIZE = 5


def get_page_count(pdf_path: str) -> int:
    """Devuelve el número de páginas de un PDF usando pypdfium2."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def convert_pdf_in_batches(
    converter: DocumentConverter,
    pdf_path: str,
    batch_size: int = BATCH_SIZE,
) -> str:
    """Convierte un PDF a Markdown procesando sus páginas por lotes.

    Se invoca a Docling con ``page_range`` acotado para que las estructuras
    intermedias (imágenes, OCR, layout) se liberen entre lotes y se evite
    el agotamiento de memoria (``std::bad_alloc``) observado en PDFs grandes.
    """
    total_pages = get_page_count(pdf_path)
    if total_pages == 0:
        return ""

    markdown_parts: list[str] = []

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        print(f"   \u21b3 Procesando p\u00e1ginas {start}-{end} de {total_pages}...")

        result = None
        try:
            result = converter.convert(pdf_path, page_range=(start, end))
            markdown_parts.append(result.document.export_to_markdown())
        except Exception as batch_error:
            print(f"   \u26a0\ufe0f Fallo en p\u00e1ginas {start}-{end}: {batch_error}")
            markdown_parts.append(
                f"\n<!-- ERROR extracting pages {start}-{end}: {batch_error} -->\n"
            )
        finally:
            # Liberar memoria del lote antes de pedir el siguiente.
            del result
            gc.collect()

    return "\n\n".join(part for part in markdown_parts if part)


def convert_pdf_to_md_pro() -> None:
    input_folder = "pdf"
    output_folder = "dataset_carnicos"

    os.makedirs(output_folder, exist_ok=True)

    converter = DocumentConverter()

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"\u26a0\ufe0f No hay archivos en '{input_folder}'")
        return

    print("\U0001f9e0 Iniciando extracci\u00f3n inteligente con Docling (procesamiento por lotes)...")

    for pdf in pdf_files:
        pdf_path = os.path.join(input_folder, pdf)
        output_name = os.path.splitext(pdf)[0] + ".md"
        output_path = os.path.join(output_folder, output_name)

        print(f"\U0001f4c4 {pdf}")

        try:
            markdown_output = convert_pdf_in_batches(converter, pdf_path, BATCH_SIZE)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"--- SOURCE: {pdf} (Extracted with AI) ---\n\n")
                f.write(markdown_output)

            print(f"\u2705 \u00a1Tabla y estructura preservada!: {pdf}")

        except Exception as e:
            print(f"\u274c Error en {pdf}: {e}")
        finally:
            gc.collect()


if __name__ == "__main__":
    convert_pdf_to_md_pro()
