from __future__ import annotations

import argparse
import gc
from pathlib import Path

from carnicos_kb.paths import DEFAULT_DATASET_DIR, DEFAULT_RAW_PDF_DIR


DEFAULT_BATCH_SIZE = 5


def get_page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def convert_pdf_in_batches(converter, pdf_path: Path, batch_size: int) -> str:
    total_pages = get_page_count(pdf_path)
    if total_pages == 0:
        return ""

    markdown_parts: list[str] = []

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        print(f"   Procesando paginas {start}-{end} de {total_pages}...")

        result = None
        try:
            result = converter.convert(pdf_path, page_range=(start, end))
            markdown_parts.append(result.document.export_to_markdown())
        except Exception as batch_error:
            print(f"   Fallo en paginas {start}-{end}: {batch_error}")
            markdown_parts.append(
                f"\n<!-- ERROR extracting pages {start}-{end}: {batch_error} -->\n"
            )
        finally:
            del result
            gc.collect()

    return "\n\n".join(part for part in markdown_parts if part)


def convert_pdfs(input_dir: Path, output_dir: Path, batch_size: int) -> int:
    from docling.document_converter import DocumentConverter

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"), key=lambda path: path.name.lower())

    if not pdf_files:
        print(f"No hay archivos PDF en: {input_dir}")
        return 0

    converter = DocumentConverter()
    converted_count = 0

    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.md"
        print(f"PDF: {pdf_path}")

        try:
            markdown_output = convert_pdf_in_batches(converter, pdf_path, batch_size)
            output_path.write_text(
                f"--- SOURCE: {pdf_path.name} (Extracted with Docling) ---\n\n"
                f"{markdown_output}",
                encoding="utf-8",
            )
            converted_count += 1
            print(f"Guardado: {output_path}")
        except Exception as error:
            print(f"Error en {pdf_path}: {error}")
        finally:
            gc.collect()

    return converted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraccion estructurada de PDFs a Markdown.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RAW_PDF_DIR,
        help="Carpeta con PDFs fuente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Carpeta de salida para los Markdown.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Cantidad de paginas por lote.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size debe ser mayor que cero")

    converted_count = convert_pdfs(args.input_dir, args.output_dir, args.batch_size)
    print(f"PDFs convertidos: {converted_count}")


if __name__ == "__main__":
    main()

