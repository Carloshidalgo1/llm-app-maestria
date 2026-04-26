from __future__ import annotations

import argparse
from pathlib import Path

from carnicos_kb.paths import DEFAULT_DATASET_DIR, DEFAULT_RAW_PDF_DIR


def convert_pdfs_plain_text(input_dir: Path, output_dir: Path) -> int:
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"), key=lambda path: path.name.lower())

    if not pdf_files:
        print(f"No hay archivos PDF en: {input_dir}")
        return 0

    converted_count = 0
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.md"

        try:
            doc = fitz.open(pdf_path)
            markdown_parts = [f"--- SOURCE PDF: {pdf_path.name} ---\n"]

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                markdown_parts.append(f"## Pagina {page_num + 1}\n\n{page.get_text()}\n")

            output_path.write_text("\n".join(markdown_parts), encoding="utf-8")
            converted_count += 1
            print(f"Guardado: {output_path}")
        except Exception as error:
            print(f"Error procesando {pdf_path}: {error}")

    return converted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraccion rapida de texto plano de PDFs.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted_count = convert_pdfs_plain_text(args.input_dir, args.output_dir)
    print(f"PDFs convertidos: {converted_count}")


if __name__ == "__main__":
    main()

