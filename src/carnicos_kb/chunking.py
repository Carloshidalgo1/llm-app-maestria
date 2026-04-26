from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from carnicos_kb.paths import DEFAULT_CHUNKS_FILE, DEFAULT_DATASET_DIR


DEFAULT_MAX_CHARS = 3200

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CITE_RE = re.compile(r"\[cite(?:_start)?[^\]]*\]")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
SOURCE_MARKER_RE = re.compile(r"^---\s*SOURCE.*?---\s*$", re.IGNORECASE | re.MULTILINE)
HORIZONTAL_RULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    section: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def clean_text(raw_text: str) -> str:
    """Apply conservative cleanup while preserving Markdown structure."""
    text = raw_text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SOURCE_MARKER_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = CITE_RE.sub("", text)
    text = HORIZONTAL_RULE_RE.sub("", text)

    cleaned_lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_useful_content(text: str) -> bool:
    """Skip chunks that only contain headings, page numbers, or separators."""
    body_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        body_lines.append(stripped)

    body = "\n".join(body_lines)
    alnum = re.sub(r"[\W_]+", "", body, flags=re.UNICODE)
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", body)
    return len(alnum) >= 12 and len(words) >= 3


def split_by_headings(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("Documento completo", text)] if text else []

    sections: list[tuple[str, str]] = []

    preface = text[: matches[0].start()].strip()
    if preface:
        sections.append(("Introduccion", preface))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_title = match.group(2).strip()
        if section_text:
            sections.append((section_title, section_text))

    return sections


def split_long_block(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]

    lines = [line for line in block.split("\n") if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            parts.append("\n".join(current).strip())
            current = []
            current_len = 0

        if line_len > max_chars:
            for start in range(0, len(line), max_chars):
                parts.append(line[start : start + max_chars].strip())
            continue

        current.append(line)
        current_len += line_len

    if current:
        parts.append("\n".join(current).strip())

    return [part for part in parts if part]


def split_section(section_text: str, max_chars: int) -> list[str]:
    if len(section_text) <= max_chars:
        return [section_text]

    blocks = [block.strip() for block in re.split(r"\n{2,}", section_text) if block.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for block in blocks:
        if len(block) > max_chars:
            flush_current()
            chunks.extend(split_long_block(block, max_chars))
            continue

        projected_len = current_len + len(block) + (2 if current else 0)
        if current and projected_len > max_chars:
            flush_current()

        current.append(block)
        current_len += len(block) + (2 if current_len else 0)

    flush_current()
    return chunks


def build_chunks(input_dir: Path, max_chars: int) -> list[Chunk]:
    markdown_files = sorted(input_dir.glob("*.md"), key=lambda path: path.name.lower())
    chunks: list[Chunk] = []

    for markdown_file in markdown_files:
        text = clean_text(markdown_file.read_text(encoding="utf-8"))
        if not text:
            continue

        source = markdown_file.as_posix()
        sections = split_by_headings(text)

        for section_title, section_text in sections:
            for section_chunk in split_section(section_text, max_chars):
                if not has_useful_content(section_chunk):
                    continue

                chunk_id = f"C{len(chunks) + 1:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=source,
                        section=section_title,
                        text=section_chunk,
                    )
                )

    return chunks


def render_markdown(chunks: list[Chunk], input_dir: Path, max_chars: int) -> str:
    lines: list[str] = [
        "# Base de conocimiento segmentada",
        "",
        f"Fuente: `{input_dir.as_posix()}`",
        f"Total de chunks: {len(chunks)}",
        f"Tamano maximo objetivo por chunk: {max_chars} caracteres",
        "",
        "Uso sugerido: incluir este archivo como contexto del prompt de sistema para que el LLM responda solo con base en la informacion documentada.",
        "",
        "---",
        "",
    ]

    for chunk in chunks:
        lines.extend(
            [
                f"## {chunk.chunk_id} | {Path(chunk.source).name} | {chunk.section}",
                "",
                f"**Fuente:** `{chunk.source}`",
                f"**Seccion:** {chunk.section}",
                f"**Caracteres:** {chunk.char_count}",
                "",
                chunk.text,
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una base de conocimiento en Markdown segmentada en chunks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Carpeta con archivos Markdown fuente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHUNKS_FILE,
        help="Archivo Markdown de salida.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Tamano maximo objetivo de cada chunk en caracteres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"No existe la carpeta de entrada: {args.input_dir}")

    if args.max_chars < 500:
        raise SystemExit("--max-chars debe ser igual o mayor a 500")

    chunks = build_chunks(args.input_dir, args.max_chars)
    if not chunks:
        raise SystemExit(f"No se encontraron textos Markdown en: {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_text = render_markdown(chunks, args.input_dir, args.max_chars)
    args.output.write_text(output_text, encoding="utf-8")

    total_chars = sum(chunk.char_count for chunk in chunks)
    print(f"Chunks generados: {len(chunks)}")
    print(f"Caracteres utiles: {total_chars}")
    print(f"Archivo de salida: {args.output}")


if __name__ == "__main__":
    main()

