from pathlib import Path

from carnicos_kb.chunking import build_chunks, clean_text, render_markdown, split_by_headings


def test_clean_text_removes_noise() -> None:
    raw = """
--- SOURCE: test.pdf ---

# Titulo [cite_start]

Texto con cita [cite: 1].

<!-- comentario -->

---
"""

    cleaned = clean_text(raw)

    assert "SOURCE" not in cleaned
    assert "[cite" not in cleaned
    assert "comentario" not in cleaned
    assert "Texto con cita" in cleaned


def test_split_by_headings_preserves_sections() -> None:
    sections = split_by_headings("# Historia\nTexto A\n\n## Contacto\nTexto B")

    assert sections == [
        ("Historia", "# Historia\nTexto A"),
        ("Contacto", "## Contacto\nTexto B"),
    ]


def test_build_chunks_from_markdown_files(tmp_path: Path) -> None:
    source_file = tmp_path / "empresa.md"
    source_file.write_text(
        "# Historia\nAlimentos Carnicos tiene trayectoria empresarial.\n\n"
        "## Contacto\nLa sede principal esta en Yumbo, Valle del Cauca.",
        encoding="utf-8",
    )

    chunks = build_chunks(tmp_path, max_chars=500)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "C0001"
    assert chunks[0].source.endswith("empresa.md")
    assert "trayectoria empresarial" in chunks[0].text


def test_render_markdown_contains_metadata(tmp_path: Path) -> None:
    source_file = tmp_path / "empresa.md"
    source_file.write_text("# Historia\nContenido util de prueba.", encoding="utf-8")

    chunks = build_chunks(tmp_path, max_chars=500)
    output = render_markdown(chunks, tmp_path, max_chars=500)

    assert "# Base de conocimiento segmentada" in output
    assert "Total de chunks: 1" in output
    assert "**Fuente:**" in output

