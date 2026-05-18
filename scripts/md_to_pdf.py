"""
Convierte docs/informe_modulo_2.md a PDF usando PyMuPDF (fitz.Story).

Uso:
    .venv/Scripts/python.exe scripts/md_to_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = PROJECT_ROOT / "docs" / "informe_modulo_2.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "informe_modulo_2.pdf"

PAGE_RECT = fitz.paper_rect("A4")
MARGIN = 54  # ~1.9 cm
CONTENT_RECT = PAGE_RECT + (MARGIN, MARGIN, -MARGIN, -MARGIN)

CSS = """
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    color: #1a1a1a;
    line-height: 1.55;
}
h1 {
    font-size: 17pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 0;
    margin-bottom: 8pt;
    border-bottom: 1.5pt solid #333;
    padding-bottom: 4pt;
}
h2 {
    font-size: 13pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 16pt;
    margin-bottom: 5pt;
    border-bottom: 0.5pt solid #aaa;
    padding-bottom: 2pt;
}
h3 {
    font-size: 11pt;
    font-weight: bold;
    color: #333;
    margin-top: 10pt;
    margin-bottom: 3pt;
}
p {
    margin: 0 0 6pt 0;
}
pre {
    background: #f5f5f5;
    border: 0.5pt solid #ccc;
    border-radius: 3pt;
    padding: 6pt 8pt;
    font-size: 8.5pt;
    margin: 6pt 0;
}
code {
    font-size: 8.5pt;
    background: #f0f0f0;
    padding: 0 2pt;
}
ul, ol {
    margin: 4pt 0 6pt 0;
    padding-left: 18pt;
}
li {
    margin-bottom: 2pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-size: 9.5pt;
}
th {
    background: #e8e8e8;
    font-weight: bold;
    padding: 4pt 6pt;
    border: 0.5pt solid #aaa;
    text-align: left;
}
td {
    padding: 3pt 6pt;
    border: 0.5pt solid #ccc;
    vertical-align: top;
}
hr {
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 10pt 0;
}
.meta {
    font-size: 9.5pt;
    color: #555;
    margin-bottom: 4pt;
}
strong { font-weight: bold; }
em { font-style: italic; }
"""


# ---------------------------------------------------------------------------
# Markdown -> HTML conversion
# ---------------------------------------------------------------------------

def md_to_html(md: str) -> str:
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]

        # --- fenced code block ---
        if line.startswith("```"):
            close_lists()
            lang = line[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(_escape(lines[i]))
                i += 1
            html_parts.append(
                f'<pre><code class="{lang}">' + "\n".join(code_lines) + "</code></pre>"
            )
            i += 1
            continue

        # --- horizontal rule ---
        if re.match(r"^---+\s*$", line):
            close_lists()
            html_parts.append("<hr/>")
            i += 1
            continue

        # --- headings ---
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            close_lists()
            level = len(m.group(1))
            content = _inline(m.group(2))
            html_parts.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        # --- table ---
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|?[\s\-|:]+\|", lines[i + 1]):
            close_lists()
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            html_parts.append(_parse_table(table_lines))
            continue

        # --- unordered list ---
        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            if not in_ul:
                close_lists()
                html_parts.append("<ul>")
                in_ul = True
            html_parts.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # --- ordered list ---
        m = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if m:
            if not in_ol:
                close_lists()
                html_parts.append("<ol>")
                in_ol = True
            html_parts.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # --- blank line ---
        if not line.strip():
            close_lists()
            i += 1
            continue

        # --- paragraph (bold meta lines like **key:** value) ---
        close_lists()
        text = _inline(line)
        html_parts.append(f"<p>{text}</p>")
        i += 1

    close_lists()
    body = "\n".join(html_parts)
    return f"<html><body>{body}</body></html>"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    # Escape HTML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # inline code
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    # bold+italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    # bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def _parse_table(table_lines: list[str]) -> str:
    def split_row(row: str) -> list[str]:
        row = row.strip().strip("|")
        return [cell.strip() for cell in row.split("|")]

    rows = [split_row(line) for line in table_lines if not re.match(r"^\|?[\s\-|:]+\|", line)]
    if not rows:
        return ""

    header = rows[0]
    body_rows = rows[1:]

    th_cells = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
    thead = f"<thead><tr>{th_cells}</tr></thead>"

    tbody_rows = ""
    for row in body_rows:
        td_cells = "".join(f"<td>{_inline(cell)}</td>" for cell in row)
        tbody_rows += f"<tr>{td_cells}</tr>"
    tbody = f"<tbody>{tbody_rows}</tbody>"

    return f"<table>{thead}{tbody}</table>"


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf(html: str, output_path: Path) -> None:
    import io

    story = fitz.Story(html=html, user_css=CSS)
    buf = io.BytesIO()

    writer = fitz.DocumentWriter(buf)
    more = True
    while more:
        device = writer.begin_page(PAGE_RECT)
        more, _ = story.place(CONTENT_RECT)
        story.draw(device)
        writer.end_page()
    writer.close()

    output_path.write_bytes(buf.getvalue())


def main() -> None:
    if not INPUT_MD.exists():
        print(f"ERROR: No se encontro {INPUT_MD}", file=sys.stderr)
        sys.exit(1)

    md_content = INPUT_MD.read_text(encoding="utf-8")
    html_content = md_to_html(md_content)
    render_pdf(html_content, OUTPUT_PDF)
    print(f"PDF generado: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
