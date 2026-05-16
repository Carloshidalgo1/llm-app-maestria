"""Utilidades para parsear chunks Markdown de la base de conocimiento."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


DOCUMENTAL_KNOWLEDGE_TOOL_NAME = "base_documental_carnicos"
CHUNK_HEADING_RE = re.compile(r"^## (C\d{4}) \| (.+)$", re.MULTILINE)
SOURCE_RE = re.compile(r"\*\*Fuente:\*\* `([^`]+)`")


@dataclass(frozen=True)
class KnowledgeChunk:
    """Fragmento recuperable de la base documental."""

    chunk_id: str
    title: str
    source: str
    text: str

    def render(self, max_chars: int = 1800) -> str:
        """Presenta el fragmento con fuente y recorte controlado."""
        trimmed_text = self.text.strip()
        if len(trimmed_text) > max_chars:
            trimmed_text = trimmed_text[: max_chars - 3].rstrip() + "..."

        return "\n".join(
            [
                f"### {self.chunk_id} | {self.title}",
                f"Fuente: {self.source}",
                trimmed_text,
            ]
        )



def parse_knowledge_chunks(knowledge_base: str) -> list[KnowledgeChunk]:
    """Divide el Markdown consolidado en chunks recuperables."""
    matches = list(CHUNK_HEADING_RE.finditer(knowledge_base))
    if not matches:
        return _fallback_paragraph_chunks(knowledge_base)

    chunks: list[KnowledgeChunk] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(knowledge_base)
        raw_chunk = knowledge_base[start:end].strip()
        chunk_id = match.group(1)
        title = match.group(2).strip()
        source = _extract_source(raw_chunk) or title
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                title=title,
                source=source,
                text=raw_chunk,
            )
        )
    return chunks


def _extract_source(raw_chunk: str) -> Optional[str]:
    match = SOURCE_RE.search(raw_chunk)
    return match.group(1) if match else None


def _fallback_paragraph_chunks(knowledge_base: str) -> list[KnowledgeChunk]:
    """Permite usar la herramienta aunque la base no tenga encabezados C0001."""
    paragraphs = [part.strip() for part in knowledge_base.split("\n\n") if part.strip()]
    return [
        KnowledgeChunk(
            chunk_id=f"P{index:04d}",
            title="Parrafo recuperado",
            source="base de conocimiento",
            text=paragraph,
        )
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


