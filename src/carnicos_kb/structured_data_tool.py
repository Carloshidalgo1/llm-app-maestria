"""Herramienta LangChain para consultar datos estructurados y verificables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool

from .paths import DEFAULT_STRUCTURED_DATA_FILE
from .text_matching import meaningful_tokens, normalize_for_search


STRUCTURED_DATA_TOOL_NAME = "datos_estructurados_carnicos"


@dataclass(frozen=True)
class StructuredRecord:
    """Registro pequeno del archivo JSON de preguntas frecuentes."""

    record_id: str
    title: str
    answer: str
    keywords: tuple[str, ...]
    question_examples: tuple[str, ...]
    source: str
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw_record: dict[str, Any]) -> "StructuredRecord":
        """Construye un registro validando los campos minimos."""
        return cls(
            record_id=str(raw_record["id"]),
            title=str(raw_record["title"]),
            answer=str(raw_record["answer"]),
            keywords=tuple(str(item) for item in raw_record.get("keywords", [])),
            question_examples=tuple(
                str(item) for item in raw_record.get("question_examples", [])
            ),
            source=str(raw_record["source"]),
            notes=str(raw_record.get("notes", "")),
        )

    def searchable_text(self) -> str:
        """Une los campos que deben participar en la busqueda determinista."""
        return " ".join(
            [
                self.record_id,
                self.title,
                self.answer,
                self.source,
                self.notes,
                *self.keywords,
                *self.question_examples,
            ]
        )

    def render(self) -> str:
        """Presenta el registro como texto facil de leer por el LLM."""
        lines = [
            f"### {self.title}",
            self.answer,
            f"Fuente: {self.source}",
        ]
        if self.notes:
            lines.append(f"Nota: {self.notes}")
        return "\n".join(lines)


class StructuredDataStore:
    """Carga y consulta el JSON estructurado sin usar embeddings ni vectores."""

    def __init__(self, records: list[StructuredRecord]):
        if not records:
            raise ValueError("El archivo de datos estructurados no contiene registros.")
        self.records = records

    @classmethod
    def from_file(cls, data_path: Path) -> "StructuredDataStore":
        """Lee el archivo JSON usado por la herramienta determinista."""
        raw_data = json.loads(data_path.read_text(encoding="utf-8"))
        records = [
            StructuredRecord.from_mapping(record)
            for record in raw_data.get("records", [])
        ]
        return cls(records)

    def search(self, query: str, max_results: int = 3) -> str:
        """Busca respuestas concretas con coincidencia lexica explicable."""
        query = query.strip()
        tokens = meaningful_tokens(query)
        if not query or not tokens:
            return "No se recibio una consulta estructurada valida."

        ranked_records = sorted(
            (
                (self._score_record(query, tokens, record), record)
                for record in self.records
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        best_score = ranked_records[0][0]
        minimum_score = max(3, int(best_score * 0.45))
        matches = [
            record
            for score, record in ranked_records
            if score > 0 and score >= minimum_score
        ]
        if not matches:
            return (
                "No encontre un dato estructurado verificado para esa consulta. "
                "Usa la base documental o verifica en canales oficiales."
            )

        rendered_matches = [record.render() for record in matches[:max_results]]
        return "\n\n".join(rendered_matches)

    def _score_record(
        self,
        original_query: str,
        query_tokens: set[str],
        record: StructuredRecord,
    ) -> int:
        """Calcula un puntaje simple que puede explicarse en sustentacion."""
        normalized_query = normalize_for_search(original_query)
        normalized_keywords = normalize_for_search(
            " ".join([record.record_id, record.title, *record.keywords])
        )
        normalized_examples = normalize_for_search(" ".join(record.question_examples))
        normalized_all_text = normalize_for_search(record.searchable_text())
        keyword_tokens = set(normalized_keywords.split())
        example_tokens = set(normalized_examples.split())
        all_tokens = set(normalized_all_text.split())

        score = 0
        for token in query_tokens:
            if token in keyword_tokens:
                score += 5
            elif token in example_tokens:
                score += 3
            elif token in all_tokens:
                score += 1

        for phrase in [*record.keywords, *record.question_examples]:
            normalized_phrase = normalize_for_search(phrase)
            if normalized_phrase and _contains_complete_phrase(
                normalized_query,
                normalized_phrase,
            ):
                score += 8

        return score


def _contains_complete_phrase(text: str, phrase: str) -> bool:
    """Evita falsos positivos como encontrar `nit` dentro de `cunit`."""
    return f" {phrase} " in f" {text} "


def build_structured_data_tool(
    data_path: Optional[Path] = None,
) -> StructuredTool:
    """Crea la herramienta LangChain que el agente puede seleccionar."""
    selected_path = data_path or DEFAULT_STRUCTURED_DATA_FILE
    store = StructuredDataStore.from_file(selected_path)

    def buscar_datos_estructurados(query: str) -> str:
        """Devuelve datos concretos: telefonos, sedes, NIT, empleo o visitas."""
        return store.search(query)

    return StructuredTool.from_function(
        name=STRUCTURED_DATA_TOOL_NAME,
        func=buscar_datos_estructurados,
        description=(
            "Usar para preguntas concretas sobre telefonos, lineas de atencion, "
            "sedes, puntos de venta, NIT, fecha de creacion, sitio web, empleo, "
            "visitas a planta u horarios documentados."
        ),
    )
