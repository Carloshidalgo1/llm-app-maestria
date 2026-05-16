"""Utilidades pequenas para busqueda lexica en texto del proyecto."""

from __future__ import annotations

import re
import unicodedata


SPANISH_STOPWORDS = {
    "a",
    "al",
    "con",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "donde",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "para",
    "por",
    "que",
    "se",
    "su",
    "sus",
    "un",
    "una",
    "y",
}


def normalize_for_search(text: str) -> str:
    """Convierte texto libre en una forma estable para comparar palabras.

    La normalizacion elimina tildes, pasa a minusculas y reemplaza signos por
    espacios. Es deliberadamente simple para que la recuperacion sea explicable
    durante la sustentacion.
    """
    without_accents = unicodedata.normalize("NFKD", text)
    ascii_text = without_accents.encode("ascii", "ignore").decode("ascii")
    lowercase_text = ascii_text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowercase_text).strip()


def meaningful_tokens(text: str) -> set[str]:
    """Extrae palabras utiles para comparar una consulta con documentos."""
    return {
        token
        for token in normalize_for_search(text).split()
        if len(token) >= 3 and token not in SPANISH_STOPWORDS
    }
