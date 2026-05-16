"""Estado de configuracion de LangSmith sin exponer secretos."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_LANGSMITH_PROJECT = "carnicos-kb-agent"


@dataclass(frozen=True)
class LangSmithStatus:
    """Resumen seguro para consola o Streamlit."""

    tracing_enabled: bool
    api_key_configured: bool
    project: str

    @property
    def ready(self) -> bool:
        return self.tracing_enabled and self.api_key_configured


def get_langsmith_status() -> LangSmithStatus:
    """Lee variables de entorno LangSmith sin imprimir la API key."""
    tracing_value = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    project = os.getenv("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT).strip()

    return LangSmithStatus(
        tracing_enabled=tracing_value in {"1", "true", "yes", "on"},
        api_key_configured=_looks_like_real_langsmith_key(api_key),
        project=project or DEFAULT_LANGSMITH_PROJECT,
    )


def _looks_like_real_langsmith_key(api_key: str) -> bool:
    if not api_key:
        return False
    if "tu_api_key" in api_key or "placeholder" in api_key.lower():
        return False
    return api_key.startswith(("lsv2_", "ls__")) and len(api_key) > 20
